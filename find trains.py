"""
Finds pulse trains among the detections and recovers pings the network missed.

An active sonar transmits on a schedule, so its pings arrive at a near-constant pulse repetition
interval. That interval is what defines a train here, and nothing else: a transmitter may send any
mixture of waveforms - CW, LFM and HFM in any order, changing from ping to ping - and those pulses
still belong to one train because they arrive on one clock. Pulse type, band and duration are
reported for each train but take no part in deciding membership. That regularity is information the
per-window detector cannot use: it looks at five seconds at a time and has no idea that a similar
pulse arrived ten seconds earlier. Grouping detections into trains recovers that context.

The recovery works because a gap in a train is not a blind search. The train fixes when the ping
should have arrived, in which band, with which sweep and duration - so the missing ping can be
looked for with a replica of itself. Matched filtering against a known waveform is the optimal
detector for that waveform, and it reaches several dB below the level a general detector needs,
which is exactly where the missed pings are.

Nothing here assumes the pings repeat. A source with no regular interval simply produces no train,
no prediction and no recovery, and the output is then the input. The stage is opportunistic: it
improves recall on periodic sources and is inert on everything else.

Two limits worth keeping in mind:

  - Only misses inside a train can be recovered. An isolated ping, the first ping of a sequence, or
    a source that changes its interval has no context to extrapolate from and stays missed.
  - Recovery is circular by construction: it finds what it predicted. The threshold is therefore
    strict, recovered pings are written with their score and never merged silently into the
    detector's own output, and recall should be quoted both with and without this stage.

Run:
    python find_trains.py --wav recording.wav --detections recording_detections_refined.csv
    python find_trains.py --wav rec.wav --detections rec_refined.csv --plot --min-snr 8
"""

import argparse
import csv
import os

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

from data_config import FS as DEFAULT_FS, NPERSEG, NOVERLAP
from detect_pulses import read_wav
from plot_style import FIG_WIDE, TYPE_COLORS, REFERENCE, ACCENT, type_name, hms, hms_axis, finish, save


def read_detections(path):

    """
    Reads a detections CSV and returns one dict per pulse in physical units, preferring the
    refined columns when refine_detections.py has been run and falling back to the network's own
    values otherwise.

    ----------

    Parameters:
        path (str) - path to the detections CSV.

    Returns:
        dets (list) - one dict per detection with keys id, t_start, t_stop, duration, f1, f2,
                      type, direction, refined.
    """

    dets = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            refined = row.get("refined", "") == "1"
            g = (lambda key, fallback:
                 float(row[key]) if refined and row.get(key, "") not in ("", None)
                 else float(row[fallback]))
            t_start = g("refined_t_start_s", "t_start_s")
            t_stop = g("refined_t_stop_s", "t_stop_s")
            f1 = g("refined_f1_hz", "f_start_hz")
            f2 = g("refined_f2_hz", "f_stop_hz")
            kind = (row.get("refined_type") or row["type"]) if refined else row["type"]
            # A CW pulse has no direction and no bandwidth: its two endpoints estimate the same
            # tone, so any difference between them is measurement noise. Deriving a direction
            # from that noise would label successive pings of one CW train "up" and "down" at
            # random and tear the train apart, so the type decides instead.
            if kind == "cw":
                direction = "flat"
            else:
                direction = row.get("refined_direction", "") or \
                            ("up" if f2 > f1 else "down" if f2 < f1 else "flat")
            dets.append({"id": int(row["pulse_id"]), "t_start": t_start, "t_stop": t_stop,
                         "duration": max(t_stop - t_start, 1e-3), "f1": f1, "f2": f2,
                         "type": kind, "direction": direction, "refined": refined})
    return dets


def _grid_members(dets, period, anchor, tol):

    """
    Returns the detections that sit on the grid of the given period anchored at the given time,
    within the relative tolerance, keeping at most one detection per slot (the closest).

    ----------

    Parameters:
        dets (list) - all detections.
        period (float) - candidate repetition interval in seconds.
        anchor (float) - a time the grid passes through.
        tol (float) - relative tolerance, as a fraction of the period.

    Returns:
        members (list) - the detections on the grid, sorted by start time.
    """

    best = {}
    for d in dets:
        k = round((d["t_start"] - anchor) / period)
        err = abs(d["t_start"] - (anchor + k * period))
        if err > tol * period:
            continue
        if k not in best or err < best[k][0]:
            best[k] = (err, d)
    return [d for _e, d in sorted(best.values(), key=lambda p: p[1]["t_start"])]


def _train_quality(members, period, tol):

    """
    Scores a candidate train. A set of arrivals is a train when it is REGULAR, which means two
    things at once: the arrivals sit tightly on the grid, and they fill most of the slots the grid
    implies. Tightness alone is not enough, because a sparse grid can always be laid over a few
    scattered arrivals; occupancy alone is not enough either, because a short period fills its
    slots trivially. Both are therefore required.

    Tightness is measured on the circle, since phases wrap: 0.01 and 0.99 are adjacent, not far
    apart. Each arrival's phase within its slot is mapped to a unit vector and the length of their
    mean is taken, which is 1 when every arrival sits exactly on a grid line and near 0 when they
    are scattered.

    ----------

    Parameters:
        members (list) - detections on the grid.
        period (float) - the candidate interval in seconds.
        tol (float) - relative tolerance (unused directly, kept for symmetry with _grid_members).

    Returns:
        scatter (float) - circular spread of the phases, in units of one slot.
        occupancy (float) - fraction of the implied slots that contain an arrival.
        n_slots (int) - number of slots between the first and last member.
    """

    times = np.array([d["t_start"] for d in members], dtype=float)
    if len(times) < 2:
        return 1e9, 0.0, 0
    span = times[-1] - times[0]
    n_slots = int(round(span / period)) + 1
    phase = 2 * np.pi * (((times - times[0]) / period) % 1.0)
    R = float(np.hypot(np.mean(np.cos(phase)), np.mean(np.sin(phase))))
    scatter = float(np.sqrt(-2.0 * np.log(max(R, 1e-12)))) / (2 * np.pi)
    return scatter, len(times) / max(n_slots, 1), n_slots


def build_trains(dets, tol=0.1, min_pulses=4, max_scatter=0.06, min_occupancy=0.5,
                 min_period=0.5, freq_tol=None):

    """
    Finds the pulse trains among the detections. A train is defined by its REPETITION INTERVAL and
    by nothing else: a transmitter may send any mixture of waveforms - CW, LFM and HFM in any
    order, changing from ping to ping - and those pulses still belong to one train because they
    arrive on one clock. Pulse type, band and duration are therefore recorded and reported, but
    they are not used to decide membership.

    The search enumerates candidate intervals from the gaps between pairs of detections, since the
    true interval is the gap between some pair even when pings in between were missed. For each
    candidate the detections lying on that grid are collected and the set is accepted only if it
    is regular in both senses described in _train_quality. Candidates are then taken in order of
    how many pulses they explain, each detection joining at most one train, so the strongest
    periodicity claims its pulses first and weaker coincidences cannot re-use them.

    ----------

    Parameters:
        dets (list) - detections from read_detections.
        tol (float) - default 0.1. Relative tolerance on the interval.
        min_pulses (int) - default 4. Fewest pulses required to call a set a train.
        max_scatter (float) - default 0.06. Largest accepted phase spread, in units of one slot.
        min_occupancy (float) - default 0.5. Smallest fraction of the implied slots that must
                                contain a pulse; this is what stops a short interval from
                                "explaining" arrivals by offering more slots than it fills.
        min_period (float) - default 0.5. Shortest interval considered, in seconds.
        freq_tol (float or None) - default None (any band). When given, only detections whose
                                   bands lie within this many Hz of each other may share a train.

    Returns:
        trains (list) - dicts with keys members, interval, scatter, occupancy.
        loners (list) - detections belonging to no train.
    """

    dets = sorted(dets, key=lambda d: d["t_start"])
    times = [d["t_start"] for d in dets]
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    if len(dets) < min_pulses or span <= 0:
        return [], list(dets)

    # Candidate intervals: every gap between a pair of detections. A train with missed pings still
    # shows its interval as the gap between two survivors, which is why pairs are used rather than
    # successive gaps.
    cands = set()
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            p = times[j] - times[i]
            if p > 0.5 * span:
                break                      # times are sorted, so later j only give larger gaps
            if p >= min_period:
                cands.add(round(p, 3))

    proposals = []
    for period in sorted(cands):
        seen_anchor = set()
        for anchor_det in dets:
            # anchors differing by a whole number of periods describe the same grid
            key = round(((anchor_det["t_start"] - times[0]) / period) % 1.0, 3)
            if key in seen_anchor:
                continue
            seen_anchor.add(key)

            members = _grid_members(dets, period, anchor_det["t_start"], tol)
            if len(members) < min_pulses:
                continue
            if freq_tol is not None:
                lo = min(min(d["f1"], d["f2"]) for d in members)
                hi = max(max(d["f1"], d["f2"]) for d in members)
                if hi - lo > freq_tol:
                    continue
            scatter, occ, n_slots = _train_quality(members, period, tol)
            if scatter > max_scatter or occ < min_occupancy or n_slots < min_pulses:
                continue
            proposals.append({"members": members, "interval": period,
                              "scatter": scatter, "occupancy": occ})

    # Prefer the train explaining most pulses; break ties toward the tighter grid, then the
    # shorter interval, which avoids accepting a multiple of the true period.
    proposals.sort(key=lambda p: (-len(p["members"]), p["scatter"], p["interval"]))

    trains, claimed = [], set()
    for p in proposals:
        free = [d for d in p["members"] if id(d) not in claimed]
        if len(free) < min_pulses:
            continue
        scatter, occ, n_slots = _train_quality(free, p["interval"], tol)
        if scatter > max_scatter or occ < min_occupancy:
            continue
        for d in free:
            claimed.add(id(d))
        trains.append({"members": free, "interval": p["interval"],
                       "scatter": scatter, "occupancy": occ})

    trains.sort(key=lambda t: t["members"][0]["t_start"])
    loners = [d for d in dets if id(d) not in claimed]
    return trains, loners


def train_composition(train):

    """
    Describes what a train contains: how many pulses of each waveform, and the band they span.
    Membership does not depend on any of this - it is reported so that a mixed train can be read
    at a glance.

    ----------

    Parameters:
        train (dict) - one train from build_trains.

    Returns:
        (str) - a short description, e.g. 'LFM x 7, CW x 6, 4980-7020 Hz'.
    """

    members = train["members"]
    counts = {}
    for d in members:
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    parts = [f"{type_name(k)} x {n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    lo = min(min(d["f1"], d["f2"]) for d in members)
    hi = max(max(d["f1"], d["f2"]) for d in members)
    return f"{', '.join(parts)}, {lo:.0f}-{hi:.0f} Hz"


def predict_gaps(train, duration, tol=0.1, edge_margin=0.0, all_detections=None):

    """
    Lists the times at which the train should have contained a ping but does not. The grid runs
    from the first to the last member of the train, so only interior gaps are predicted: the stage
    never invents pings before a train starts or after it ends, where there is no evidence a
    transmission was taking place at all.

    ----------

    Parameters:
        train (dict) - one train from build_trains.
        duration (float) - length of the recording in seconds.
        tol (float) - default 0.1. Relative tolerance when matching a slot to a member.
        edge_margin (float) - default 0.0. Seconds to stay clear of the recording edges.
        all_detections (list or None) - default None (use the train's own members). Every
                                        detection in the recording; a slot occupied by any of
                                        them is not a gap, so a pulse already found cannot be
                                        reported a second time as a recovery.

    Returns:
        gaps (list) - predicted start times in seconds.
    """

    members = train["members"]
    period = train["interval"]
    t0, t1 = members[0]["t_start"], members[-1]["t_start"]

    # A slot is only a gap if NO detection sits there - not merely no member of this train. A
    # detection that failed to join the train (its band or duration drifted, or it belongs to
    # another source) still means a pulse was already found at that time, and "recovering" it
    # would report the same physical pulse twice.
    have = np.array([d["t_start"] for d in (all_detections if all_detections else members)])

    gaps = []
    n = int(round((t1 - t0) / period))
    for k in range(1, n):
        t = t0 + k * period
        if np.min(np.abs(have - t)) < tol * period:
            continue
        if t < edge_margin or t > duration - edge_margin:
            continue
        gaps.append(float(t))
    return gaps


def train_replicas(train, fs):

    """
    Builds one expected waveform per pulse type present in the train. A train may mix waveforms,
    so a missing ping's type is not known in advance: the search has to try each waveform the
    transmitter is known to use and let the scores decide which one is actually there.

    Each replica uses the median duration and band of the members OF THAT TYPE, since a
    transmitter's CW and FM pulses need not share a band or a length.

    ----------

    Parameters:
        train (dict) - one train from build_trains.
        fs (int) - sampling frequency in Hz.

    Returns:
        replicas (list) - one dict per type, with keys type, duration, f1, f2, wave, and the
                          observed band spread of that type within the train.
    """

    replicas = []
    for kind in sorted({d["type"] for d in train["members"]}):
        same = [d for d in train["members"] if d["type"] == kind]
        duration = float(np.median([d["duration"] for d in same]))
        f1 = float(np.median([d["f1"] for d in same]))
        f2 = float(np.median([d["f2"] for d in same]))

        t = np.arange(max(int(round(duration * fs)), 8)) / fs
        if kind == "cw" or abs(f2 - f1) < 1.0:
            wave = np.sin(2 * np.pi * f1 * t)
        else:
            wave = sps.chirp(t, f1, t[-1], f2,
                             method="hyperbolic" if kind == "hfm" else "linear")
        wave = wave * sps.windows.tukey(len(wave), 0.2)
        wave = wave / (np.linalg.norm(wave) + 1e-12)
        # How much this waveform's band moves from ping to ping. A transmitter that keeps a
        # fixed band gives a spread of a few hundred Hz (measurement noise); one that changes
        # frequency between pings gives much more, and the search then has to scan rather than
        # assume the median.
        lo = min(min(d["f1"], d["f2"]) for d in same)
        hi = max(max(d["f1"], d["f2"]) for d in same)
        spread = float(hi - lo) - abs(f2 - f1)
        replicas.append({"type": kind, "duration": duration, "f1": f1, "f2": f2, "wave": wave,
                         "spread": max(spread, 0.0), "lo": float(lo), "hi": float(hi)})
    return replicas


def matched_filter(data, fs, replica, t_predicted, search, f1, f2, pad_hz=1000.0,
                   duration=None, kind="lfm"):

    """
    Looks for the train's pulse near a predicted time and returns how well it is matched.

    The obvious approach - correlating a replica waveform against the signal - is not usable here.
    Coherent matched filtering is optimal only when the replica matches the true waveform closely,
    and its response falls away far faster than the parameters are known: a replica built from the
    train's median band loses most of its score against a pulse whose endpoints differ by a few
    tens of hertz, which is smaller than the measurement error on those endpoints. The detection
    would then depend on the accuracy of the estimate rather than on whether a pulse is there.

    The match is therefore made in the spectrogram instead. The expected sweep is drawn as a track
    through the time-frequency plane, the energy along that track is summed, and the result is
    compared with the energy of the same track displaced elsewhere in the band. The score is the
    excess in dB of the track over that background, so it measures how much louder the predicted
    sweep is than the surrounding noise, and it degrades gracefully with an imperfect estimate.

    ----------

    Parameters:
        data (ndarray) - the full resampled signal.
        fs (int) - sampling frequency in Hz.
        replica (ndarray) - unused, kept so callers may still pass a waveform.
        t_predicted (float) - predicted pulse start in seconds.
        search (float) - half-width of the search window in seconds.
        f1 (float) - expected start frequency in Hz.
        f2 (float) - expected stop frequency in Hz.
        pad_hz (float) - default 1000.0. Margin around the expected band kept in the crop.
        duration (float or None) - expected pulse duration in seconds.
        kind (str) - default 'lfm'. Sweep law of the train, 'cw', 'lfm' or 'hfm'.

    Returns:
        score (float) - excess of the track over the local background, in dB.
        t_found (float) - the pulse start the best track corresponds to, in seconds.
    """

    from detect_pulses import db

    if duration is None:
        duration = max(len(replica) / fs, 0.05)

    t0 = max(0.0, t_predicted - search)
    t1 = min(len(data) / fs, t_predicted + search + duration)
    seg = np.asarray(data[int(t0 * fs):int(t1 * fs)], dtype=float)
    if seg.size < NPERSEG * 2:
        return 0.0, t_predicted

    f_axis, t_axis, zxx = sps.stft(seg, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    S = db(zxx)
    del zxx
    df = float(f_axis[1] - f_axis[0])
    dt = float(t_axis[1] - t_axis[0])
    n_cols = max(int(round(duration / dt)), 2)
    if n_cols >= S.shape[1]:
        return 0.0, t_predicted

    def track_energy(start_col, shift_hz):

        """Mean level along the expected sweep, displaced by shift_hz."""

        tau = np.linspace(0.0, 1.0, n_cols)
        if kind == "hfm" and min(f1, f2) > 1.0 and abs(f2 - f1) > 1.0:
            track = 1.0 / (1.0 / f1 + (1.0 / f2 - 1.0 / f1) * tau)
        else:
            track = f1 + (f2 - f1) * tau
        rows = np.round((track + shift_hz - f_axis[0]) / df).astype(int)
        cols = start_col + np.arange(n_cols)
        ok = (rows >= 0) & (rows < S.shape[0]) & (cols < S.shape[1])
        if ok.sum() < 0.8 * n_cols:
            return None
        return float(np.mean(S[rows[ok], cols[ok]]))

    # background: the same track shifted well away in frequency, on both sides
    shifts = [s for s in (-4 * pad_hz, -2 * pad_hz, 2 * pad_hz, 4 * pad_hz)
              if 0 < min(f1, f2) + s and max(f1, f2) + s < fs / 2]

    best_score, best_col = -np.inf, 0
    for start_col in range(0, S.shape[1] - n_cols):
        on = track_energy(start_col, 0.0)
        if on is None:
            continue
        off = [track_energy(start_col, s) for s in shifts]
        off = [v for v in off if v is not None]
        if not off:
            continue
        score = on - float(np.median(off))
        if score > best_score:
            best_score, best_col = score, start_col

    if not np.isfinite(best_score):
        return 0.0, t_predicted
    return float(best_score), float(t0 + best_col * dt)


def plot_trains(trains, loners, recovered, duration, path, fs):

    """
    Draws the trains in the same two-panel layout as the detection and refinement overviews, so
    the three stages can be read against one another: the upper panel is frequency against time,
    the lower panel a per-pulse score. Detected pings are filled markers, pings recovered at a
    predicted time are open markers, and detections belonging to no train are grey crosses. A thin
    line joins the members of each train, which is what makes the repetition visible without
    giving every train its own row.

    The lower panel shows the match score of the recovered pings, in dB above the surrounding
    noise. Detected pings have no such score - they came from the network, not from this search -
    so they appear on the zero line as a reference.

    ----------

    Parameters:
        trains (list) - trains from build_trains.
        loners (list) - detections in no train.
        recovered (list) - recovered ping dicts with keys train, t_start, score.
        duration (float) - length of the recording in seconds.
        path (str) - output file path.
        fs (int) - sampling frequency in Hz, used to set the frequency axis to Nyquist.

    Returns:
        None
    """

    fig, (ax_f, ax_s) = plt.subplots(2, 1, figsize=FIG_WIDE, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    seen = set()
    for i, train in enumerate(trains):
        members = train["members"]

        # A train may mix waveforms, so each pulse takes the colour of its OWN type and the line
        # joining them shows that they nonetheless belong to one train.
        colour = "0.45"

        # thin line through the band centres shows the train as one object
        mids = [(0.5 * (d["t_start"] + d["t_stop"]), 0.5 * (d["f1"] + d["f2"])) for d in members]
        got = sorted([r for r in recovered if r["train"] == i], key=lambda r: r["t_start"])
        allpts = sorted(mids + [(0.5 * (r["t_start"] + r["t_stop"]), 0.5 * (r["f1"] + r["f2"]))
                                for r in got])
        ax_f.plot([p[0] for p in allpts], [p[1] for p in allpts], "-", color=colour,
                  lw=0.8, alpha=0.55)

        for d in members:
            lo, hi = sorted((d["f1"], d["f2"]))
            mid = 0.5 * (d["t_start"] + d["t_stop"])
            c = TYPE_COLORS.get(d["type"], REFERENCE)
            label = type_name(d["type"]) if d["type"] not in seen else None
            seen.add(d["type"])
            ax_f.plot([mid, mid], [lo, hi], color=c, lw=2.0, solid_capstyle="round", label=label)
            ax_f.plot([mid], [0.5 * (lo + hi)], marker="o", ms=3.5, color=c)
            ax_s.plot([mid], [0.0], marker="o", ms=3, color=c, alpha=0.6)

        for r in got:
            lo, hi = sorted((r["f1"], r["f2"]))
            mid = 0.5 * (r["t_start"] + r["t_stop"])
            ax_f.plot([mid, mid], [lo, hi], color=TYPE_COLORS.get(r["type"], REFERENCE),
                      lw=1.6, ls="--", alpha=0.9)
            ax_f.plot([mid], [0.5 * (lo + hi)], marker="o", ms=6, mfc="none", mew=1.4,
                      color=ACCENT, label="Recovered" if "rec" not in seen else None)
            seen.add("rec")
            ax_s.plot([mid, mid], [0.0, r["score"]], color=ACCENT, lw=1.4)
            ax_s.plot([mid], [r["score"]], marker="o", ms=3.5, color=ACCENT)

    for d in loners:
        lo, hi = sorted((d["f1"], d["f2"]))
        mid = 0.5 * (d["t_start"] + d["t_stop"])
        ax_f.plot([mid], [0.5 * (lo + hi)], marker="x", ms=6, color="0.55",
                  label="No train" if "lone" not in seen else None)
        seen.add("lone")

    ax_f.set_xlim(0, max(duration, 1e-3))
    ax_f.set_ylim(0, fs / 2)
    title = f"{len(trains)} train(s) over {hms(duration, 0)}, {len(recovered)} ping(s) recovered"
    if trains:
        title += "\n" + "; ".join(f"train {i}: every {t['interval']:.2f} s, "
                                  f"{train_composition(t)}" for i, t in enumerate(trains[:3]))
        if len(trains) > 3:
            title += "; ..."
    finish(ax_f, None, "Frequency [Hz]", title, legend=bool(seen))
    ax_s.axhline(0.0, color=REFERENCE, lw=0.8)
    ax_s.set_ylim(bottom=0)
    hms_axis(ax_s)
    finish(ax_s, "Time [h:mm:ss]", "Match score [dB]", None)
    save(fig, path)


def main():

    """
    Parses arguments, builds trains from a detections CSV, tries to recover the missing pings by
    matched filtering, and writes the recovered pulses and the train assignments.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="the recording the detections came from")
    ap.add_argument("--detections", required=True,
                    help="CSV from detect_pulses.py, or the refined CSV")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <detections>_trains.csv)")
    ap.add_argument("--fs", type=int, default=DEFAULT_FS,
                    help="sampling frequency the detections were produced at")
    ap.add_argument("--min-pulses", type=int, default=4,
                    help="fewest pulses on a consistent interval to call a set a train")
    ap.add_argument("--min-occupancy", type=float, default=0.5,
                    help="smallest fraction of the slots implied by the interval that must "
                         "contain a pulse; stops a short interval from explaining arrivals "
                         "by offering more slots than it fills")
    ap.add_argument("--min-period", type=float, default=0.5,
                    help="shortest repetition interval considered, in seconds")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="relative tolerance of the repetition interval")
    ap.add_argument("--max-scatter", type=float, default=0.06,
                    help="largest accepted spread of arrival times around the repetition grid, "
                         "in units of one interval; guards against reading a period into "
                         "irregular arrivals (uniform scatter gives about 0.29 by this measure)")
    ap.add_argument("--freq-tol", type=float, default=None,
                    help="optional: require all pulses of a train to lie within this many Hz of "
                         "each other. Off by default, since a transmitter may change waveform "
                         "and band from ping to ping while keeping one interval")
    ap.add_argument("--search", type=float, default=None,
                    help="half-width in seconds of the window searched around a predicted ping "
                         "(default: 10%% of the repetition interval)")
    ap.add_argument("--min-snr", type=float, default=6.0,
                    help="how many dB the predicted sweep must stand above the surrounding noise "
                         "to count as a recovered ping; keep this strict, since the search "
                         "already knows what it is looking for")
    ap.add_argument("--no-recover", action="store_true",
                    help="only find trains, do not search for missing pings")
    ap.add_argument("--plot", action="store_true", help="save a figure of the trains")
    ap.add_argument("--outdir", default="trains")
    args = ap.parse_args()

    dets = read_detections(args.detections)
    data, duration = read_wav(args.wav, args.fs)
    print(f"{args.wav}: {duration:.1f} s | {len(dets)} detections")

    trains, loners = build_trains(dets, tol=args.tol, min_pulses=args.min_pulses,
                                  max_scatter=args.max_scatter,
                                  min_occupancy=args.min_occupancy,
                                  min_period=args.min_period, freq_tol=args.freq_tol)
    if not trains:
        print("No repeating trains found; every detection stands alone and nothing is predicted.")
    for i, train in enumerate(trains):
        members = train["members"]
        print(f"  train {i}: {len(members)} pulses, interval {train['interval']:.2f} s, "
              f"{hms(members[0]['t_start'])}-{hms(members[-1]['t_start'])}, "
              f"{train['occupancy'] * 100:.0f}% of slots filled")
        print(f"           contains {train_composition(train)}")
    if loners:
        print(f"  {len(loners)} detection(s) in no train")



    recovered, candidates = [], []
    if not args.no_recover:
        for i, train in enumerate(trains):
            gaps = predict_gaps(train, duration, tol=args.tol, all_detections=dets)
            if not gaps:
                continue
            replicas = train_replicas(train, args.fs)
            search = args.search if args.search is not None else 0.1 * train["interval"]
            for t_pred in gaps:
                # The waveform of a missing ping is unknown, so every waveform the train uses is
                # tried and the best match decides which one was there.
                tried = []
                for rep in replicas:
                    # A band that stays put needs one look; a band that moves between pings must
                    # be scanned, since the median tells us nothing about where THIS ping was.
                    if rep["spread"] > 400.0:
                        # The step has to suit the pulse, not the spread: a CW tone occupies one
                        # frequency bin, so a coarse step walks straight past it, while a wide
                        # sweep tolerates a much larger one. The number of looks is capped so the
                        # search stays quick.
                        bw = abs(rep["f2"] - rep["f1"])
                        step = max(2.0 * args.fs / NPERSEG, 0.25 * bw, rep["spread"] / 40.0)
                        shifts = np.arange(-rep["spread"] / 2, rep["spread"] / 2 + step, step)
                    else:
                        shifts = [0.0]
                    for sh in shifts:
                        score, t_found = matched_filter(
                            data, args.fs, rep["wave"], t_pred, search,
                            rep["f1"] + sh, rep["f2"] + sh,
                            duration=rep["duration"], kind=rep["type"])
                        tried.append((score, t_found,
                                      {**rep, "f1": rep["f1"] + sh, "f2": rep["f2"] + sh}))
                score, t_found, rep = max(tried, key=lambda x: x[0])
                if score < args.min_snr:
                    print(f"    gap at {hms(t_pred)} (train {i}): best {score:+.1f} dB "
                          f"({type_name(rep['type'])} near {min(rep['f1'], rep['f2']):.0f} Hz) "
                          f"-> nothing there")
                    continue
                alt = ""
                others = [x for x in tried if x[2]["type"] != rep["type"]]
                if others:
                    best_other = max(others, key=lambda x: x[0])
                    alt = (f", best {type_name(best_other[2]['type'])} scored "
                           f"{best_other[0]:+.1f} dB")
                candidates.append({"train": i, "t_start": t_found,
                                   "t_stop": t_found + rep["duration"],
                                   "f1": rep["f1"], "f2": rep["f2"],
                                   "type": rep["type"], "score": score,
                                   "t_predicted": t_pred, "duration": rep["duration"],
                                   "alt": alt})

    # Several trains may predict a ping at the same instant - unavoidable when they share a
    # grid, as interleaved waveforms of one transmitter do. Each searched with its own replica,
    # so the highest score identifies which waveform is actually there; taking whichever train
    # asked first would label the pulse by accident instead.
    for cand in sorted(candidates, key=lambda c: -c["score"]):
        if any(abs(cand["t_start"] - r["t_start"]) < 0.5 * cand["duration"] for r in recovered):
            continue
        recovered.append(cand)
    recovered.sort(key=lambda r: r["t_start"])

    for r in recovered:
        rivals = [c for c in candidates
                  if abs(c["t_start"] - r["t_start"]) < 0.5 * r["duration"] and c is not r]
        note = ""
        if rivals:
            best_rival = max(rivals, key=lambda c: c["score"])
            note = (f"; train {best_rival['train']} also matched it as "
                    f"{type_name(best_rival['type'])} at {best_rival['score']:+.1f} dB")
        print(f"    recovered at {hms(r['t_start'])}: {type_name(r['type'])} from train "
              f"{r['train']}, {r['score']:+.1f} dB over background"
              f"{r.get('alt', '')}{note}")

    path = args.out or os.path.splitext(args.detections)[0] + "_trains.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "train_id", "pulse_id", "t_start_s", "t_stop_s",
                    "t_start_hms", "f1_hz", "f2_hz", "type", "interval_s", "match_score"])
        for i, train in enumerate(trains):
            for d in train["members"]:
                w.writerow(["detected", i, d["id"], f"{d['t_start']:.3f}",
                            f"{d['t_stop']:.3f}", hms(d["t_start"]), f"{d['f1']:.1f}",
                            f"{d['f2']:.1f}", d["type"], f"{train['interval']:.3f}", ""])
        for d in loners:
            w.writerow(["detected", "", d["id"], f"{d['t_start']:.3f}",
                        f"{d['t_stop']:.3f}", hms(d["t_start"]), f"{d['f1']:.1f}",
                        f"{d['f2']:.1f}", d["type"], "", ""])
        for r in recovered:
            w.writerow(["recovered", r["train"], "", f"{r['t_start']:.3f}",
                        f"{r['t_stop']:.3f}", hms(r["t_start"]), f"{r['f1']:.1f}",
                        f"{r['f2']:.1f}", r["type"], f"{trains[r['train']]['interval']:.3f}",
                        f"{r['score']:.3f}"])

    print(f"\n{len(trains)} train(s), {len(recovered)} ping(s) recovered at predicted times.")
    print("Recovered pings are marked 'recovered' in the output and are NOT detections: quote "
          "recall with and without them.")
    print(f"Saved {path}")

    if args.plot:
        os.makedirs(args.outdir, exist_ok=True)
        plot_trains(trains, loners, recovered, duration,
                    os.path.join(args.outdir, "trains.png"), args.fs)


if __name__ == "__main__":
    main()
