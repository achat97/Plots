"""
Finds pulse trains among the detections and recovers pings the network missed.

An active sonar usually transmits on a schedule, so its pings arrive at a near-constant pulse
repetition interval with nearly the same parameters each time. That regularity is information the
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
            dets.append({"id": int(row["pulse_id"]), "t_start": t_start, "t_stop": t_stop,
                         "duration": max(t_stop - t_start, 1e-3), "f1": f1, "f2": f2,
                         "type": kind,
                         "direction": row.get("refined_direction", "") or
                                      ("up" if f2 > f1 else "down" if f2 < f1 else "flat"),
                         "refined": refined})
    return dets


def group_by_parameters(dets, freq_tol=600.0, dur_tol=0.5, bw_tol=0.5):

    """
    Groups detections that could come from the same transmitter: the same sweep direction, bands
    within freq_tol of each other, and durations and bandwidths agreeing to within the given
    relative tolerances. Grouping is agglomerative on the raw detections, so a train may still
    contain unrelated pulses; the interval search that follows is what separates them.

    ----------

    Parameters:
        dets (list) - detections from read_detections.
        freq_tol (float) - default 600.0. Largest band difference in Hz within a group.
        dur_tol (float) - default 0.5. Largest relative duration difference within a group.
        bw_tol (float) - default 0.5. Largest relative bandwidth difference within a group.

    Returns:
        groups (list) - lists of detections, each sorted by start time.
    """

    groups = []
    for det in sorted(dets, key=lambda d: d["t_start"]):
        placed = False
        for group in groups:
            ref = group[0]
            if det["direction"] != ref["direction"]:
                continue
            if (abs(det["f1"] - ref["f1"]) > freq_tol
                    or abs(det["f2"] - ref["f2"]) > freq_tol):
                continue
            if abs(det["duration"] - ref["duration"]) > dur_tol * max(ref["duration"], 1e-3):
                continue
            bw_ref = abs(ref["f2"] - ref["f1"])
            bw_det = abs(det["f2"] - det["f1"])
            if abs(bw_det - bw_ref) > bw_tol * max(bw_ref, 100.0):
                continue
            group.append(det)
            placed = True
            break
        if not placed:
            groups.append([det])
    return groups


def find_interval(times, tol=0.1, min_pulses=3, max_period_frac=0.25, min_slots=5,
                  max_scatter=0.06):

    """
    Finds the pulse repetition interval of a set of arrival times, if there is one. Every gap
    between pairs of arrivals is a candidate interval, and the arrivals are tested against the
    grid that candidate implies. Working from pairwise gaps rather than consecutive gaps means
    missing pings do not hide the interval: a train with a hole still shows the period as the gap
    between its surviving members.

    The test that separates a real train from an irregular sequence is not how MANY arrivals sit
    near a grid line but how TIGHTLY they sit. Arrivals scattered at random fall uniformly between
    grid lines, giving a phase scatter near that of a uniform distribution; a transmitter running
    on a clock puts every ping within a small fraction of a slot of its line. The phase spread is
    therefore measured directly, using the circular standard deviation of the arrival phases, and
    a candidate is only accepted when that spread is well below the uniform value.

    ----------

    Parameters:
        times (ndarray) - arrival times in seconds, sorted.
        tol (float) - default 0.1. Relative tolerance when counting arrivals as fitting a slot.
        min_pulses (int) - default 3. Fewest arrivals that must fit the interval to accept it.
        max_period_frac (float) - default 0.25. Largest candidate period, as a fraction of the
                                  span between the first and last arrival.
        min_slots (int) - default 5. Fewest slots a candidate period must imply; with three or
                          four slots almost any set of arrivals can be fitted by some period.
        max_scatter (float) - default 0.15. Largest accepted circular standard deviation of the
                              arrival phases, in units of one slot. Arrivals scattered at
                              random rarely fall below about 0.1 by this measure for any candidate
                              period, while a transmitter running on a clock gives a small
                              fraction of a slot even with several hundred milliseconds of jitter.

    Returns:
        interval (float or None) - the repetition interval in seconds, or None when the arrivals
                                   show no consistent one.
        score (int) - how many arrivals fit the winning interval.
    """

    times = np.asarray(sorted(times), dtype=float)
    if len(times) < min_pulses:
        return None, 0
    span = times[-1] - times[0]
    if span <= 0:
        return None, 0

    candidates = sorted({round(b - a, 3) for i, a in enumerate(times)
                         for b in times[i + 1:] if b - a > 1e-3})
    best, best_score, best_scatter = None, 0, 1e9
    for period in candidates:
        if period > max_period_frac * span:
            continue
        n_slots = int(round(span / period)) + 1
        if n_slots < min_slots:
            continue

        # Circular spread of the arrival phases within a slot: near 0 when every arrival sits on
        # a grid line, and at the uniform value when they are scattered at random.
        phase = 2 * np.pi * ((times - times[0]) / period % 1.0)
        R = float(np.hypot(np.mean(np.cos(phase)), np.mean(np.sin(phase))))
        if R <= 1e-9:
            continue
        scatter = float(np.sqrt(-2.0 * np.log(R))) / (2 * np.pi)   # in units of one slot
        if scatter > max_scatter:
            continue

        offsets = (times - times[0]) / period
        fits = int(np.sum(np.abs(offsets - np.round(offsets)) < tol))
        if fits < min_pulses:
            continue

        # prefer the tightest grid, then the shortest period explaining it
        if scatter < best_scatter - 1e-9 or (abs(scatter - best_scatter) < 1e-9
                                             and best is not None and period < best):
            best, best_score, best_scatter = period, fits, scatter

    if best is None:
        return None, 0
    return float(best), best_score


def build_trains(dets, tol=0.1, min_pulses=3, max_scatter=0.06, **group_kwargs):

    """
    Splits the detections into trains: groups of pulses with matching parameters that also arrive
    on a consistent repetition interval. Detections that fit no train are returned separately.

    ----------

    Parameters:
        dets (list) - detections from read_detections.
        tol (float) - default 0.1. Relative tolerance of the interval search.
        min_pulses (int) - default 3. Fewest pulses required to call a group a train.
        max_scatter (float) - default 0.06. Largest accepted phase spread, in units of a slot.
        **group_kwargs - passed to group_by_parameters.

    Returns:
        trains (list) - dicts with keys members, interval, n_fit.
        loners (list) - detections belonging to no train.
    """

    trains, loners = [], []
    for group in group_by_parameters(dets, **group_kwargs):
        interval, score = find_interval([d["t_start"] for d in group], tol, min_pulses,
                                        max_scatter=max_scatter)
        if interval is None:
            loners.extend(group)
            continue
        members = sorted(group, key=lambda d: d["t_start"])
        trains.append({"members": members, "interval": interval, "n_fit": score})
    return trains, loners


def predict_gaps(train, duration, tol=0.1, edge_margin=0.0):

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

    Returns:
        gaps (list) - predicted start times in seconds.
    """

    members = train["members"]
    period = train["interval"]
    t0, t1 = members[0]["t_start"], members[-1]["t_start"]
    have = np.array([d["t_start"] for d in members])

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


def train_replica(train, fs):

    """
    Builds the waveform the train's next ping is expected to look like: a Tukey-windowed chirp
    with the median parameters of the train's members, using the sweep law of their type.

    ----------

    Parameters:
        train (dict) - one train from build_trains.
        fs (int) - sampling frequency in Hz.

    Returns:
        replica (ndarray) - the expected waveform, unit energy.
        params (dict) - the median duration, f1 and f2 used.
    """

    members = train["members"]
    duration = float(np.median([d["duration"] for d in members]))
    f1 = float(np.median([d["f1"] for d in members]))
    f2 = float(np.median([d["f2"] for d in members]))
    kind = max(set(d["type"] for d in members),
               key=[d["type"] for d in members].count)

    t = np.arange(int(round(duration * fs))) / fs
    if len(t) < 8:
        t = np.arange(8) / fs
    method = "hyperbolic" if kind == "hfm" else "linear"
    if abs(f2 - f1) < 1.0 or kind == "cw":
        wave = np.sin(2 * np.pi * f1 * t)
    else:
        wave = sps.chirp(t, f1, t[-1], f2, method=method)
    wave = wave * sps.windows.tukey(len(wave), 0.2)
    wave /= np.linalg.norm(wave) + 1e-12
    return wave, {"duration": duration, "f1": f1, "f2": f2, "type": kind}


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
    Draws the trains over the recording: detected pings as filled markers, recovered pings as open
    markers, and detections belonging to no train in grey. One row per train makes the repetition
    visible, and a gap that stayed empty is left blank.

    ----------

    Parameters:
        trains (list) - trains from build_trains.
        loners (list) - detections in no train.
        recovered (list) - recovered ping dicts with keys train, t_start, score.
        duration (float) - length of the recording in seconds.
        path (str) - output file path.
        fs (int) - sampling frequency in Hz.

    Returns:
        None
    """

    rows = len(trains) + (1 if loners else 0)
    fig, ax = plt.subplots(figsize=(FIG_WIDE[0], max(3.0, 0.7 * rows + 2.0)))

    labels = []
    for i, train in enumerate(trains):
        y = len(trains) - 1 - i
        colour = TYPE_COLORS.get(train["members"][0]["type"], REFERENCE)
        ax.plot([d["t_start"] for d in train["members"]], [y] * len(train["members"]),
                "o", ms=6, color=colour,
                label="Detected" if i == 0 else None)
        got = [r for r in recovered if r["train"] == i]
        if got:
            ax.plot([r["t_start"] for r in got], [y] * len(got), "o", ms=8,
                    mfc="none", mew=1.8, color=ACCENT,
                    label="Recovered" if not labels else None)
            labels.append(1)
        f1 = np.median([d["f1"] for d in train["members"]])
        ax.text(-0.01 * duration, y, f"{type_name(train['members'][0]['type'])} "
                                     f"{f1 / 1000:.1f} kHz, {train['interval']:.1f} s",
                ha="right", va="center", fontsize=9)

    if loners:
        ax.plot([d["t_start"] for d in loners], [-1] * len(loners), "x", ms=6, color="0.5",
                label="No train")
        ax.text(-0.01 * duration, -1, "unmatched", ha="right", va="center", fontsize=9)

    ax.set_yticks([])
    ax.set_xlim(-0.18 * duration, duration)
    ax.set_ylim(-2, len(trains))
    hms_axis(ax)
    finish(ax, "Time [h:mm:ss]", None,
           f"{len(trains)} train(s), {len(recovered)} recovered ping(s)", legend=True)
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
    ap.add_argument("--min-pulses", type=int, default=3,
                    help="fewest pulses on a consistent interval to call a group a train")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="relative tolerance of the repetition interval")
    ap.add_argument("--max-scatter", type=float, default=0.06,
                    help="largest accepted spread of arrival times around the repetition grid, "
                         "in units of one interval; guards against reading a period into "
                         "irregular arrivals (uniform scatter gives about 0.29 by this measure)")
    ap.add_argument("--freq-tol", type=float, default=600.0,
                    help="largest band difference in Hz within one train")
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
                                  max_scatter=args.max_scatter, freq_tol=args.freq_tol)
    if not trains:
        print("No repeating trains found; every detection stands alone and nothing is predicted.")
    for i, train in enumerate(trains):
        members = train["members"]
        f1 = np.median([d["f1"] for d in members])
        print(f"  train {i}: {len(members)} pulses, interval {train['interval']:.2f} s, "
              f"{type_name(members[0]['type'])} near {f1:.0f} Hz, "
              f"{members[0]['t_start']:.1f}-{members[-1]['t_start']:.1f} s")
    if loners:
        print(f"  {len(loners)} detection(s) in no train")

    recovered = []
    if not args.no_recover:
        for i, train in enumerate(trains):
            gaps = predict_gaps(train, duration, tol=args.tol)
            if not gaps:
                continue
            replica, params = train_replica(train, args.fs)
            search = args.search if args.search is not None else 0.1 * train["interval"]
            for t_pred in gaps:
                score, t_found = matched_filter(data, args.fs, replica, t_pred, search,
                                                params["f1"], params["f2"],
                                                duration=params["duration"],
                                                kind=params["type"])
                status = "recovered" if score >= args.min_snr else "not found"
                print(f"    gap at {hms(t_pred)}: sweep {score:+.1f} dB over background "
                      f"-> {status}")
                if score >= args.min_snr:
                    recovered.append({"train": i, "t_start": t_found,
                                      "t_stop": t_found + params["duration"],
                                      "f1": params["f1"], "f2": params["f2"],
                                      "type": params["type"], "score": score,
                                      "t_predicted": t_pred})

    path = args.out or os.path.splitext(args.detections)[0] + "_trains.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "train_id", "pulse_id", "t_start_s", "t_stop_s", "t_start_hms",
                    "f1_hz", "f2_hz", "type", "interval_s", "match_score"])
        for i, train in enumerate(trains):
            for d in train["members"]:
                w.writerow(["detected", i, d["id"], f"{d['t_start']:.3f}", f"{d['t_stop']:.3f}",
                            hms(d["t_start"]), f"{d['f1']:.1f}", f"{d['f2']:.1f}", d["type"],
                            f"{train['interval']:.3f}", ""])
        for d in loners:
            w.writerow(["detected", "", d["id"], f"{d['t_start']:.3f}", f"{d['t_stop']:.3f}",
                        hms(d["t_start"]), f"{d['f1']:.1f}", f"{d['f2']:.1f}", d["type"], "", ""])
        for r in recovered:
            w.writerow(["recovered", r["train"], "", f"{r['t_start']:.3f}", f"{r['t_stop']:.3f}",
                        hms(r["t_start"]), f"{r['f1']:.1f}", f"{r['f2']:.1f}", r["type"],
                        f"{trains[r['train']]['interval']:.3f}", f"{r['score']:.3f}"])

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
