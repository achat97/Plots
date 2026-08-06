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
from table_io import save_table_xlsx
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


def _extend_chain(times, start, period, tol, max_gap_slots, min_pulses):

    """
    Builds one train by walking forward from a starting arrival, re-estimating the interval as it
    goes.

    Walking is necessary because a candidate interval is only approximate. Grouping arrivals by
    their phase across the whole recording would need the candidate accurate to the tolerance
    DIVIDED by the number of slots - a train of eighty pings would need its interval right to about
    a tenth of a per cent, which no practical candidate list provides, and a candidate one per cent
    out scrambles the phases completely. Walking forward only ever needs the interval right over
    the next step or two, and once a few pulses are in hand it is re-fitted from them, so the
    estimate improves as the chain grows instead of drifting away from it.

    Missed pings are skipped: the walk keeps predicting the next slot and gives up only after
    max_gap_slots consecutive empty ones, which is where the transmitter is taken to have stopped.

    ----------

    Parameters:
        times (ndarray) - all arrival times, sorted.
        start (int) - index of the arrival to start from.
        period (float) - the candidate interval.
        tol (float) - relative tolerance for accepting an arrival into the chain.
        max_gap_slots (int) - consecutive empty slots that end the chain.
        min_pulses (int) - fewest members for the chain to be worth returning.

    Returns:
        chain (list) - indices of the arrivals in the chain, or [] when too short.
        period (float) - the interval re-fitted to those arrivals.
    """

    n = len(times)
    chain, slots = [start], [0]
    p = period
    slot = misses = 0
    in_chain = {start}

    while True:
        slot += 1
        predicted = times[chain[0]] + p * slot
        if predicted > times[-1] + tol * p:
            break

        j = int(np.searchsorted(times, predicted))
        best, best_err = -1, tol * p
        for cand in (j - 1, j):
            if 0 <= cand < n and cand not in in_chain:
                err = abs(times[cand] - predicted)
                if err <= best_err:
                    best, best_err = cand, err

        if best < 0:
            misses += 1
            if misses > max_gap_slots:
                break
            continue

        misses = 0
        chain.append(best)
        in_chain.add(best)
        slots.append(slot)

        # Re-fit the interval to the chain so far, which is what keeps a slightly wrong candidate
        # from drifting off a long train.
        if len(chain) >= 3:
            k = np.array(slots, dtype=float)
            A = np.vstack([np.ones_like(k), k]).T
            try:
                coef, *_ = np.linalg.lstsq(A, times[chain], rcond=None)
                if coef[1] > 0:
                    p = float(coef[1])
            except np.linalg.LinAlgError:
                pass

    if len(chain) < min_pulses:
        return [], p
    return sorted(chain), p


def _candidate_periods(times, min_period, max_period, tol, max_candidates=250, min_hits=3):

    """
    Proposes candidate repetition intervals from the gaps between pairs of arrivals. A true
    interval shows up as a gap between many pairs - between neighbours, and between arrivals two,
    three or more slots apart when pings in between were missed - so the gaps that recur most often
    are the intervals worth testing. Ranking them and keeping only the strongest is what keeps the
    search fast: collecting the arrivals for every distinct gap is cubic in the number of
    detections, which is unusable on a long recording.

    ----------

    Parameters:
        times (ndarray) - arrival times in seconds, sorted.
        min_period (float) - shortest interval to consider.
        max_period (float) - longest interval to consider.
        tol (float) - relative tolerance used when binning gaps together.
        max_candidates (int) - default 250. Most candidates returned.
        min_hits (int) - default 3. Fewest pairs a candidate must be supported by.

    Returns:
        periods (list) - candidate intervals, strongest first.
    """

    n = len(times)
    gaps = []
    for i in range(n):
        d = times[i + 1:] - times[i]
        d = d[(d >= min_period) & (d <= max_period)]
        if d.size:
            gaps.append(d)
    if not gaps:
        return []
    gaps = np.concatenate(gaps)

    # Bin more finely than the membership tolerance. A bin is what a candidate is quantised to,
    # and a candidate several per cent from the truth mixes two trains into one peak before the
    # refinement gets a chance to separate them. Finer bins cost only a longer candidate list,
    # which is cheap to test.
    width = np.log1p(tol / 4.0)
    idx = np.round(np.log(gaps) / width).astype(np.int64)
    uniq, counts = np.unique(idx, return_counts=True)
    centres = np.exp(uniq * width)

    # Rank by DENSITY, not by raw count. The bins are relative, so a bin at a long period is wide
    # in absolute terms and sweeps up unrelated gaps, while a bin at a short period is narrow.
    # Unrelated arrivals therefore contribute counts in proportion to the period, and dividing by
    # it removes that bias - without this correction the search reliably prefers a multiple of the
    # true interval to the interval itself.
    keep = counts >= max(3, min_hits)
    if not np.any(keep):
        keep = counts >= 2
    density = np.where(keep, counts / centres, -np.inf)
    order = np.argsort(-density)

    periods, seen = [], []
    for k in order[: max_candidates * 3]:
        if not np.isfinite(density[k]):
            break
        p = float(centres[k])
        # Deduplicate only near-identical candidates. Merging everything within the membership
        # tolerance would discard a real interval whenever a slightly different one happened to
        # rank above it, and the one kept would then be a few per cent out - enough to break the
        # train it was meant to find.
        if any(abs(p - q) <= 0.25 * tol * q for q in seen):
            continue
        seen.append(p)
        periods.append(p)
        if len(periods) >= max_candidates:
            break
    return periods


def _train_quality(members, period, tol):

    """
    Scores a candidate train. A set of arrivals is a train when it is REGULAR, which means two
    things at once: the arrivals sit tightly on the grid, and they fill most of the slots the grid
    implies. Tightness alone is not enough, because a sparse grid can always be laid over a few
    scattered arrivals; occupancy alone is not enough either, because a short interval fills its
    slots trivially. Both are therefore required.

    Tightness is measured against the BEST-FITTING grid rather than against a grid anchored on the
    first arrival. Anchoring on one arrival makes the measurement depend on how exact the interval
    happens to be: an error of a thousandth of a period looks like nothing at the start and like a
    large offset three hundred slots later, so a long, perfectly regular train would be judged
    irregular. Fitting the grid removes that, and what remains is the jitter itself.

    ----------

    Parameters:
        members (list) - detections on the grid, sorted by time.
        period (float) - the candidate interval in seconds.
        tol (float) - relative tolerance (kept for symmetry with the other helpers).

    Returns:
        scatter (float) - spread of the arrivals about the fitted grid, in units of one slot.
        occupancy (float) - fraction of the implied slots that contain an arrival.
        n_slots (int) - number of slots between the first and last member.
    """

    times = np.array([d["t_start"] for d in members], dtype=float)
    if len(times) < 2 or period <= 0:
        return 1e9, 0.0, 0

    span = times[-1] - times[0]
    n_slots = int(round(span / period)) + 1

    k = np.round((times - times[0]) / period)
    A = np.vstack([np.ones_like(k), k]).T
    try:
        coef, *_ = np.linalg.lstsq(A, times, rcond=None)
        resid = times - A @ coef
    except np.linalg.LinAlgError:
        return 1e9, 0.0, n_slots

    scatter = float(np.std(resid)) / period
    return scatter, len(times) / max(n_slots, 1), n_slots


def _refine_grid(times, member_idx, period, tol, n_iter=3):

    """
    Refines a candidate interval and the set of arrivals on its grid.

    Candidate intervals come from a histogram, so they are quantised to a bin centre and can be a
    per cent or two away from the truth. That error accumulates: over thirty cycles a one per cent
    error drifts by a third of a period, which is enough to break the very train the candidate was
    meant to find. Each candidate is therefore refined before it is judged - the arrivals are
    assigned to slots, the interval is re-estimated by least squares against those slot numbers,
    and the grid is re-collected. A couple of passes converge.

    ----------

    Parameters:
        times (ndarray) - all arrival times, sorted.
        member_idx (list) - indices of the arrivals the candidate grid started from.
        period (float) - the candidate interval.
        tol (float) - relative tolerance for grid membership.
        n_iter (int) - default 3. Refinement passes.

    Returns:
        members (list) - indices of the arrivals on the refined grid, sorted by time.
        period (float) - the refined interval.
    """

    idx = list(member_idx)
    for _ in range(n_iter):
        if len(idx) < 2 or period <= 0:
            break
        t = times[idx]

        # Estimate the interval from CONSECUTIVE gaps first. A candidate coming from a histogram
        # bin can be a per cent or two out, and numbering slots across the whole train with such a
        # value drifts by most of a period by the end - so the slot numbers, and any fit based on
        # them, would be wrong before the fit began. A consecutive gap spans only one slot or a
        # few, so the same error cannot accumulate, and the median over all of them is robust to
        # the odd outlier.
        d = np.diff(t)
        m = np.round(d / period)
        good = m >= 1
        if np.any(good):
            period = float(np.median(d[good] / m[good]))
        if period <= 0:
            return [], 0.0

        k = np.round((t - t[0]) / period)
        if k.max() <= 0:
            break
        # least squares fit of t = a + period * k, which is the interval that best explains the
        # arrivals given their slot numbers
        A = np.vstack([np.ones_like(k), k]).T
        try:
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        except np.linalg.LinAlgError:
            break
        a, period = float(coef[0]), float(coef[1])
        if period <= 0:
            return [], 0.0

        # re-collect: every arrival close to a slot of the refined grid, one per slot
        kk = np.round((times - a) / period)
        err = np.abs(times - (a + kk * period))
        best = {}
        for i in np.nonzero(err <= tol * period)[0]:
            key = int(kk[i])
            if key not in best or err[i] < err[best[key]]:
                best[key] = int(i)
        new_idx = sorted(best.values())
        if new_idx == idx:
            break
        idx = new_idx

    # Trim contaminants. Where two trains run at once, a pulse of one can fall close enough to the
    # other's grid to be collected by it, and a handful of such pulses is enough to make a clean
    # train look irregular and be discarded. Members are therefore compared with the spread of the
    # train itself: those lying far outside it are dropped, which leaves a genuine train intact
    # while releasing borrowed pulses back to their own.
    if len(idx) >= 3 and period > 0:
        t = times[idx]
        k = np.round((t - t[0]) / period)
        A = np.vstack([np.ones_like(k), k]).T
        try:
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
            resid = np.abs(t - A @ coef)
            scale = max(float(np.median(resid)) * 4.0, 0.01 * period)
            keep = resid <= scale
            if keep.sum() >= 3 and keep.sum() < len(idx):
                idx = [i for i, k_ in zip(idx, keep) if k_]
        except np.linalg.LinAlgError:
            pass

    return idx, period


def _chance_occupancy(n_detections, span, period, tol):

    """
    The occupancy a grid of this period would reach on arrivals that are not a train at all.

    Membership accepts an arrival within tol * period of a slot, so each slot claims a window whose
    width GROWS with the period. On a busy recording a long period therefore fills most of its
    slots whatever the arrivals do, and judging it by occupancy alone would call any long grid a
    train. Modelling the arrivals as randomly placed, a slot's window is empty with probability
    exp(-lambda), where lambda is the expected number of arrivals inside it - so the occupancy to
    beat is one minus that.

    ----------

    Parameters:
        n_detections (int) - number of detections in the recording.
        span (float) - length of the recording covered by detections, in seconds.
        period (float) - candidate interval in seconds.
        tol (float) - relative tolerance.

    Returns:
        (float) - the occupancy expected by chance, between 0 and 1.
    """

    if span <= 0 or period <= 0:
        return 1.0
    density = n_detections / span
    lam = density * 2.0 * tol * period
    return float(1.0 - np.exp(-lam))


def _contiguous_runs(times, idx, period, max_gap_slots=5):

    """
    Splits a set of arrivals on a grid into contiguous runs of transmission.

    A transmitter runs for a while and stops. Two trains of different intervals will, over a long
    recording, occasionally place pulses on each other's grid, and those stragglers stretch a
    train's apparent extent far beyond where it actually transmitted - which makes a real train
    look sparse and be rejected. Splitting wherever the grid stays empty for several slots keeps
    each genuine stretch of transmission together and leaves the stragglers as short fragments that
    fail the minimum length on their own.

    ----------

    Parameters:
        times (ndarray) - all arrival times, sorted.
        idx (list) - indices of the arrivals on the grid.
        period (float) - the grid interval in seconds.
        max_gap_slots (int) - default 5. Consecutive empty slots that end a run.

    Returns:
        runs (list) - lists of indices, each a contiguous stretch.
    """

    if len(idx) < 2 or period <= 0:
        return [list(idx)]
    t = times[idx]
    slot = np.round((t - t[0]) / period).astype(np.int64)
    runs, current = [], [idx[0]]
    for a in range(1, len(idx)):
        if slot[a] - slot[a - 1] > max_gap_slots:
            runs.append(current)
            current = [idx[a]]
        else:
            current.append(idx[a])
    runs.append(current)
    return runs


def build_trains(dets, tol=0.1, min_pulses=8, max_scatter=0.06, min_occupancy=0.6,
                 min_period=0.5, freq_tol=None, max_candidates=250, max_gap_slots=5):

    """
    Finds the pulse trains among the detections. A train is defined by its REPETITION INTERVAL and
    by nothing else: a transmitter may send any mixture of waveforms - CW, LFM and HFM in any
    order, changing from ping to ping - and those pulses still belong to one train because they
    arrive on one clock. Pulse type, band and duration are recorded and reported, but they take no
    part in deciding membership.

    The search has two stages. Candidate intervals are proposed from the gaps that recur most often
    between pairs of arrivals (see _candidate_periods), since a true interval separates many pairs
    even when pings in between were missed. Each candidate is then tested by grouping the arrivals
    by their phase within one period (see _phase_clusters), which finds every grid of that period
    in a single pass. A cluster is accepted as a train only if it is regular in both senses
    described in _train_quality.

    Candidates are finally taken in order of how many pulses they explain, each detection joining
    at most one train, so the strongest periodicity claims its pulses first and weaker coincidences
    cannot re-use them.

    ----------

    Parameters:
        dets (list) - detections from read_detections.
        tol (float) - default 0.1. Relative tolerance on the interval.
        min_pulses (int) - default 8. Fewest pulses required to call a set a train. Measured on
                           irregular arrivals with no train present, a bar of six reported a
                           spurious train in 13 of 60 recordings and a bar of eight in 2 of 60,
                           with no real train lost in either case.
        max_scatter (float) - default 0.06. Largest accepted phase spread, in units of one slot.
        min_occupancy (float) - default 0.6. Smallest fraction of the implied slots that must
                                contain a pulse; this is what stops a short interval from
                                "explaining" arrivals by offering more slots than it fills.
        min_period (float) - default 0.5. Shortest interval considered, in seconds.
        freq_tol (float or None) - default None (any band). When given, only detections whose
                                   bands lie within this many Hz of each other may share a train.
        max_candidates (int) - default 250. Most candidate intervals tested.
        max_gap_slots (int) - default 5. Consecutive empty slots that end a train; beyond this the
                              transmitter is taken to have stopped.

    Returns:
        trains (list) - dicts with keys members, interval, scatter, occupancy.
        loners (list) - detections belonging to no train.
    """

    dets = sorted(dets, key=lambda d: d["t_start"])
    times = np.array([d["t_start"] for d in dets], dtype=float)
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    if len(dets) < min_pulses or span <= 0:
        return [], list(dets)

    proposals, seen_grids = [], set()
    for period0 in _candidate_periods(times, min_period, 0.5 * span, tol, max_candidates,
                                      min_hits=min_pulses - 1):
        covered = set()
        # A start too near the end cannot reach the minimum number of slots, so it is skipped
        # rather than walked.
        last_useful = np.searchsorted(times, times[-1] - (min_pulses - 1) * period0 * (1 - tol))
        for start in range(int(last_useful) + 1):
            if start in covered:
                continue
            cluster, period1 = _extend_chain(times, start, period0, tol, max_gap_slots,
                                             min_pulses)
            if not cluster:
                continue
            covered.update(cluster)
            idx, period = _refine_grid(times, cluster, period1, tol)
            if len(idx) < min_pulses or period < min_period:
                continue
            for run in _contiguous_runs(times, idx, period, max_gap_slots):
                if len(run) < min_pulses:
                    continue
                # different candidates often converge on the same refined grid
                key = (round(period, 3), run[0], len(run))
                if key in seen_grids:
                    continue
                seen_grids.add(key)
                members = [dets[i] for i in run]
                if freq_tol is not None:
                    lo = min(min(d["f1"], d["f2"]) for d in members)
                    hi = max(max(d["f1"], d["f2"]) for d in members)
                    if hi - lo > freq_tol:
                        continue
                scatter, occ, n_slots = _train_quality(members, period, tol)
                if scatter > max_scatter or occ < min_occupancy or n_slots < min_pulses:
                    continue
                # Beat chance, not just the fixed floor: a long period fills its slots easily
                # because its acceptance windows are wide (see _chance_occupancy). The margin is
                # three standard deviations of a binomial, so a grid has to be filled clearly
                # better than an unrelated one would be.
                p_chance = _chance_occupancy(len(dets), span, period, tol)
                margin = 3.0 * np.sqrt(max(p_chance * (1 - p_chance), 1e-6) / max(n_slots, 1))
                if occ <= p_chance + margin:
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
        # Absorb any still-unclaimed detection that fits this grid. A chain can be interrupted by
        # a long silence and picked up again as a second proposal; letting the accepted train take
        # the rest of its own pulses puts those halves back together.
        t_free = np.array([d["t_start"] for d in free])
        k0 = t_free[0]
        taken = {int(round((t - k0) / p["interval"])) for t in t_free}
        extra = []
        for d in dets:
            if id(d) in claimed or d in free:
                continue
            k = int(round((d["t_start"] - k0) / p["interval"]))
            if k in taken:
                continue                      # a slot holds one pulse; do not double-fill it
            if abs(d["t_start"] - (k0 + k * p["interval"])) <= tol * p["interval"]:
                extra.append(d)
                taken.add(k)
        if extra:
            merged = sorted(free + extra, key=lambda d: d["t_start"])
            for run in _contiguous_runs(np.array([d["t_start"] for d in merged]),
                                        list(range(len(merged))), p["interval"], max_gap_slots):
                cand = [merged[i] for i in run]
                if len(cand) > len(free):
                    sc2, occ2, _ns2 = _train_quality(cand, p["interval"], tol)
                    if sc2 <= max_scatter and occ2 >= min_occupancy:
                        free, scatter, occ = cand, sc2, occ2
                        break

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
    ap.add_argument("--min-pulses", type=int, default=8,
                    help="fewest pulses on a consistent interval to call a set a train")
    ap.add_argument("--min-occupancy", type=float, default=0.6,
                    help="smallest fraction of the slots implied by the interval that must "
                         "contain a pulse; stops a short interval from explaining arrivals "
                         "by offering more slots than it fills")
    ap.add_argument("--min-period", type=float, default=0.5,
                    help="shortest repetition interval considered, in seconds")
    ap.add_argument("--max-candidates", type=int, default=250,
                    help="most candidate intervals tested; the gaps that recur most often are "
                         "tried first, so raising this rarely finds more trains")
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
    ap.add_argument("--xlsx", action="store_true",
                    help="also write the table as an Excel file beside the CSV")
    ap.add_argument("--plot", action="store_true", help="save a figure of the trains")
    ap.add_argument("--outdir", default="trains")
    args = ap.parse_args()

    dets = read_detections(args.detections)
    data, duration = read_wav(args.wav, args.fs)
    print(f"{args.wav}: {duration:.1f} s | {len(dets)} detections")

    trains, loners = build_trains(dets, tol=args.tol, min_pulses=args.min_pulses,
                                  max_scatter=args.max_scatter,
                                  min_occupancy=args.min_occupancy,
                                  min_period=args.min_period, freq_tol=args.freq_tol,
                                  max_candidates=args.max_candidates)
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
    header = ["source", "train_id", "pulse_id", "t_start_s", "t_stop_s",
              "t_start_hms", "f1_hz", "f2_hz", "type", "interval_s", "match_score"]
    table = []
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, train in enumerate(trains):
            for d in train["members"]:
                table.append(["detected", i, d["id"], round(d["t_start"], 3),
                              round(d["t_stop"], 3), hms(d["t_start"]), round(d["f1"], 1),
                              round(d["f2"], 1), d["type"], round(train["interval"], 3), ""])
                w.writerow(table[-1])
        for d in loners:
            table.append(["detected", "", d["id"], round(d["t_start"], 3),
                          round(d["t_stop"], 3), hms(d["t_start"]), round(d["f1"], 1),
                          round(d["f2"], 1), d["type"], "", ""])
            w.writerow(table[-1])
        for r in recovered:
            table.append(["recovered", r["train"], "", round(r["t_start"], 3),
                          round(r["t_stop"], 3), hms(r["t_start"]), round(r["f1"], 1),
                          round(r["f2"], 1), r["type"],
                          round(trains[r["train"]]["interval"], 3), round(r["score"], 3)])
            w.writerow(table[-1])

    if args.xlsx:
        save_table_xlsx(header, table, os.path.splitext(path)[0] + ".xlsx", sheet="trains")

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
