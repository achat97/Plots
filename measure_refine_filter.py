"""
Measures whether a failure to refine is evidence that a detection was a false alarm.

The refinement stage looks at the spectrogram at full resolution and asks a much more specific
question than the detector did: is there energy here that follows a sweep law? A broadband
transient that fooled the network has no ridge to track, so it fails. That makes the outcome of
refinement a second opinion on whether a pulse was there at all - but only if it discriminates,
and only if the price in recall is acceptable. Real pulses close to the noise also fail, for the
same reason they are hard to detect.

This script measures both sides on data where the truth is known, by building a recording that
contains pulses at known times and adding detections that are deliberately wrong:

  P(refinement fails | a real pulse)   the recall that filtering on refinement would cost
  P(refinement fails | a false alarm)  the precision it would buy

Filtering is worth adopting when the second is much larger than the first. The reasons for failure
are broken out as well, since "no ridge at all" is far stronger evidence than "the ridge was a
little scattered", and a filter can be made to act only on the strong reasons.

Run:
    python measure_refine_filter.py
    python measure_refine_filter.py --n-pulses 60 --n-false 60 --seed 3
"""

import argparse
import numpy as np
import scipy.signal as sps

from data_config import FS, PULSE_F_MIN, PULSE_BW_MIN, PULSE_BW_MAX
from refine_detections import refine_one


def build_recording(n_pulses, n_false, seed, duration=None, snr_range=(-6.0, 24.0)):

    """
    Builds a recording containing pulses at known times, and returns the detections a detector
    might have produced: one per real pulse, plus false alarms of two kinds.

    The false alarms are the two kinds that actually occur. A transient is a short broadband burst,
    which is what a click, a knock or a hull slam looks like and what the network most often
    mistakes for a pulse. An empty detection is a report on ocean noise where nothing happened at
    all. Both are given plausible pulse parameters, so nothing but the audio itself distinguishes
    them from a true detection.

    ----------

    Parameters:
        n_pulses (int) - number of real pulses to inject.
        n_false (int) - number of false alarms to fabricate, split evenly between the two kinds.
        seed (int) - random seed.
        duration (float or None) - recording length in seconds; chosen to fit if None.
        snr_range (tuple) - range of injected SNR in dB.

    Returns:
        data (ndarray) - the recording.
        dets (list) - detections, each with keys t_start, t_stop, f1, f2, truth, kind.
    """

    rng = np.random.default_rng(seed)
    spacing = 6.0
    n_slots = n_pulses + n_false
    if duration is None:
        duration = spacing * (n_slots + 2)

    data = rng.normal(0.0, 1.0, int(duration * FS))
    dets = []
    slot = 1

    def band():

        """A plausible pulse band."""

        f1 = rng.uniform(PULSE_F_MIN, FS / 2 - PULSE_BW_MAX - 1000.0)
        f2 = f1 + rng.uniform(PULSE_BW_MIN, PULSE_BW_MAX)
        return f1, f2

    # --- real pulses ---
    for _ in range(n_pulses):
        t0 = spacing * slot + rng.uniform(-0.3, 0.3); slot += 1
        dur = rng.uniform(0.5, 1.5)
        kind = rng.choice(["cw", "lfm", "hfm"])
        f1, f2 = band()
        if kind == "cw":
            f2 = f1
        snr = rng.uniform(*snr_range)
        amp = 10 ** (snr / 20.0)

        t = np.arange(int(dur * FS)) / FS
        if kind == "cw":
            wave = np.sin(2 * np.pi * f1 * t)
        else:
            wave = sps.chirp(t, f1, t[-1], f2,
                             method="hyperbolic" if kind == "hfm" else "linear")
        wave = wave * sps.windows.tukey(len(wave), 0.2)
        n0 = int(t0 * FS)
        data[n0:n0 + len(wave)] += amp * wave

        # the detection a network would give: right pulse, imperfect parameters
        dets.append({"t_start": t0 + rng.normal(0, 0.08), "t_stop": t0 + dur + rng.normal(0, 0.08),
                     "f1": f1 + rng.normal(0, 400), "f2": f2 + rng.normal(0, 400),
                     "truth": "pulse", "kind": kind, "snr": snr})

    # --- false alarms on broadband transients ---
    for _ in range(n_false // 3):
        t0 = spacing * slot + rng.uniform(-0.3, 0.3); slot += 1
        dur = rng.uniform(0.05, 0.2)
        n0 = int(t0 * FS)
        burst = rng.normal(0.0, 1.0, int(dur * FS)) * sps.windows.tukey(int(dur * FS), 0.3)
        data[n0:n0 + len(burst)] += 8.0 * burst
        f1, f2 = band()
        dets.append({"t_start": t0 - 0.2, "t_stop": t0 + 0.9, "f1": f1, "f2": f2,
                     "truth": "false", "kind": "transient", "snr": np.nan})

    # --- false alarms on a continuous tonal ---
    # The hard case for this filter. Ship machinery radiates a steady narrowband tone that lasts
    # far longer than any pulse, and it DOES give a ridge, so refinement fits it happily. It is
    # caught, if at all, by its duration rather than by its shape.
    for _ in range(n_false // 3):
        t0 = spacing * slot + rng.uniform(-0.3, 0.3); slot += 1
        f_tone = rng.uniform(PULSE_F_MIN, FS / 2 - 2000.0)
        dur = rng.uniform(4.0, 5.5)
        t = np.arange(int(dur * FS)) / FS
        n0 = int(t0 * FS)
        data[n0:n0 + len(t)] += 4.0 * np.sin(2 * np.pi * f_tone * t)
        dets.append({"t_start": t0 + 1.0, "t_stop": t0 + 2.0,
                     "f1": f_tone + rng.normal(0, 300), "f2": f_tone + rng.normal(0, 300),
                     "truth": "false", "kind": "tonal", "snr": np.nan})

    # --- false alarms on nothing at all ---
    for _ in range(n_false - 2 * (n_false // 3)):
        t0 = spacing * slot + rng.uniform(-0.3, 0.3); slot += 1
        f1, f2 = band()
        dets.append({"t_start": t0, "t_stop": t0 + rng.uniform(0.5, 1.2), "f1": f1, "f2": f2,
                     "truth": "false", "kind": "empty", "snr": np.nan})

    return data, dets


def reason_class(note):

    """
    Groups a refinement failure into the reason it happened, since the reasons differ in how much
    they say about whether a pulse was there.

    ----------

    Parameters:
        note (str) - the reason string from refine_one.

    Returns:
        (str) - 'no ridge', 'scatter', 'coverage', or 'other'.
    """

    n = (note or "").lower()
    if "no ridge" in n or "ridge points" in n:
        return "no ridge"
    if "scatter" in n:
        return "scatter"
    if "covers" in n:
        return "coverage"
    return "other"


def main():

    """
    Builds the recording, refines every detection, and reports how often refinement fails for real
    pulses and for false alarms, overall and by reason.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pulses", type=int, default=80)
    ap.add_argument("--n-false", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-min", type=float, default=-6.0)
    ap.add_argument("--snr-max", type=float, default=24.0)
    args = ap.parse_args()

    data, dets = build_recording(args.n_pulses, args.n_false, args.seed,
                                 snr_range=(args.snr_min, args.snr_max))
    print(f"{args.n_pulses} real pulses (SNR {args.snr_min:.0f} to {args.snr_max:.0f} dB), "
          f"{args.n_false} false alarms, {len(data) / FS:.0f} s of audio\n")

    rows = []
    for d in dets:
        res = refine_one(data, FS, d)
        rows.append({"truth": d["truth"], "kind": d["kind"], "snr": d["snr"],
                     "ok": res["ok"],
                     "reason": "" if res["ok"] else reason_class(res.get("reason", ""))})

    real = [r for r in rows if r["truth"] == "pulse"]
    fake = [r for r in rows if r["truth"] == "false"]
    fail_real = [r for r in real if not r["ok"]]
    fail_fake = [r for r in fake if not r["ok"]]

    print("=" * 62)
    print(f"  P(refinement fails | real pulse ) = {len(fail_real) / len(real):.3f}"
          f"   <- recall this filter would cost")
    print(f"  P(refinement fails | false alarm) = {len(fail_fake) / len(fake):.3f}"
          f"   <- precision it would buy")
    print("=" * 62)

    print("\nfalse alarms surviving refinement, by kind:")
    for kind in ("transient", "tonal", "empty"):
        got = [r for r in fake if r["kind"] == kind]
        if got:
            survived = sum(1 for r in got if r["ok"])
            print(f"  {kind:10s} {survived:>3d} of {len(got):>3d} survived")

    print("\nfailures by reason:")
    print(f"  {'reason':12s} {'real pulses':>12s} {'false alarms':>13s}")
    for reason in ("no ridge", "scatter", "coverage", "other"):
        a = sum(1 for r in fail_real if r["reason"] == reason)
        b = sum(1 for r in fail_fake if r["reason"] == reason)
        if a or b:
            print(f"  {reason:12s} {a:>12d} {b:>13d}")

    print("\nfailure rate of real pulses by SNR:")
    edges = [-6, 0, 6, 12, 18, 24]
    for lo, hi in zip(edges[:-1], edges[1:]):
        got = [r for r in real if lo <= r["snr"] < hi]
        if got:
            f = sum(1 for r in got if not r["ok"])
            print(f"  {lo:>3d} to {hi:>3d} dB : {f}/{len(got)} fail ({f / len(got) * 100:.0f} %)")

    # what a filter on refinement would do to the detection counts
    tp_before, fp_before = len(real), len(fake)
    tp_after = sum(1 for r in real if r["ok"])
    fp_after = sum(1 for r in fake if r["ok"])
    print("\neffect of dropping every detection that fails to refine:")
    print(f"  {'':10s} {'true':>6s} {'false':>6s} {'precision':>10s}")
    print(f"  {'before':10s} {tp_before:>6d} {fp_before:>6d} "
          f"{tp_before / (tp_before + fp_before):>10.3f}")
    print(f"  {'after':10s} {tp_after:>6d} {fp_after:>6d} "
          f"{tp_after / max(tp_after + fp_after, 1):>10.3f}")
    print(f"  recall retained: {tp_after / tp_before:.3f}")


if __name__ == "__main__":
    main()
