"""
Measures, on YOUR test split, whether a failure to refine is evidence that a detection was a false
alarm.

The companion script measure_refine_filter.py answers the same question on a recording it builds
itself, which fixes the pulse statistics and the kinds of false alarm considered. This one uses the
test split of your own dataset and your own checkpoint, so the pulses have the distribution you
trained on and the false alarms are the ones your model actually makes.

Each detection the model produces is first labelled against the ground truth, using the same
matching rule the evaluation uses - a true positive if it pairs with a real pulse, a false alarm if
it does not. Every detection is then refined, and the two are cross-tabulated:

  P(refinement fails | true positive)  the recall that filtering on refinement would cost
  P(refinement fails | false alarm)    the precision it would buy

Filtering is worth adopting when the second is much larger than the first. Because the test split
also carries the SNR of each example, the cost is broken down by SNR, which is where it is paid.

Refinement here runs on the stored spectrograms rather than on a recording: a dataset keeps
spectrograms, and the phase needed to recover a waveform is not kept. It is the same ridge
tracking and sweep fitting used in deployment.

Run:
    python measure_refine_filter_testset.py --checkpoint best_model_....pth
"""

import argparse

import numpy as np
import torch

from model import CONFIG
from dataset_io import make_splits, MemmapPulseDataset, load_axes
from metrics_core import predict_dataset, truth_pulses, match_pulses
from snr_analysis import load_model
from refine_detections import refine_spectrogram


def main():

    """
    Loads the checkpoint and the test split, runs the model, labels every detection against the
    ground truth, refines each one from its stored spectrogram, and reports how often refinement
    fails for true positives and for false alarms.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", default="INPUT.npy")
    ap.add_argument("--target", default="padded_sequences.npy")
    ap.add_argument("--meta", default="META.npy")
    ap.add_argument("--axes", default="AXES.npz")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=0,
                    help="cap on examples, for a quick look (0 = the whole split)")
    ap.add_argument("--snr-bins", default="-15,-5,5,15,25,40")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(args.input, mmap_mode="r")
    y = np.load(args.target, allow_pickle=True).astype(np.float32)
    meta = np.load(args.meta)
    t_axis, f_axis = load_axes(args.axes)

    modules, cfg = load_model(args.checkpoint, (X.shape[1], X.shape[2]), device)
    idx_train, idx_val, idx_test = make_splits(meta, cfg)
    idx = np.sort({"train": idx_train, "val": idx_val, "test": idx_test}[args.split])
    if args.max_examples:
        idx = idx[:args.max_examples]

    ds = MemmapPulseDataset(args.input, y, idx)
    print(f"{len(idx)} examples from the {args.split} split | device {device}")

    all_pred = predict_dataset(modules, ds, cfg, device, batch_size=args.batch_size)
    all_truth = truth_pulses(ds.targets, cfg, scaled=False)
    snr = meta[idx, 0]

    rows = []
    for n, (pred, truth) in enumerate(zip(all_pred, all_truth)):
        pairs, false_alarms, _misses = match_pulses(pred, truth)
        matched = {i for i, _j in pairs}
        S = np.array(X[idx[n]], dtype=np.float64)

        for i, det in enumerate(pred):
            res = refine_spectrogram(S, f_axis, t_axis, det)
            rows.append({"true_positive": i in matched, "ok": res["ok"],
                         "reason": "" if res["ok"] else res.get("reason", ""),
                         "snr": float(snr[n])})

        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(all_pred)} examples")

    tp = [r for r in rows if r["true_positive"]]
    fa = [r for r in rows if not r["true_positive"]]
    if not tp or not fa:
        print("\nNeed both true positives and false alarms to compare; "
              f"got {len(tp)} and {len(fa)}.")
        return

    fail_tp = sum(1 for r in tp if not r["ok"])
    fail_fa = sum(1 for r in fa if not r["ok"])

    print("\n" + "=" * 64)
    print(f"  detections: {len(tp)} true positives, {len(fa)} false alarms")
    print(f"  P(refinement fails | true positive) = {fail_tp / len(tp):.3f}"
          f"   <- recall cost")
    print(f"  P(refinement fails | false alarm  ) = {fail_fa / len(fa):.3f}"
          f"   <- precision gain")
    print("=" * 64)

    print("\nreasons refinement failed:")
    reasons = sorted({r["reason"].split(" (")[0].split(" at ")[0] for r in rows
                      if not r["ok"]})
    print(f"  {'reason':40s} {'true pos':>9s} {'false':>7s}")
    for reason in reasons:
        a = sum(1 for r in tp if not r["ok"] and r["reason"].startswith(reason))
        b = sum(1 for r in fa if not r["ok"] and r["reason"].startswith(reason))
        print(f"  {reason[:40]:40s} {a:>9d} {b:>7d}")

    print("\nrecall cost by SNR (true positives only):")
    edges = [float(v) for v in args.snr_bins.split(",")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        got = [r for r in tp if lo <= r["snr"] < hi]
        if got:
            f = sum(1 for r in got if not r["ok"])
            print(f"  {lo:>6.0f} to {hi:>4.0f} dB : {f:>4d}/{len(got):<4d} fail "
                  f"({f / len(got) * 100:5.1f} %)")

    keep_tp, keep_fa = len(tp) - fail_tp, len(fa) - fail_fa
    print("\neffect of --drop-unrefined on this split:")
    print(f"  {'':8s} {'true':>6s} {'false':>6s} {'precision':>10s} {'recall kept':>12s}")
    print(f"  {'before':8s} {len(tp):>6d} {len(fa):>6d} "
          f"{len(tp) / (len(tp) + len(fa)):>10.3f} {1.0:>12.3f}")
    print(f"  {'after':8s} {keep_tp:>6d} {keep_fa:>6d} "
          f"{keep_tp / max(keep_tp + keep_fa, 1):>10.3f} {keep_tp / len(tp):>12.3f}")


if __name__ == "__main__":
    main()
