"""
Diagnostic: how sensitive is a trained checkpoint to a GLOBAL dB offset of its input spectrograms?

Motivation: before this revision, training data pinned the noise floor at a fixed absolute dB level
(z-scoring before pulse injection) while deployment windows were z-scored with the pulse included,
shifting the whole image by up to ~20 dB at high SNR. That mismatch is now fixed at the source
(pulses.py re-normalizes after injection), but this sweep remains the direct test of level
robustness: it adds a constant offset to every test spectrogram and reports the detection metrics
per offset. A robust model holds its F1 over a wide range; a model keying on absolute level
collapses quickly. Also useful for judging how much headroom there is against gain/calibration
differences of real recordings.

Run:
    python offset_sweep.py --checkpoint best_model.pth --offsets=-20,-10,-5,0,5,10
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from dataset_io import make_splits, MemmapPulseDataset
from metrics_core import predict_dataset, truth_pulses, ar_score
from snr_analysis import load_model
from plot_style import FIG, PRIMARY, ACCENT, REFERENCE, finish, save


class OffsetDataset(Dataset):

    """
    Wraps a spectrogram dataset and adds a constant dB offset to every input; targets pass through.
    """

    def __init__(self, base, offset_db):
        self.base = base
        self.offset_db = float(offset_db)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x + self.offset_db, y


def main():

    """
    Parses arguments, loads the checkpoint and the chosen split, evaluates the model under every
    requested dB offset, prints a metric table, and saves a CSV plus a recall/F1-vs-offset figure.

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
    ap.add_argument("--meta", default="META.npy",
                    help="META file supplying the segment groups for the split")
    ap.add_argument("--offsets", default="-40,-30,-20,-10,-5,0,5",
                    help="comma-separated global offsets in dB added to every spectrogram")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=0,
                    help="cap on examples per offset (0 = all in the split)")
    ap.add_argument("--outdir", default="offset_sweep")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(args.input, mmap_mode="r")
    y = np.load(args.target, allow_pickle=True).astype(np.float32)
    meta = np.load(args.meta)
    assert len(meta) == len(y), "META and targets must be index-aligned"
    modules, cfg = load_model(args.checkpoint, (X.shape[1], X.shape[2]), device)

    idx_train, idx_val, idx_test = make_splits(meta, cfg)
    split_map = {"train": idx_train, "val": idx_val, "test": idx_test,
                 "all": np.arange(len(y))}
    idx = np.sort(np.asarray(split_map[args.split]))
    base = MemmapPulseDataset(args.input, y, idx)
    print(f"{len(base)} examples ({args.split} split) | device {device}")

    n = min(args.max_examples, len(base)) if args.max_examples else len(base)
    all_truth = truth_pulses(base.targets[:n], cfg, scaled=False)

    offsets = [float(v) for v in args.offsets.split(",")]
    results = []
    print(f"\n  {'offset':>8s} {'AR':>8s} {'F1':>7s} {'recall':>7s} {'prec':>7s} "
          f"{'cnt=':>6s} {'MAEt':>7s} {'MAEf':>8s}")
    for off in offsets:
        ds = OffsetDataset(base, off)
        all_pred = predict_dataset(modules, ds, cfg, device,
                                   batch_size=args.batch_size, max_examples=args.max_examples)
        score, m = ar_score(all_pred, all_truth, cfg)
        d, c = m["detection"], m["count"]
        ts, fs = m["regression"]["t_start"], m["regression"]["f1"]
        mae_t = ts["mae"] if ts else float("nan")
        mae_f = fs["mae"] if fs else float("nan")
        results.append({"offset_db": off, "ar_score": score, "f1": d["f1"],
                        "recall": d["recall"], "precision": d["precision"],
                        "count_exact": c["exact_match_rate"],
                        "mae_t_start": mae_t, "mae_f1": mae_f})
        print(f"  {off:>+7.1f}d {score:>8.4f} {d['f1']:>7.3f} {d['recall']:>7.3f} "
              f"{d['precision']:>7.3f} {c['exact_match_rate']:>6.3f} {mae_t:>6.3f}s {mae_f:>7.1f}H")

    csv_path = os.path.join(args.outdir, "offset_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=FIG)
    xs = [r["offset_db"] for r in results]
    ax.plot(xs, [r["f1"] for r in results], marker="o", color=PRIMARY, label="F1")
    ax.plot(xs, [r["recall"] for r in results], marker="s", color="#55A868", label="Recall")
    ax.plot(xs, [r["precision"] for r in results], marker="^", color=ACCENT, label="Precision")
    ax.axvline(0.0, color=REFERENCE, ls=":", lw=1.4, label="Unmodified input")
    finish(ax, "Level shift applied to the input [dB]", "Rate",
           f"Sensitivity to a global level shift ({args.split} split)",
           legend=True, rate_axis=True)
    save(fig, os.path.join(args.outdir, "offset_sweep.png"))


if __name__ == "__main__":
    main()
