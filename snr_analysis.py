"""
Analyzes a trained checkpoint's detection behavior as a function of SNR, and the sweep-type
classification as a function of the curvature gap between the two swept shapes.

Requires the metadata file META.npy saved by pulses.generate_train alongside the inputs, with one
row [snr, distance, n_pulses, seg_idx] per example. Four analyses are produced:

  1. Recall vs SNR         - fraction of true pulses detected, per SNR bin, with the SNR at which
                             90% recall is reached (SNR90).
  2. Precision vs SNR      - fraction of detections that are real, per SNR bin, with the false-alarm
                             counts. False alarms are attributed to their example's SNR.
  3. Sweep accuracy vs df  - LFM/HFM accuracy against the curvature gap
                             df = (f2-f1)^2 / (2*(f1+f2)), the mid-pulse separation between a
                             linear and a hyperbolic sweep with the same endpoints; shown for all
                             swept pulses and for the high-SNR slice alone.
  4. False-alarm autopsy   - one row per false alarm, testing the echo fingerprint: does it start
                             shortly after a true pulse in the same frequency band?

By default the analysis runs ONLY on the held-out test split, reconstructed deterministically from
the checkpoint's stored config so it is identical to the split used by train_model.py and
evaluate.py. Use --split to override (e.g. 'all' for a quick sanity pass, but its numbers include
training examples and are optimistic).

Results are printed, saved as CSV files, and plotted as figures.

Run:
    python snr_analysis.py --checkpoint best_model.pth --input INPUT.npy --target padded_sequences.npy --meta META.npy
"""

import argparse
import csv
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_models, strides_per_block, CONFIG
from dataset_io import make_splits, MemmapPulseDataset
from metrics_core import match_pulses, predict_dataset, truth_pulses
from plot_style import FIG, PRIMARY, ACCENT, REFERENCE, finish, save


class_names = {0: "cw", 1: "lfm", 2: "hfm", 3: "eos"}


def delta_f(f1, f2):

    """
    Computes the curvature gap between a linear and a hyperbolic sweep with the same endpoint
    frequencies: the separation of their instantaneous frequencies at the pulse midpoint,
    (f2 - f1)^2 / (2 * (f1 + f2)). This is the size of the only feature that distinguishes the two
    swept types on the spectrogram.

    ----------

    Parameters:
        f1 (float) - start frequency in Hz.
        f2 (float) - end frequency in Hz.

    Returns:
        (float) - the curvature gap in Hz.
    """

    return (f2 - f1) ** 2 / (2.0 * (f1 + f2))


def frequency_cell(cfg, n_freq_bins=1025):

    """
    Computes the model's effective frequency resolution: the frequency span covered by one row of
    the final CNN feature map, given the input bin width and the total frequency downsampling of
    the configuration's stride list.

    ----------

    Parameters:
        cfg (dict) - the model configuration.
        n_freq_bins (int) - default 1025. Number of frequency bins in the input spectrogram.

    Returns:
        (float) - the frequency span of one feature cell, in Hz.
    """

    factor = 1
    for s in strides_per_block(cfg):
        factor *= s[0]
    return cfg["freq_max"] / (n_freq_bins - 1) * factor


def load_model(checkpoint_path, input_hw, device):

    """
    Loads a checkpoint and rebuilds the exact architecture it was trained with, reading the
    configuration from the checkpoint itself when present and falling back to model.CONFIG for
    older checkpoints.

    ----------

    Parameters:
        checkpoint_path (str) - path to the .pth checkpoint file.
        input_hw (tuple) - the (height, width) of a single input spectrogram.
        device (torch.device) - device to load the model on.

    Returns:
        modules (tuple) - (encoder_cnn, encoder_lstm, decoder) in evaluation mode.
        cfg (dict) - the configuration the model was built from.
    """

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", CONFIG)
    if "config" not in ckpt:
        print("Checkpoint has no stored config; falling back to model.CONFIG.")

    encoder_cnn, encoder_lstm, decoder = build_models(input_hw, cfg, device)
    encoder_cnn.load_state_dict(ckpt["encoder_cnn_state_dict"])
    encoder_lstm.load_state_dict(ckpt["encoder_lstm_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    for m in (encoder_cnn, encoder_lstm, decoder):
        m.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('validation_loss', float('nan')):.4f}")
    return (encoder_cnn, encoder_lstm, decoder), cfg


def collect_records(all_pred, all_truth, meta):

    """
    Matches predictions to truth for every example and flattens the outcome into per-pulse records
    for the SNR and curvature analyses.

    ----------

    Parameters:
        all_pred (list) - predicted pulses per example.
        all_truth (list) - true pulses per example.
        meta (ndarray) - metadata rows [snr, distance, n_pulses, seg_idx] per example.

    Returns:
        recall_rec (list) - one (snr, detected) tuple per true pulse.
        fp_rec (list) - one dict per false alarm: snr, example index, the false pulse, and the true
                        pulses of its example.
        sweep_rec (list) - one (snr, delta_f, correct) tuple per matched swept pulse.
    """

    recall_rec, fp_rec, sweep_rec = [], [], []

    for i, (pred, truth) in enumerate(zip(all_pred, all_truth)):
        snr = float(meta[i, 0])
        pairs, false_alarms, misses = match_pulses(pred, truth)
        matched_truth = {j for _, j in pairs}

        for j in range(len(truth)):
            recall_rec.append((snr, j in matched_truth))

        for (pi, j) in pairs:
            if truth[j]["type"] in ("lfm", "hfm"):
                sweep_rec.append((snr,
                                  delta_f(truth[j]["f1"], truth[j]["f2"]),
                                  pred[pi]["type"] == truth[j]["type"]))

        for pi in false_alarms:
            fp_rec.append({"snr": snr, "example": i, "fp": pred[pi], "truth": truth})

    return recall_rec, fp_rec, sweep_rec


def bin_rate(records, edges):

    """
    Bins (value, success) records into rate statistics per bin.

    ----------

    Parameters:
        records (list) - (value, success) tuples; NaN values are dropped.
        edges (ndarray) - bin edges, length n_bins + 1.

    Returns:
        stats (list) - one dict per bin: lo, hi, center, n, rate, se (binomial standard error);
                       rate and se are NaN for empty bins.
    """

    stats = []
    vals = np.array([v for v, _ in records], float)
    hits = np.array([h for _, h in records], float)
    ok = ~np.isnan(vals)
    vals, hits = vals[ok], hits[ok]

    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        n = int(m.sum())
        if n == 0:
            stats.append({"lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                          "n": 0, "rate": float("nan"), "se": float("nan")})
            continue
        p = hits[m].mean()
        stats.append({"lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                      "n": n, "rate": float(p),
                      "se": float(np.sqrt(max(p * (1 - p), 1e-12) / n))})
    return stats


def snr90(stats):

    """
    Finds the SNR at which the binned recall curve first reaches 90%, by linear interpolation
    between bin centers.

    ----------

    Parameters:
        stats (list) - binned recall statistics from bin_rate, in ascending SNR order.

    Returns:
        (float or None) - the interpolated SNR at 90% recall, or None when it is never reached.
    """

    pts = [(s["center"], s["rate"]) for s in stats if s["n"] > 0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 < 0.9 <= y1:
            return x0 + (0.9 - y0) / (y1 - y0) * (x1 - x0)
    if pts and pts[0][1] >= 0.9:
        return pts[0][0]
    return None


def autopsy(fp_rec, offset_max=1.0, overlap_min=0.5):

    """
    Tests every false alarm for the echo fingerprint: a start shortly after the end of a true pulse
    of the same example, in an overlapping frequency band. False alarms with the fingerprint are
    labeled echo-like, the rest noise-like.

    ----------

    Parameters:
        fp_rec (list) - false-alarm records from collect_records.
        offset_max (float) - default 1.0. Largest start-after-parent-stop gap, in seconds, that
                             still counts as an echo.
        overlap_min (float) - default 0.5. Smallest band-overlap fraction that still counts.

    Returns:
        rows (list) - one dict per false alarm: example, t_start, offset, band_overlap, same_type,
                      snr, verdict.
    """

    rows = []
    for r in fp_rec:
        fp, truth = r["fp"], r["truth"]
        parents = [t for t in truth if t["t_stop"] <= fp["t_start"]]
        if parents:
            parent = max(parents, key=lambda t: t["t_stop"])
            offset = fp["t_start"] - parent["t_stop"]
            f_lo, f_hi = sorted((fp["f1"], fp["f2"]))
            p_lo, p_hi = sorted((parent["f1"], parent["f2"]))
            width = max(f_hi - f_lo, 1.0)
            overlap = max(0.0, min(f_hi, p_hi) - max(f_lo, p_lo)) / width
            same_type = fp["type"] == parent["type"]
            echo = offset <= offset_max and overlap >= overlap_min
        else:
            offset, overlap, same_type, echo = float("nan"), float("nan"), False, False
        rows.append({"example": r["example"], "t_start": fp["t_start"],
                     "offset": offset, "band_overlap": overlap,
                     "same_type": same_type, "snr": r["snr"],
                     "verdict": "echo-like" if echo else "noise-like"})
    return rows


def save_bins_csv(stats, path, value_name, extra=None):

    """
    Saves one binned curve as a CSV file.

    ----------

    Parameters:
        stats (list) - binned statistics from bin_rate.
        path (str) - output file path.
        value_name (str) - name of the rate column (e.g. 'recall').
        extra (dict or None) - default None. Optional extra columns, mapping name to a list of
                               per-bin values.

    Returns:
        None
    """

    fields = ["lo", "hi", "center", "n", value_name, "se"] + (list(extra) if extra else [])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i, s in enumerate(stats):
            row = [s["lo"], s["hi"], s["center"], s["n"], s["rate"], s["se"]]
            if extra:
                row += [extra[k][i] for k in extra]
            w.writerow(row)
    print(f"Saved {path}")


def errorbar_plot(stats_sets, labels, xlabel, ylabel, title, path,
                  hline=None, hline_label=None, vline=None, vline_label=None):

    """
    Draws one or more binned rate curves with binomial error bars. Reference lines always carry a
    legend entry, so nothing unexplained appears on the figure.

    ----------

    Parameters:
        stats_sets (list) - one or more binned statistics lists from bin_rate.
        labels (list) - one label per curve.
        xlabel (str) - x-axis label.
        ylabel (str) - y-axis label.
        title (str) - figure title.
        path (str) - output file path.
        hline (float or None) - default None. Horizontal reference line.
        hline_label (str or None) - default None. Legend entry for the horizontal line.
        vline (float or None) - default None. Vertical reference line.
        vline_label (str or None) - default None. Legend entry for the vertical line.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=FIG)
    colours = [PRIMARY, ACCENT, "#55A868"]
    markers = ["o", "s", "^"]
    for stats, label, colour, marker in zip(stats_sets, labels, colours, markers):
        pts = [s for s in stats if s["n"] > 0]
        ax.errorbar([s["center"] for s in pts], [s["rate"] for s in pts],
                    yerr=[s["se"] for s in pts], marker=marker, capsize=3,
                    label=label, color=colour)
    if hline is not None:
        ax.axhline(hline, color=REFERENCE, ls=":", lw=1.4, label=hline_label)
    if vline is not None:
        ax.axvline(vline, color=REFERENCE, ls="--", lw=1.4, label=vline_label)
    finish(ax, xlabel, ylabel, title, legend=True, rate_axis=True)
    save(fig, path)


def main():

    """
    Parses arguments, runs inference on the analysis dataset, and produces the four analyses:
    recall vs SNR (with SNR90), precision and false alarms vs SNR, sweep accuracy vs curvature gap
    (all pulses and the high-SNR slice), and the false-alarm autopsy. Results are printed, saved as
    CSV files, and plotted.

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
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"],
                    help="which data split to analyze; 'test' (default) reproduces the held-out "
                         "conditions, 'all' includes training examples and gives OPTIMISTIC numbers")
    ap.add_argument("--snr-bins", default="-15,-10,-5,0,5,10,20,30,40",
                    help="comma-separated SNR bin edges in dB (match SNR_MIN/SNR_MAX in data_config.py)")
    ap.add_argument("--dfreq-bins", default="0,50,100,200,400,700,1100,1400",
                    help="comma-separated curvature-gap bin edges in Hz")
    ap.add_argument("--high-snr", type=float, default=25.0,
                    help="threshold for the high-SNR slice of the curvature analysis "
                         "(must lie inside the SNR_MIN..SNR_MAX training range)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outdir", default="snr_analysis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(args.input, mmap_mode="r")
    y = np.load(args.target, allow_pickle=True).astype(np.float32)
    meta = np.load(args.meta)
    assert len(X) == len(y) == len(meta), "INPUT, targets, and META must be index-aligned"

    modules, cfg = load_model(args.checkpoint, (X.shape[1], X.shape[2]), device)

    # Restrict the analysis to one split (default: the held-out test set). The split is
    # reconstructed deterministically from the checkpoint's own config, so it is identical
    # to the one used by train_model.py and evaluate.py. Running on 'all' mixes in ~2/3
    # training examples and makes every number optimistic.
    idx_train, idx_val, idx_test = make_splits(meta, cfg)
    split_map = {"train": idx_train, "val": idx_val, "test": idx_test,
                 "all": np.arange(len(y))}
    idx = np.sort(np.asarray(split_map[args.split]))
    ds = MemmapPulseDataset(args.input, y, idx)
    meta = meta[idx]
    print(f"{len(idx)} examples ({args.split} split of {len(y)} total) | device {device}")

    all_pred = predict_dataset(modules, ds, cfg, device, batch_size=args.batch_size)
    all_truth = truth_pulses(ds.targets, cfg, scaled=False)
    recall_rec, fp_rec, sweep_rec = collect_records(all_pred, all_truth, meta)

    snr_edges = np.array([float(v) for v in args.snr_bins.split(",")])
    df_edges = np.array([float(v) for v in args.dfreq_bins.split(",")])

    # --- 1. recall vs SNR ---
    rec_stats = bin_rate(recall_rec, snr_edges)
    thr = snr90(rec_stats)
    print("\n--- RECALL vs SNR ---")
    print(f"  {'bin [dB]':>16s} {'n':>7s} {'recall':>8s} {'+/-':>7s}")
    for s in rec_stats:
        print(f"  [{s['lo']:5.0f},{s['hi']:5.0f}) {s['n']:>7d} "
              f"{s['rate']:>8.3f} {s['se']:>7.3f}")
    print(f"  SNR90 (90% recall reached): "
          + (f"{thr:.1f} dB" if thr is not None else "not reached"))

    # --- 2. precision and false alarms vs SNR ---
    tp_by_bin, fp_by_bin, prec_stats = [], [], []
    tps = np.array([s for s, hit in recall_rec if hit and not np.isnan(s)], float)
    fps = np.array([r["snr"] for r in fp_rec if not np.isnan(r["snr"])], float)
    n_fp_noise_only = sum(1 for r in fp_rec if np.isnan(r["snr"]))
    print("\n--- PRECISION / FALSE ALARMS vs SNR ---")
    print(f"  {'bin [dB]':>16s} {'TP':>7s} {'FP':>5s} {'precision':>10s}")
    for lo, hi in zip(snr_edges[:-1], snr_edges[1:]):
        tp = int(((tps >= lo) & (tps < hi)).sum())
        fp = int(((fps >= lo) & (fps < hi)).sum())
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        se = float(np.sqrt(max(p * (1 - p), 1e-12) / (tp + fp))) if (tp + fp) else float("nan")
        tp_by_bin.append(tp); fp_by_bin.append(fp)
        prec_stats.append({"lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                           "n": tp + fp, "rate": p, "se": se})
        print(f"  [{lo:5.0f},{hi:5.0f}) {tp:>7d} {fp:>5d} {p:>10.3f}")
    if n_fp_noise_only:
        print(f"  (plus {n_fp_noise_only} false alarms in noise-only examples, no SNR)")

    # --- 3. sweep accuracy vs curvature gap ---
    all_sw = [(d, c) for _s, d, c in sweep_rec]
    hi_sw = [(d, c) for s, d, c in sweep_rec if not np.isnan(s) and s >= args.high_snr]
    sw_all = bin_rate(all_sw, df_edges)
    sw_hi = bin_rate(hi_sw, df_edges)
    cell = frequency_cell(cfg, n_freq_bins=X.shape[1])
    print(f"\n--- SWEEP ACCURACY vs CURVATURE GAP (model cell ~ {cell:.0f} Hz) ---")
    print(f"  {'bin [Hz]':>16s} {'n(all)':>7s} {'acc':>7s} {'n(hi)':>7s} {'acc(hi)':>8s}")
    for a, h in zip(sw_all, sw_hi):
        print(f"  [{a['lo']:5.0f},{a['hi']:5.0f}) {a['n']:>7d} {a['rate']:>7.3f} "
              f"{h['n']:>7d} {h['rate']:>8.3f}")

    # --- 4. false-alarm autopsy ---
    rows = autopsy(fp_rec)
    n_echo = sum(r["verdict"] == "echo-like" for r in rows)
    print(f"\n--- FALSE-ALARM AUTOPSY ({len(rows)} false alarms) ---")
    print(f"  {'ex':>6s} {'t_start':>8s} {'offset':>8s} {'overlap':>8s} "
          f"{'same':>5s} {'snr':>7s}  verdict")
    for r in rows:
        print(f"  {r['example']:>6d} {r['t_start']:>7.2f}s {r['offset']:>7.2f}s "
              f"{r['band_overlap']:>8.2f} {str(r['same_type']):>5s} {r['snr']:>7.1f}  {r['verdict']}")
    print(f"  SUMMARY: {n_echo} of {len(rows)} false alarms carry the echo fingerprint; "
          f"{len(rows) - n_echo} look like noise.")

    # --- outputs ---
    save_bins_csv(rec_stats, os.path.join(args.outdir, "recall_vs_snr.csv"), "recall")
    save_bins_csv(prec_stats, os.path.join(args.outdir, "precision_vs_snr.csv"), "precision",
                  extra={"tp": tp_by_bin, "fp": fp_by_bin})
    save_bins_csv(sw_all, os.path.join(args.outdir, "sweep_acc_vs_dfreq_all.csv"), "accuracy")
    save_bins_csv(sw_hi, os.path.join(args.outdir, "sweep_acc_vs_dfreq_highsnr.csv"), "accuracy")
    with open(os.path.join(args.outdir, "fp_autopsy.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["example", "t_start", "offset", "band_overlap",
                                           "same_type", "snr", "verdict"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {os.path.join(args.outdir, 'fp_autopsy.csv')}")

    errorbar_plot([rec_stats], ["Recall"], "SNR [dB]", "Recall",
                  "Recall against SNR", os.path.join(args.outdir, "recall_vs_snr.png"),
                  vline=thr, vline_label=(f"SNR90 = {thr:.1f} dB" if thr is not None else None))
    errorbar_plot([prec_stats], ["Precision"], "SNR [dB]", "Precision",
                  "Precision against SNR", os.path.join(args.outdir, "precision_vs_snr.png"))
    errorbar_plot([sw_all, sw_hi], ["All swept pulses", f"SNR at least {args.high_snr:.0f} dB"],
                  "Curvature gap between an LFM and an HFM sweep [Hz]", "Correct sweep type",
                  "Telling LFM from HFM against how far apart their sweeps are",
                  os.path.join(args.outdir, "sweep_acc_vs_dfreq.png"),
                  hline=0.5, hline_label="Chance (two sweep types)",
                  vline=cell, vline_label=f"One CNN frequency cell = {cell:.0f} Hz")


if __name__ == "__main__":
    main()
