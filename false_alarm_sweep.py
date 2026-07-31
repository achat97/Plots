"""
Operational false-alarm measurement (#6): runs the full detection pipeline over one or more
PULSE-FREE recordings and reports false alarms per hour as a function of the detection-confidence
threshold. Every merged detection is by definition a false alarm, so this gives the operating
curve an operator cares about: pick the threshold where FA/hour is acceptable, then read the
matching recall off snr_analysis at that same threshold.

Two rules for the input recordings:
  1. They must contain no real pulses (otherwise true detections are counted as false alarms).
  2. They must NOT have contributed noise segments to training - the model partly memorizes
     trained backgrounds and looks artificially quiet on them. Use a held-out recording.
The upsampling warning from detect_pulses applies here too: recordings should be natively >= FS.

Detection runs ONCE per recording at threshold 0; the thresholds are then applied to the merged
detections exactly as detect_pulses --min-confidence would (filtering after merging), so every
row of the sweep corresponds to a real deployment setting.

Run:
    python false_alarm_sweep.py --wav quiet1.wav quiet2.wav --checkpoint best_model.pth
"""

import argparse
import csv
import os

import numpy as np
import scipy.signal as sps
import torch
import matplotlib.pyplot as plt

from model import CONFIG
from data_config import FS as DEFAULT_FS, NPERSEG, NOVERLAP
from metrics_core import gate as default_gate, freq_gate as default_freq_gate
from plot_style import FIG, PRIMARY, finish, save
from detect_pulses import (read_wav, make_windows, windows_to_tensor, load_checkpoint,
                           predict_windows, to_global, merge_detections)


def main():

    """
    Parses arguments, runs detection over every recording, sweeps the confidence thresholds over
    the merged detections, prints the FA/hour table, and saves a CSV and a figure.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, nargs="+",
                    help="one or more PULSE-FREE .wav recordings, not used in training")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pth checkpoint")
    ap.add_argument("--thresholds", default="0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99",
                    help="comma-separated detection-confidence thresholds to sweep")
    ap.add_argument("--fs", type=int, default=None,
                    help="training sample rate in Hz (default: read from the checkpoint)")
    ap.add_argument("--hop", type=float, default=None,
                    help="hop between windows in seconds (default: half the window)")
    ap.add_argument("--gate", type=float, default=default_gate,
                    help="merge gate in seconds (same as detect_pulses)")
    ap.add_argument("--freq-gate", type=float, default=default_freq_gate,
                    help="merge band padding in Hz (same as detect_pulses)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outdir", default="fa_sweep")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peek_cfg = torch.load(args.checkpoint, map_location="cpu").get("config", CONFIG)
    fs = args.fs if args.fs is not None else int(peek_cfg.get("fs", DEFAULT_FS))
    probe = sps.stft(np.zeros(int(fs * peek_cfg["time_max"])), fs=fs,
                     nperseg=NPERSEG, noverlap=NOVERLAP)[2]
    modules, cfg = load_checkpoint(args.checkpoint, probe.shape, device)
    window_s = cfg["time_max"]
    hop_s = args.hop if args.hop is not None else window_s / 2

    # Detect once per recording; merging never crosses recordings.
    merged_all = []
    total_hours = 0.0
    for path in args.wav:
        data, duration = read_wav(path, fs)
        total_hours += duration / 3600.0
        windows = make_windows(data, fs, window_s, hop_s)
        X = windows_to_tensor(windows, fs)
        per_window = predict_windows(modules, X, cfg, device, args.batch_size)
        detections = to_global(per_window, windows, window_s)
        merged = merge_detections(detections, args.gate, args.freq_gate)
        merged_all.extend(merged)
        print(f"{path}: {duration:.1f} s -> {len(merged)} merged detections (all false alarms)")

    print(f"\nTotal audited duration: {total_hours:.3f} h "
          f"| {len(merged_all)} false alarms at threshold 0\n")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    conf = np.array([p["confidence_det"] for p in merged_all])
    types = np.array([p["type"] for p in merged_all])

    rows = []
    print(f"  {'thr':>5s} {'FA':>6s} {'FA/hour':>9s} {'cw':>5s} {'lfm':>5s} {'hfm':>5s}")
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        row = {"threshold": t, "n_fa": n, "fa_per_hour": n / total_hours,
               "cw": int((types[keep] == "cw").sum()),
               "lfm": int((types[keep] == "lfm").sum()),
               "hfm": int((types[keep] == "hfm").sum())}
        rows.append(row)
        print(f"  {t:>5.2f} {n:>6d} {row['fa_per_hour']:>9.2f} "
              f"{row['cw']:>5d} {row['lfm']:>5d} {row['hfm']:>5d}")

    csv_path = os.path.join(args.outdir, "fa_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=FIG)
    ax.plot([r["threshold"] for r in rows], [r["fa_per_hour"] for r in rows],
            marker="o", color=PRIMARY)
    ax.set_ylim(bottom=0)
    finish(ax, "Detection confidence threshold", "False alarms per hour",
           f"False alarms over {total_hours:.2f} h of pulse-free recording")
    save(fig, os.path.join(args.outdir, "fa_sweep.png"))


if __name__ == "__main__":
    main()
