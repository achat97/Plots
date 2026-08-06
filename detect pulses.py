"""
Detects pulses in a .wav recording using a trained checkpoint and reports them at their
position in the whole time series.

The recording travels the exact same road as the training data: resample to the training
rate, 5 s windows, per-window z-score normalization, STFT, dB. The window slides with 50%
overlap so no pulse is lost on a window boundary; detections of the same pulse from
neighbouring windows are merged, keeping the most confident one. The architecture is read
from the checkpoint itself (falling back to model.CONFIG for old checkpoints), so the
script cannot mismatch the weights. Results are printed as a time-sorted table and always
saved as CSV; Excel and an annotated spectrogram figure are optional.

Run:
    python detect_pulses.py --wav recording.wav --checkpoint best_model.pth
    python detect_pulses.py --wav recording.wav --checkpoint best_model.pth --xlsx --plot
    python detect_pulses.py --wav recording.wav --checkpoint best_model.pth --plot \
           --plot-min-confidence 0.98 --max-plots 30 --plot-context 5

Times are reported in clock time (h:mm:ss with fractional seconds); the CSV and Excel exports keep
the numeric seconds columns as well. With --plot the script writes a spectrogram-free overview of
all detections plus one spectrogram crop per detection - a single spectrogram of a multi-hour
recording compresses every pulse to less than a pixel.
"""

import argparse
import csv
import os
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as sps
from scipy.io import wavfile
from scipy.stats import zscore

from model import CONFIG, build_models, concatenate_bidirectional_states
from metrics_core import gate as default_gate, freq_gate as default_freq_gate
from data_config import FS as DEFAULT_FS, NPERSEG, NOVERLAP
from table_io import save_table_xlsx
from plot_style import (FIG_WIDE, FONT, TYPE_COLORS, REFERENCE,
                        type_name, hms, hms_axis, finish, save)


class_names = {0: "cw", 1: "lfm", 2: "hfm", 3: "eos"}


def db(z, eps=1e-6):

    """
    Converts a magnitude input to decibels using 20*log10(|z|) with a reference of 1.0.
    A small offset (1e-6) is added to the magnitude to avoid taking the logarithm of zero.

    ----------

    Parameters:
        z (float, complex, or ndarray) - input value(s) to convert to dB. Can be real or complex.

    Returns:
        (float or ndarray) - the input expressed in decibels.
    """

    return 20 * np.log10(np.abs(z) + eps)


def read_wav(path, fs_target):

    """
    Reads a .wav file, averages stereo channels to mono, and resamples to the target rate using
    polyphase filtering. Timestamps are preserved: a point at t seconds in the original recording
    is at t seconds in the resampled signal.

    ----------

    Parameters:
        path (str) - path to the .wav file.
        fs_target (int) - target sampling frequency in Hz (the rate the model was trained at).

    Returns:
        data (ndarray) - 1-D float64 array of the resampled signal.
        duration (float) - duration of the recording in seconds.
    """

    rate, data = wavfile.read(path)
    data = data.astype(np.float64)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if rate < fs_target:
        print(f"WARNING: native rate {rate} Hz < target {fs_target} Hz. Upsampling cannot create "
              f"content above {rate / 2:.0f} Hz, so the band {rate / 2:.0f}-{fs_target / 2:.0f} Hz "
              f"of the spectrogram will be empty (near the dB epsilon floor), unlike the training "
              f"noise. Detections there are unreliable; prefer recordings sampled natively at "
              f">= {fs_target} Hz.")
    if rate != fs_target:
        frac = Fraction(fs_target, rate).limit_denominator(1000)
        data = sps.resample_poly(data, frac.numerator, frac.denominator)

    duration = len(data) / fs_target
    print(f"Read {path}: {duration:.2f} s at {rate} Hz -> resampled to {fs_target} Hz")
    return data, duration


def make_windows(data, fs, window_s, hop_s):

    """
    Slides a window over the signal and z-score normalizes each window independently, exactly as the
    training segments were normalized. A final partial window is normalized on its real samples and
    then zero-padded to full length (zeros sit at the mean level); partial windows shorter than half
    a second are dropped.

    ----------

    Parameters:
        data (ndarray) - 1-D array of the full resampled signal.
        fs (int) - sampling frequency of the data in Hz.
        window_s (float) - window length in seconds (the model's training window).
        hop_s (float) - hop between window starts in seconds.

    Returns:
        windows (list) - list of (t0, segment, valid_s) tuples, where t0 is the window's start time
                         in the recording, segment is the normalized window of length window_s * fs,
                         and valid_s is how many seconds of it are real samples (< window_s only for
                         the padded tail).
    """

    window_samples = int(round(window_s * fs))
    hop_samples = int(round(hop_s * fs))

    windows = []
    start = 0
    while start < len(data):
        seg = data[start:start + window_samples]
        valid_s = len(seg) / fs

        if len(seg) < window_samples:
            if valid_s < 0.5:
                break
            std = seg.std()
            seg = (seg - seg.mean()) / std if std > 0 else np.zeros_like(seg)
            seg = np.concatenate([seg, np.zeros(window_samples - len(seg))])
        else:
            std = seg.std()
            seg = zscore(seg) if std > 0 else np.zeros_like(seg)

        windows.append((start / fs, seg, valid_s))

        if start + window_samples >= len(data):
            break
        start += hop_samples

    print(f"Split into {len(windows)} windows of {window_s:.1f} s (hop {hop_s:.2f} s)")
    return windows


def windows_to_tensor(windows, fs):

    """
    Computes the dB spectrogram of every window with the training STFT settings (segment length
    2048, 50% overlap) and stacks them into a batched model input tensor.

    ----------

    Parameters:
        windows (list) - list of (t0, segment, valid_s) tuples from make_windows.
        fs (int) - sampling frequency of the data in Hz.

    Returns:
        X (Tensor) - 4-D float tensor of shape (n_windows, 1, frequency_bins, time_bins).
    """

    specs = []
    for _t0, seg, _valid in windows:
        f, t, zxx = sps.stft(seg, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
        specs.append(torch.from_numpy(db(zxx)).float().unsqueeze(0))

    X = torch.stack(specs)
    print(f"Spectrograms: {tuple(X.shape)}")
    return X


def load_checkpoint(path, input_hw, device):

    """
    Loads a checkpoint and rebuilds the exact architecture it was trained with. The configuration is
    read from the checkpoint itself, so the weights can never be poured into a mismatched model;
    checkpoints saved before the configuration was stored fall back to model.CONFIG.

    ----------

    Parameters:
        path (str) - path to the .pth checkpoint file.
        input_hw (tuple) - (frequency_bins, time_bins) of one input spectrogram.
        device (torch.device) - device to load the model onto.

    Returns:
        modules (tuple) - (encoder_cnn, encoder_lstm, decoder) in evaluation mode.
        cfg (dict) - the configuration the model was built from.
    """

    ckpt = torch.load(path, map_location=device)

    if "config" in ckpt:
        cfg = ckpt["config"]
        print(f"Loaded architecture from checkpoint (epoch {ckpt.get('epoch', '?')}, "
              f"val_loss {ckpt.get('validation_loss', float('nan')):.4f})")
    else:
        cfg = CONFIG
        print("Checkpoint has no stored config; falling back to model.CONFIG "
              "(make sure it matches the checkpoint's architecture)")

    encoder_cnn, encoder_lstm, decoder = build_models(input_hw, cfg, device)
    encoder_cnn.load_state_dict(ckpt["encoder_cnn_state_dict"])
    encoder_lstm.load_state_dict(ckpt["encoder_lstm_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    for m in (encoder_cnn, encoder_lstm, decoder):
        m.eval()

    return (encoder_cnn, encoder_lstm, decoder), cfg


@torch.no_grad()
def predict_windows(modules, X, cfg, device, batch_size=8):

    """
    Runs autoregressive inference on every window and returns the detected pulses in window-local
    physical units. Each pulse carries two scores from the classifier softmax at its decoding step:
    confidence_det = 1 - P(EOS), how sure the model is that a pulse (of any type) is present, and
    confidence_type = P(predicted type), how sure it is of the chosen type. Neither is a calibrated
    probability (the heads are trained with teacher forcing); they are ordering heuristics.

    ----------

    Parameters:
        modules (tuple) - (encoder_cnn, encoder_lstm, decoder) in evaluation mode.
        X (Tensor) - batched spectrograms of shape (n_windows, 1, frequency_bins, time_bins).
        cfg (dict) - the model configuration (supplies the rescaling constants).
        device (torch.device) - device to run on.
        batch_size (int) - default 8. Inference batch size.

    Returns:
        per_window (list) - one list of pulse dicts per window, keys type, t_start, t_stop, f1, f2,
                            confidence_det, confidence_type, with times local to the window.
    """

    encoder_cnn, encoder_lstm, decoder = modules
    per_window = []

    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size].to(device)
        feats = encoder_cnn(xb)
        _, (h, c) = encoder_lstm(feats)
        if encoder_lstm.bidirectional:
            h, c = concatenate_bidirectional_states(h, c)
        out, _ = decoder.generate(h, c)

        for b in range(xb.shape[0]):
            cls = out['classification'][b]
            pulses = []
            for s in range(cls.shape[0]):
                probs = F.softmax(cls[s], dim=-1)
                k = int(torch.argmax(cls[s]))
                if k == cfg["eos_token_id"]:
                    break
                pulses.append({
                    "type":            class_names[k],
                    "t_start":         float(out['start_time'][b, s, 0]) * cfg["time_max"],
                    "t_stop":          float(out['end_time'][b, s, 0]) * cfg["time_max"],
                    "f1":              float(out['start_freq'][b, s, 0]) * cfg["freq_max"],
                    "f2":              float(out['end_freq'][b, s, 0]) * cfg["freq_max"],
                    "confidence_det":  float(1.0 - probs[cfg["eos_token_id"]]),
                    "confidence_type": float(probs[k]),
                })
            per_window.append(pulses)
        print(f"  inference {min(start + batch_size, X.shape[0])}/{X.shape[0]} windows", end="\r")

    print("")
    return per_window


def to_global(per_window, windows, window_s):

    """
    Converts window-local detections to positions in the whole recording. Detections lying entirely
    in a padded tail region are dropped; detections that only end inside padding are kept and
    flagged. Each detection also records how centrally its pulse sat in the window, used to break
    confidence ties when merging.

    ----------

    Parameters:
        per_window (list) - one list of pulse dicts per window, from predict_windows.
        windows (list) - list of (t0, segment, valid_s) tuples from make_windows.
        window_s (float) - window length in seconds.

    Returns:
        detections (list) - flat list of pulse dicts with global t_start/t_stop, keys type, t_start,
                            t_stop, f1, f2, confidence, window, in_padding, centrality.
    """

    detections = []
    for w, pulses in enumerate(per_window):
        t0, _seg, valid_s = windows[w]
        for p in pulses:
            if p["t_start"] >= valid_s:
                continue
            center = 0.5 * (p["t_start"] + p["t_stop"])
            detections.append({
                "type":            p["type"],
                "t_start":         t0 + p["t_start"],
                "t_stop":          t0 + p["t_stop"],
                "f1":              p["f1"],
                "f2":              p["f2"],
                "confidence_det":  p["confidence_det"],
                "confidence_type": p["confidence_type"],
                "window":          w,
                "in_padding":      p["t_stop"] > valid_s,
                "centrality":      abs(center - window_s / 2),
            })
    return detections


def merge_detections(detections, gate, freq_gate):

    """
    Merges detections of the same physical pulse seen from overlapping windows. Detections sorted by
    start time join an existing cluster when they satisfy BOTH conditions: (1) they start within the
    gate of the cluster's first start or overlap the cluster in time, and (2) their frequency band,
    padded by freq_gate, overlaps the cluster's band envelope. The band condition keeps two
    simultaneous pulses in different bands from collapsing into one detection, which a time-only
    rule would do. Every existing cluster is checked (not just the most recent), so interleaved
    pulse streams in different bands each keep their own cluster. Each cluster is reported once,
    represented by its member with the highest detection confidence (most central on ties);
    clusters whose members disagree on the pulse type are flagged with the set of types seen.

    ----------

    Parameters:
        detections (list) - flat list of global-time pulse dicts from to_global.
        gate (float) - maximum start-time gap in seconds to still call two detections the same pulse.
        freq_gate (float) - band padding in Hz for the frequency-overlap condition (also gives CW
                            pulses, whose band is a line, a finite width to overlap with).

    Returns:
        merged (list) - one pulse dict per physical pulse, sorted by start time, with the added keys
                        n_windows, windows, and type_conflict.
    """

    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["t_start"])
    clusters = []

    for det in detections:
        d_lo, d_hi = sorted((det["f1"], det["f2"]))
        placed = False
        for cluster in clusters:
            cluster_start = cluster[0]["t_start"]
            cluster_stop = max(d["t_stop"] for d in cluster)
            time_ok = (det["t_start"] - cluster_start <= gate
                       or det["t_start"] < cluster_stop)
            c_lo = min(min(d["f1"], d["f2"]) for d in cluster)
            c_hi = max(max(d["f1"], d["f2"]) for d in cluster)
            band_ok = (d_lo - freq_gate) <= c_hi and (c_lo - freq_gate) <= d_hi
            if time_ok and band_ok:
                cluster.append(det)
                placed = True
                break
        if not placed:
            clusters.append([det])

    merged = []
    for cluster in clusters:
        rep = min(cluster, key=lambda d: (-d["confidence_det"], d["centrality"]))
        types = sorted({d["type"] for d in cluster})
        merged.append({
            "type":            rep["type"],
            "t_start":         rep["t_start"],
            "t_stop":          rep["t_stop"],
            "f1":              rep["f1"],
            "f2":              rep["f2"],
            "confidence_det":  rep["confidence_det"],
            "confidence_type": rep["confidence_type"],
            "n_windows":       len(cluster),
            "windows":         sorted({d["window"] for d in cluster}),
            "in_padding":      rep["in_padding"],
            "type_conflict":   "" if len(types) == 1 else "/".join(types),
        })
    merged.sort(key=lambda p: p["t_start"])
    return merged


def print_table(pulses):

    """
    Prints the merged detections as a time-sorted table.

    ----------

    Parameters:
        pulses (list) - merged pulse dicts from merge_detections.

    Returns:
        None
    """

    print("\n" + "=" * 88)
    print(f"DETECTED PULSES ({len(pulses)})")
    print("=" * 88)
    if not pulses:
        print("  (none)")
        return

    print(f"  {'#':>3s} {'type':4s} {'t_start':>13s} {'t_stop':>13s} "
          f"{'f_start':>9s} {'f_stop':>9s} {'conf':>6s} {'type_c':>6s} {'wins':>5s}  flags")
    for i, p in enumerate(pulses):
        flags = []
        if p["type_conflict"]:
            flags.append(f"type? {p['type_conflict']}")
        if p["in_padding"]:
            flags.append("in padded tail")
        print(f"  {i:>3d} {p['type']:4s} {hms(p['t_start']):>13s} {hms(p['t_stop']):>13s} "
              f"{p['f1']:>8.0f}H {p['f2']:>8.0f}H {p['confidence_det']:>6.2f} "
              f"{p['confidence_type']:>6.2f} {p['n_windows']:>5d}  {', '.join(flags)}")
    print("=" * 88)


def save_csv(pulses, path):

    """
    Saves the merged detections as a CSV file.

    ----------

    Parameters:
        pulses (list) - merged pulse dicts from merge_detections.
        path (str) - output file path for the CSV.

    Returns:
        None
    """

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pulse_id", "type", "t_start_s", "t_stop_s", "t_start_hms", "t_stop_hms",
                    "f_start_hz", "f_stop_hz",
                    "confidence_det", "confidence_type", "n_windows", "windows",
                    "in_padding", "type_conflict"])
        for i, p in enumerate(pulses):
            w.writerow([i, p["type"], p["t_start"], p["t_stop"],
                        hms(p["t_start"]), hms(p["t_stop"]), p["f1"], p["f2"],
                        p["confidence_det"], p["confidence_type"], p["n_windows"],
                        " ".join(str(x) for x in p["windows"]),
                        int(p["in_padding"]), p["type_conflict"]])
    print(f"Saved {path}")


def save_xlsx(pulses, path):

    """
    Saves the merged detections as an Excel file. Requires openpyxl; if it is not installed, a
    message is printed and the CSV remains the output.

    ----------

    Parameters:
        pulses (list) - merged pulse dicts from merge_detections.
        path (str) - output file path for the .xlsx file.

    Returns:
        None
    """

    header = ["pulse_id", "type", "t_start_s", "t_stop_s", "t_start_hms", "t_stop_hms",
              "f_start_hz", "f_stop_hz", "confidence_det", "confidence_type", "n_windows",
              "windows", "in_padding", "type_conflict"]
    rows = [[i, p["type"], p["t_start"], p["t_stop"], hms(p["t_start"]), hms(p["t_stop"]),
             p["f1"], p["f2"], p["confidence_det"], p["confidence_type"], p["n_windows"],
             " ".join(str(x) for x in p["windows"]), int(p["in_padding"]), p["type_conflict"]]
            for i, p in enumerate(pulses)]
    save_table_xlsx(header, rows, path, sheet="detections")


def draw_pulse_box(ax, pulse, f_lo, f_hi, min_frac=0.02):

    """
    Draws one detection box without hiding the signal inside it. A CW pulse has f1 == f2, so its
    box would collapse onto the ridge and cover it; the box is expanded symmetrically to a minimum
    height of min_frac of the visible frequency range. The box is dashed and edge-only, and the
    type label sits above it.

    ----------

    Parameters:
        ax (Axes) - the axes to draw on.
        pulse (dict) - pulse dict with keys type, t_start, t_stop, f1, f2.
        f_lo (float) - lower limit of the visible frequency range in Hz.
        f_hi (float) - upper limit of the visible frequency range in Hz.
        min_frac (float) - default 0.02. Minimum box height as a fraction of the range.

    Returns:
        None
    """

    from matplotlib.patches import Rectangle

    colour = TYPE_COLORS.get(pulse["type"], "white")
    lo, hi = sorted((pulse["f1"], pulse["f2"]))
    min_h = max((f_hi - f_lo) * min_frac, 1.0)
    if hi - lo < min_h:
        pad = 0.5 * (min_h - (hi - lo))
        lo, hi = lo - pad, hi + pad
    ax.add_patch(Rectangle((pulse["t_start"], lo), pulse["t_stop"] - pulse["t_start"], hi - lo,
                           fill=False, edgecolor=colour, linewidth=1.4, linestyle="--"))
    ax.text(pulse["t_start"], hi + (f_hi - f_lo) * 0.015, type_name(pulse["type"]),
            color=colour, fontsize=FONT["annot"], va="bottom")


def plot_overview(pulses, duration, path, fs):

    """
    Draws every detection over the whole recording without a spectrogram, which stays readable for
    a recording of any length. The upper panel shows each detection as a vertical line spanning its
    frequency band, coloured by type; the lower panel shows the detection confidence of the same
    pulses, so the two can be read against each other.

    ----------

    Parameters:
        pulses (list) - merged pulse dicts from merge_detections.
        duration (float) - length of the recording in seconds.
        path (str) - output file path.
        fs (int) - sampling frequency in Hz, used to set the frequency axis to Nyquist.

    Returns:
        None
    """

    import matplotlib.pyplot as plt

    fig, (ax_f, ax_c) = plt.subplots(2, 1, figsize=FIG_WIDE, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    seen = set()
    for p in pulses:
        lo, hi = sorted((p["f1"], p["f2"]))
        colour = TYPE_COLORS.get(p["type"], REFERENCE)
        label = type_name(p["type"]) if p["type"] not in seen else None
        seen.add(p["type"])
        mid = 0.5 * (p["t_start"] + p["t_stop"])
        ax_f.plot([mid, mid], [lo, hi], color=colour, lw=2.0, solid_capstyle="round",
                  label=label)
        ax_f.plot([mid], [0.5 * (lo + hi)], marker="o", ms=3, color=colour)
        ax_c.plot([mid, mid], [0.0, p["confidence_det"]], color=colour, lw=1.4)
        ax_c.plot([mid], [p["confidence_det"]], marker="o", ms=3, color=colour)

    ax_f.set_xlim(0, max(duration, 1e-3))
    ax_f.set_ylim(0, fs / 2)
    finish(ax_f, None, "Frequency [Hz]",
           f"{len(pulses)} detections over {hms(duration, 0)} "
           f"(vertical line: frequency band of one pulse)", legend=bool(seen))
    ax_c.set_ylim(0, 1.05)
    hms_axis(ax_c)
    finish(ax_c, "Time [h:mm:ss]", "Detection confidence", None)
    save(fig, path)


def plot_detection_crops(data, fs, pulses, stem, context=3.0, max_plots=20):

    """
    Saves one spectrogram CROP per detection instead of one spectrogram of the whole recording:
    each figure covers the pulse plus `context` seconds on either side, so the pulse is actually
    visible however long the file is. Only the cropped samples are transformed, so cost and memory
    depend on the crop length, not on the recording length. When more detections than max_plots
    survive, the most confident ones are plotted.

    ----------

    Parameters:
        data (ndarray) - 1-D array of the full resampled signal.
        fs (int) - sampling frequency of the data in Hz.
        pulses (list) - merged pulse dicts from merge_detections.
        stem (str) - output filename stem; files are '<stem>_det_XXX.png'.
        context (float) - default 3.0. Seconds of context kept on each side of the pulse.
        max_plots (int) - default 20. Maximum number of crops written (0 = no limit).

    Returns:
        paths (list) - the file paths written.
    """

    import matplotlib.pyplot as plt

    if not pulses:
        print("No detections to plot.")
        return []

    order = sorted(range(len(pulses)), key=lambda i: -pulses[i]["confidence_det"])
    if max_plots:
        order = order[:max_plots]
    order = sorted(order)                      # write them in chronological order

    paths = []
    for i in order:
        p = pulses[i]
        t0 = max(0.0, p["t_start"] - context)
        t1 = min(len(data) / fs, p["t_stop"] + context)
        seg = data[int(t0 * fs):int(t1 * fs)]
        if seg.size < NPERSEG:
            continue

        f, t, zxx = sps.stft(seg.astype(np.float32), fs=fs,
                             nperseg=NPERSEG, noverlap=NOVERLAP)
        S = db(zxx)
        del zxx

        fig, ax = plt.subplots(figsize=FIG_WIDE)
        im = ax.imshow(S, origin="lower", aspect="auto",
                       extent=[t0, t0 + float(t[-1]), float(f[0]), float(f[-1])], cmap="plasma")
        fig.colorbar(im, ax=ax, label="dB")
        draw_pulse_box(ax, p, float(f[0]), float(f[-1]))
        # neighbours that fall inside the same crop, drawn faintly for context
        for j, q in enumerate(pulses):
            if j != i and q["t_stop"] > t0 and q["t_start"] < t1:
                ax.add_patch(plt.Rectangle(
                    (q["t_start"], min(q["f1"], q["f2"])),
                    q["t_stop"] - q["t_start"], max(abs(q["f2"] - q["f1"]), 1.0),
                    fill=False, edgecolor="0.7", linewidth=1.0, linestyle=":"))

        hms_axis(ax)
        finish(ax, "Time [h:mm:ss]", "Frequency [Hz]",
               f"Detection {i}: {type_name(p['type'])}, "
               f"{hms(p['t_start'])} to {hms(p['t_stop'])}, "
               f"{min(p['f1'], p['f2']):.0f} to {max(p['f1'], p['f2']):.0f} Hz, "
               f"detection confidence {p['confidence_det']:.2f}")
        paths.append(save(fig, f"{stem}_det_{i:03d}.png", quiet=True))

    print(f"Saved {len(paths)} detection crops as {stem}_det_XXX.png")
    return paths


def main():

    """
    Parses arguments, runs the full detection pipeline on the recording (read and resample, window,
    normalize, spectrogram, predict, merge), prints the time-sorted detections, and saves them as
    CSV plus the optional Excel file and annotated spectrogram figure.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="path to the .wav recording")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pth checkpoint")
    ap.add_argument("--fs", type=int, default=None,
                    help="training sample rate in Hz (default: read from the checkpoint)")
    ap.add_argument("--hop", type=float, default=None,
                    help="hop between windows in seconds (default: half the window)")
    ap.add_argument("--gate", type=float, default=default_gate,
                    help="max start-time gap in seconds to merge detections of the same pulse")
    ap.add_argument("--freq-gate", type=float, default=default_freq_gate,
                    help="band padding in Hz: detections merge only if their frequency bands, "
                         "padded by this, overlap (same convention as evaluation matching)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop merged detections whose DETECTION confidence (1 - P(EOS)) is below "
                         "this; note it is a heuristic score, not a calibrated probability")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--csv", default=None, help="output CSV path (default: <wav name>_detections.csv)")
    ap.add_argument("--xlsx", action="store_true", help="also save an Excel file")
    ap.add_argument("--plot", action="store_true",
                    help="save the detection overview plus one spectrogram crop per detection "
                         "(no whole-file spectrogram: pulses are invisible in one)")
    ap.add_argument("--plot-context", type=float, default=3.0,
                    help="seconds of context kept on each side of a pulse in its crop")
    ap.add_argument("--max-plots", type=int, default=20,
                    help="maximum number of detection crops to write, most confident first "
                         "(0 = no limit); the overview always shows every plotted detection")
    ap.add_argument("--plot-min-confidence", type=float, default=None,
                    help="detection-confidence threshold applied to the FIGURES only, so the "
                         "table and CSV can stay complete while the plots show only the "
                         "confident detections (default: same as --min-confidence)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = os.path.splitext(os.path.basename(args.wav))[0]
    csv_path = args.csv or f"{stem}_detections.csv"

    # the sampling rate comes from the checkpoint's stored config, so the recording is
    # preprocessed exactly as the training data was made (--fs overrides for old checkpoints)
    peek_cfg = torch.load(args.checkpoint, map_location="cpu").get("config", CONFIG)
    fs = args.fs if args.fs is not None else int(peek_cfg.get("fs", DEFAULT_FS))
    print(f"Sampling rate: {fs} Hz "
          + ("(from --fs)" if args.fs is not None else "(from checkpoint config)"))

    data, duration = read_wav(args.wav, fs)

    # peek one spectrogram to size the model input, then load the checkpoint
    probe = sps.stft(np.zeros(int(fs * peek_cfg["time_max"])), fs=fs,
                     nperseg=NPERSEG, noverlap=NOVERLAP)[2]
    modules, cfg = load_checkpoint(args.checkpoint, probe.shape, device)

    window_s = cfg["time_max"]
    hop_s = args.hop if args.hop is not None else window_s / 2

    windows = make_windows(data, fs, window_s, hop_s)
    X = windows_to_tensor(windows, fs)

    per_window = predict_windows(modules, X, cfg, device, args.batch_size)
    detections = to_global(per_window, windows, window_s)
    merged = merge_detections(detections, args.gate, args.freq_gate)
    if args.min_confidence > 0:
        merged = [p for p in merged if p["confidence_det"] >= args.min_confidence]

    print(f"\n{len(detections)} raw detections across windows -> {len(merged)} pulses after merging")
    print_table(merged)

    save_csv(merged, csv_path)
    if args.xlsx:
        save_xlsx(merged, f"{stem}_detections.xlsx")
    if args.plot:
        # NOTE: must be the resolved fs, not args.fs (which is None unless overridden)
        thr = args.min_confidence if args.plot_min_confidence is None else args.plot_min_confidence
        to_plot = [p for p in merged if p["confidence_det"] >= thr] if thr > 0 else merged
        if thr > 0:
            print(f"Plotting {len(to_plot)} of {len(merged)} detections "
                  f"(confidence_det >= {thr:g})")
        plot_overview(to_plot, len(data) / fs, f"{stem}_overview.png", fs)
        plot_detection_crops(data, fs, to_plot, stem,
                             context=args.plot_context, max_plots=args.max_plots)


if __name__ == "__main__":
    main()
