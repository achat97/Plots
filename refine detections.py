"""
Refines detected pulses by measuring them directly in the spectrogram.

The network localizes a pulse but estimates its frequencies from a heavily downsampled feature
map: six stride-2 blocks turn 1025 frequency rows into about 15, so one feature row spans well
over a kilohertz and the regressed band carries an error of a few hundred hertz. The spectrogram
it was computed from still has ~24 Hz bins. Once the pulse's time span is known, that resolution
can be used directly.

For every detection this script crops the spectrogram to the pulse, tracks the ridge of peak
energy through it, and fits the two sweep laws used to generate the training data:

    linear (LFM)      f(tau) = a + b * tau
    hyperbolic (HFM)  1 / f(tau) = alpha + beta * tau        with tau = (t - t_start) / duration

The better fit gives the pulse type, the fitted endpoints give f1 and f2 with a standard error
from the fit itself, and the sign of the slope gives the sweep direction - which the network gets
wrong whenever the sweep is narrower than its own frequency error, since the direction is then the
sign of a difference smaller than the noise on either endpoint. A slope that is not significantly
different from zero is reported as CW.

This is the same two stage arrangement used by established passive-acoustic software, where an
energy detector finds tonal sounds and a separate contour stage measures them; here the learned
detector replaces the energy detector.

Refinement is only ever a measurement: it does not add or remove detections. A pulse whose ridge
is too short or too noisy to fit keeps the network's own values and is flagged, so nothing is
silently overwritten.

Run:
    python refine_detections.py --wav recording.wav --detections recording_detections.csv
    python refine_detections.py --wav rec.wav --detections rec_detections.csv --plot --max-plots 10
"""

import argparse
import csv
import os

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

from data_config import FS as DEFAULT_FS, NPERSEG, NOVERLAP
from detect_pulses import read_wav, db
from plot_style import (FIG, FIG_WIDE, TYPE_COLORS, REFERENCE, type_name, hms,
                        hms_axis, finish, save)


def track_ridge(seg, fs, f_lo, f_hi, snr_db=8.0, rel_db=20.0, max_step_hz=None):

    """
    Tracks the ridge of peak energy through a cropped signal: for every time bin of the
    spectrogram it takes the strongest frequency bin inside the search band, refines it to
    sub-bin precision by fitting a parabola through the peak and its two neighbours in the log
    magnitude, and keeps the point only if the peak stands snr_db above the median level of that
    time bin (which removes the tapered pulse edges and any dropouts). When max_step_hz is given,
    a point is also required to lie within that distance of the previous accepted point, so a
    passing transient cannot pull the track off the pulse.

    ----------

    Parameters:
        seg (ndarray) - 1-D cropped signal containing the pulse.
        fs (int) - sampling frequency in Hz.
        f_lo (float) - lower edge of the search band in Hz.
        f_hi (float) - upper edge of the search band in Hz.
        snr_db (float) - default 8.0. Required excess of the peak over the median level of its
                         time bin, in dB.
        rel_db (float) - default 20.0. Points more than this far below the strongest column are
                         dropped: the crop includes context on either side of the pulse, where
                         the loudest bin is arbitrary noise.
        max_step_hz (float or None) - default None. Half-width, per point, of the band kept
                                      around the robust centre of the track, in Hz.

    Returns:
        t (ndarray) - times of the accepted points, in seconds from the start of the crop.
        f (ndarray) - frequencies of the accepted points, in Hz.
        level (ndarray) - peak level of each accepted point, in dB.
    """

    f_axis, t_axis, zxx = sps.stft(seg.astype(np.float64), fs=fs,
                                   nperseg=NPERSEG, noverlap=NOVERLAP)
    S = db(zxx)
    del zxx

    band = (f_axis >= f_lo) & (f_axis <= f_hi)
    if band.sum() < 3:
        return np.array([]), np.array([]), np.array([])
    f_band, S_band = f_axis[band], S[band]
    df = float(f_axis[1] - f_axis[0])

    times, freqs, levels = [], [], []
    for m in range(S_band.shape[1]):
        column = S_band[:, m]
        k = int(np.argmax(column))
        if column[k] - float(np.median(S[:, m])) < snr_db:
            continue

        # Parabolic interpolation of the peak; the correction is meaningful only for an interior
        # bin and can never exceed half a bin.
        offset = 0.0
        if 0 < k < len(column) - 1:
            y0, y1, y2 = column[k - 1], column[k], column[k + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                offset = 0.5 * (y0 - y2) / denom
                if abs(offset) > 0.5:
                    offset = 0.0

        times.append(float(t_axis[m]))
        freqs.append(float(f_band[k] + offset * df))
        levels.append(float(column[k]))

    if not times:
        return np.array([]), np.array([]), np.array([])
    t, f, level = np.array(times), np.array(freqs), np.array(levels)

    # A column whose peak is far below the strongest column is noise, not the pulse: the crop
    # deliberately includes context on both sides, and there the loudest bin is arbitrary.
    keep = level >= level.max() - rel_db
    t, f, level = t[keep], f[keep], level[keep]
    if len(t) == 0:
        return t, f, level

    # Constrain to a robust anchor (the median of the surviving points) rather than chaining
    # step to step, so one bad point cannot drag the track away.
    if max_step_hz is not None:
        anchor = float(np.median(f))
        span = max_step_hz * max(len(t), 1) * 0.5
        keep = np.abs(f - anchor) <= max(span, max_step_hz)
        t, f, level = t[keep], f[keep], level[keep]

    return t, f, level


def fit_sweeps(t, f, t0, t1):

    """
    Fits the linear and hyperbolic sweep laws to a tracked ridge and returns both fits, each with
    its endpoint values, slope, residual and standard errors. Time is normalized to the pulse
    span, so the fitted endpoints are the frequencies at the pulse start and stop.

    ----------

    Parameters:
        t (ndarray) - ridge times in seconds from the start of the crop.
        f (ndarray) - ridge frequencies in Hz.
        t0 (float) - pulse start, in seconds from the start of the crop.
        t1 (float) - pulse stop, in seconds from the start of the crop.

    Returns:
        fits (dict) - 'linear' and 'hyperbolic' entries, each a dict with keys f1, f2, slope,
                      slope_se, rss, rmse, f1_se, f2_se; None when the fit is not possible.
    """

    span = max(t1 - t0, 1e-9)
    tau = (t - t0) / span
    n = len(tau)
    fits = {"linear": None, "hyperbolic": None}
    if n < 3:
        return fits

    X = np.column_stack([np.ones(n), tau])
    XtX_inv = np.linalg.pinv(X.T @ X)

    def endpoints(beta, cov, invert):

        """Endpoint values and their standard errors, propagated through the fit covariance."""

        v0, v1 = np.array([1.0, 0.0]), np.array([1.0, 1.0])
        y0, y1 = float(beta[0]), float(beta[0] + beta[1])
        se0 = float(np.sqrt(max(v0 @ cov @ v0, 0.0)))
        se1 = float(np.sqrt(max(v1 @ cov @ v1, 0.0)))
        if not invert:
            return y0, y1, se0, se1
        if abs(y0) < 1e-12 or abs(y1) < 1e-12:
            return np.nan, np.nan, np.nan, np.nan
        # delta method: d(1/y)/dy = -1/y^2
        return 1.0 / y0, 1.0 / y1, se0 / y0**2, se1 / y1**2

    # linear: f = a + b * tau
    beta = XtX_inv @ X.T @ f
    resid = f - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / max(n - 2, 1)
    cov = sigma2 * XtX_inv
    f1, f2, se1, se2 = endpoints(beta, cov, invert=False)
    fits["linear"] = {"f1": f1, "f2": f2, "slope": float(beta[1]) / span,
                      "slope_se": float(np.sqrt(max(cov[1, 1], 0.0))) / span,
                      "rss": rss, "rmse": float(np.sqrt(rss / n)),
                      "f1_se": se1, "f2_se": se2}

    # hyperbolic: 1 / f = alpha + beta * tau
    if np.all(f > 1.0):
        g = 1.0 / f
        beta_h = XtX_inv @ X.T @ g
        resid_h = g - X @ beta_h
        sigma2_h = float(resid_h @ resid_h) / max(n - 2, 1)
        cov_h = sigma2_h * XtX_inv
        h1, h2, seh1, seh2 = endpoints(beta_h, cov_h, invert=True)
        if np.isfinite(h1) and np.isfinite(h2):
            model = 1.0 / (X @ beta_h)
            r = f - model
            rss_h = float(r @ r)
            fits["hyperbolic"] = {"f1": h1, "f2": h2, "slope": (h2 - h1) / span,
                                  "slope_se": abs(seh2 - seh1) / span,
                                  "rss": rss_h, "rmse": float(np.sqrt(rss_h / n)),
                                  "f1_se": seh1, "f2_se": seh2}
    return fits


def classify(fits, n_points, z=2.0):

    """
    Chooses the pulse type from the two fits. A slope that is not significantly different from
    zero means a constant tone, so the pulse is CW regardless of which curve fits better;
    otherwise the law with the smaller residual wins. The two laws are nearly identical over a
    narrow fractional bandwidth, so the margin between them is returned and should be checked
    before the LFM/HFM distinction is quoted.

    ----------

    Parameters:
        fits (dict) - output of fit_sweeps.
        n_points (int) - number of ridge points the fits are based on.
        z (float) - default 2.0. Significance factor for calling the slope non-zero.

    Returns:
        kind (str) - 'cw', 'lfm', 'hfm', or '' when no fit was possible.
        best (dict) - the winning fit, or None.
        margin (float) - relative residual gap between the two laws; 0 when only one fit exists.
    """

    lin, hyp = fits["linear"], fits["hyperbolic"]
    if lin is None:
        return "", None, 0.0

    if abs(lin["slope"]) < z * lin["slope_se"] or n_points < 3:
        return "cw", lin, 0.0

    if hyp is None:
        return "lfm", lin, 0.0

    total = lin["rss"] + hyp["rss"]
    margin = abs(lin["rss"] - hyp["rss"]) / total if total > 0 else 0.0
    return ("lfm", lin, margin) if lin["rss"] <= hyp["rss"] else ("hfm", hyp, margin)


def refine_one(data, fs, det, context=0.5, pad_hz=1500.0, snr_db=8.0, rel_db=20.0,
               min_points=8, max_rmse_bins=3.0):

    """
    Refines a single detection. The search band is taken from the network's estimate, widened by
    pad_hz and clamped to the physical range - the regression heads are unbounded, so an estimate
    can fall outside it, and a wider search is the right response to an implausible one.

    ----------

    Parameters:
        data (ndarray) - the full resampled signal.
        fs (int) - sampling frequency in Hz.
        det (dict) - one detection with keys t_start, t_stop, f1, f2.
        context (float) - default 0.5. Seconds of signal kept on each side of the pulse.
        pad_hz (float) - default 1500.0. Width added to each side of the predicted band.
        snr_db (float) - default 8.0. Ridge threshold above the median level of a time bin.
        rel_db (float) - default 20.0. Level range kept below the strongest column.
        min_points (int) - default 8. Fewest ridge points accepted for a refinement.
        max_rmse_bins (float) - default 3.0. Largest fit residual accepted, in frequency bins.

    Returns:
        out (dict) - refined values and diagnostics; 'ok' is False when the network's values
                     should be kept, with 'reason' explaining why.
    """

    df = fs / NPERSEG
    t0 = max(0.0, det["t_start"] - context)
    t1 = min(len(data) / fs, det["t_stop"] + context)
    seg = data[int(t0 * fs):int(t1 * fs)]
    if seg.size < NPERSEG:
        return {"ok": False, "reason": "crop shorter than one FFT window"}

    lo = max(0.0, min(det["f1"], det["f2"]) - pad_hz)
    hi = min(fs / 2, max(det["f1"], det["f2"]) + pad_hz)
    if hi - lo < 4 * df:
        lo, hi = max(0.0, lo - pad_hz), min(fs / 2, hi + pad_hz)

    duration = max(det["t_stop"] - det["t_start"], 1e-3)
    # A ridge may move by at most the sweep rate times the hop between STFT columns, with a
    # generous factor for noise; the floor keeps short pulses from being over-constrained.
    hop_s = (NPERSEG - NOVERLAP) / fs
    max_step = max(8 * df, 3.0 * (hi - lo) / duration * hop_s)   # per-point band half-width
    t, f, level = track_ridge(seg, fs, lo, hi, snr_db=snr_db, rel_db=rel_db,
                              max_step_hz=max_step)
    if len(t) < min_points:
        return {"ok": False, "reason": f"only {len(t)} ridge points", "n_points": len(t)}

    # restrict to the pulse itself, keeping the context out of the fit
    inside = (t >= det["t_start"] - t0) & (t <= det["t_stop"] - t0)
    if inside.sum() >= min_points:
        t, f, level = t[inside], f[inside], level[inside]
    fits = fit_sweeps(t, f, det["t_start"] - t0, det["t_stop"] - t0)
    kind, best, margin = classify(fits, len(t))
    if best is None:
        return {"ok": False, "reason": "fit failed", "n_points": len(t)}
    if best["rmse"] > max_rmse_bins * df:
        return {"ok": False, "reason": f"ridge scatter {best['rmse']:.0f} Hz too large",
                "n_points": len(t)}

    direction = "flat" if kind == "cw" else ("up" if best["slope"] > 0 else "down")
    return {"ok": True, "type": kind, "f1": best["f1"], "f2": best["f2"],
            "f1_se": best["f1_se"], "f2_se": best["f2_se"],
            "bandwidth": best["f2"] - best["f1"], "slope": best["slope"],
            "slope_se": best["slope_se"], "direction": direction,
            "rmse": best["rmse"], "margin": margin, "n_points": len(t),
            "t": t + t0, "f": f, "crop": (t0, t1, lo, hi), "fit": best, "kind": kind}


def plot_overview(results, duration, path, fs):

    """
    Draws every refined pulse over the whole recording, in the same layout as the detection
    overview: the upper panel shows the frequency band of each pulse, the lower panel the
    measurement uncertainty. The network's own band is drawn behind each refined one in grey, so
    the correction is visible at a glance, and pulses that could not be refined appear in grey
    alone. Detections whose direction changed are marked.

    ----------

    Parameters:
        results (list) - (detection, refinement) pairs, refinement being the output of refine_one.
        duration (float) - length of the recording in seconds.
        path (str) - output file path.
        fs (int) - sampling frequency in Hz, used to set the frequency axis to Nyquist.

    Returns:
        None
    """

    fig, (ax_f, ax_e) = plt.subplots(2, 1, figsize=FIG_WIDE, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    seen, n_flip, n_kept = set(), 0, 0
    for det, res in results:
        mid = 0.5 * (det["t_start"] + det["t_stop"])
        n_lo, n_hi = sorted((det["f1"], det["f2"]))
        ax_f.plot([mid, mid], [max(n_lo, 0), min(n_hi, fs / 2)], color="0.75", lw=3.5,
                  solid_capstyle="round",
                  label="Network estimate" if "net" not in seen else None)
        seen.add("net")

        if not (res and res["ok"]):
            n_kept += 1
            continue

        lo, hi = sorted((res["f1"], res["f2"]))
        colour = TYPE_COLORS.get(res["type"], REFERENCE)
        label = type_name(res["type"]) if res["type"] not in seen else None
        seen.add(res["type"])
        ax_f.plot([mid, mid], [lo, hi], color=colour, lw=1.8, solid_capstyle="round",
                  label=label)
        # an arrow head at the stop frequency shows the sweep direction
        if res["direction"] in ("up", "down"):
            ax_f.plot([mid], [res["f2"]], marker="^" if res["direction"] == "up" else "v",
                      ms=5, color=colour)
        else:
            ax_f.plot([mid], [res["f1"]], marker="o", ms=4, color=colour)

        net_dir = ("up" if det["f2"] > det["f1"] else
                   "down" if det["f2"] < det["f1"] else "flat")
        if res["direction"] in ("up", "down") and net_dir != res["direction"]:
            n_flip += 1
            ax_f.plot([mid], [hi], marker="x", ms=7, mew=1.6, color="black",
                      label="Direction corrected" if "flip" not in seen else None)
            seen.add("flip")

        err = max(res["f1_se"], res["f2_se"])
        ax_e.plot([mid, mid], [0.0, err], color=colour, lw=1.4)
        ax_e.plot([mid], [err], marker="o", ms=3, color=colour)

    ax_f.set_xlim(0, max(duration, 1e-3))
    ax_f.set_ylim(0, fs / 2)
    n_ref = sum(1 for _d, r in results if r and r["ok"])
    finish(ax_f, None, "Frequency [Hz]",
           f"{n_ref} of {len(results)} detections refined over {hms(duration, 0)} "
           f"({n_flip} with the direction corrected; vertical line: frequency band of one pulse)",
           legend=bool(seen))
    ax_e.set_ylim(bottom=0)
    hms_axis(ax_e)
    finish(ax_e, "Time [h:mm:ss]", "Fit uncertainty [Hz]", None)
    save(fig, path)


def plot_refinement(data, fs, det, res, path):

    """
    Draws the cropped spectrogram of one refined pulse with its tracked ridge and fitted sweep on
    top, so a refinement can be checked by eye.

    ----------

    Parameters:
        data (ndarray) - the full resampled signal.
        fs (int) - sampling frequency in Hz.
        det (dict) - the original detection.
        res (dict) - the result of refine_one.
        path (str) - output file path.

    Returns:
        None
    """

    t0, t1, lo, hi = res["crop"]
    seg = data[int(t0 * fs):int(t1 * fs)]
    f_axis, t_axis, zxx = sps.stft(seg.astype(np.float64), fs=fs,
                                   nperseg=NPERSEG, noverlap=NOVERLAP)
    S = db(zxx)
    del zxx

    fig, ax = plt.subplots(figsize=FIG)
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="plasma",
                   extent=[t0, t0 + float(t_axis[-1]), float(f_axis[0]), float(f_axis[-1])])
    fig.colorbar(im, ax=ax, label="Level [dB]")

    colour = TYPE_COLORS.get(res["type"], "white")
    ax.plot(res["t"], res["f"], ".", ms=3, color="white", label="Tracked ridge")
    tau = np.linspace(0, 1, 100)
    curve = (res["fit"]["f1"] + (res["fit"]["f2"] - res["fit"]["f1"]) * tau
             if res["kind"] != "hfm" else
             1.0 / (1.0 / res["fit"]["f1"]
                    + (1.0 / res["fit"]["f2"] - 1.0 / res["fit"]["f1"]) * tau))
    ax.plot(det["t_start"] + tau * (det["t_stop"] - det["t_start"]), curve,
            color=colour, lw=1.6, label=f"{type_name(res['type'])} fit")
    ax.axvline(det["t_start"], color=REFERENCE, ls=":", lw=1.2, label="Detected span")
    ax.axvline(det["t_stop"], color=REFERENCE, ls=":", lw=1.2)
    ax.set_ylim(max(0, lo - 500), min(fs / 2, hi + 500))
    finish(ax, "Time [s]", "Frequency [Hz]",
           f"{hms(det['t_start'])}: {type_name(res['type'])}, "
           f"{res['f1']:.0f} to {res['f2']:.0f} Hz "
           f"(network: {det['f1']:.0f} to {det['f2']:.0f} Hz)", legend=True)
    save(fig, path, quiet=True)


def main():

    """
    Parses arguments, refines every detection in a detections CSV, prints a comparison against the
    network's own values, and writes the refined CSV.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="the recording the detections came from")
    ap.add_argument("--detections", required=True, help="CSV written by detect_pulses.py")
    ap.add_argument("--out", default=None, help="output CSV (default: <detections>_refined.csv)")
    ap.add_argument("--fs", type=int, default=DEFAULT_FS,
                    help="sampling frequency the detections were produced at")
    ap.add_argument("--context", type=float, default=0.5,
                    help="seconds of signal kept on each side of a pulse")
    ap.add_argument("--pad-hz", type=float, default=1500.0,
                    help="width added to each side of the predicted band when searching")
    ap.add_argument("--snr-db", type=float, default=8.0,
                    help="ridge threshold above the median level of a time bin")
    ap.add_argument("--rel-db", type=float, default=20.0,
                    help="level range kept below the strongest column of the crop")
    ap.add_argument("--min-points", type=int, default=8,
                    help="fewest ridge points accepted for a refinement")
    ap.add_argument("--max-rmse-bins", type=float, default=3.0,
                    help="largest accepted fit residual, in frequency bins")
    ap.add_argument("--min-margin", type=float, default=0.1,
                    help="smallest residual gap between the two sweep laws for the LFM/HFM "
                         "distinction to be trusted; below this the type is reported as swept")
    ap.add_argument("--plot", action="store_true",
                    help="save an overview of all refinements plus a figure per refined pulse")
    ap.add_argument("--max-plots", type=int, default=20)
    ap.add_argument("--outdir", default="refined")
    args = ap.parse_args()

    data, duration = read_wav(args.wav, args.fs)
    with open(args.detections, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{args.wav}: {duration:.1f} s | {len(rows)} detections to refine")

    if args.plot:
        os.makedirs(args.outdir, exist_ok=True)

    out_rows, results, n_ok, n_plots = [], [], 0, 0
    print(f"\n  {'#':>3s} {'net type':>8s} {'new type':>8s} {'net f1':>8s} {'new f1':>8s} "
          f"{'net f2':>8s} {'new f2':>8s} {'+/- Hz':>7s} {'dir':>5s} {'pts':>4s}  note")
    for i, row in enumerate(rows):
        det = {"t_start": float(row["t_start_s"]), "t_stop": float(row["t_stop_s"]),
               "f1": float(row["f_start_hz"]), "f2": float(row["f_stop_hz"])}
        res = refine_one(data, args.fs, det, context=args.context, pad_hz=args.pad_hz,
                         snr_db=args.snr_db, rel_db=args.rel_db, min_points=args.min_points,
                         max_rmse_bins=args.max_rmse_bins)

        results.append((det, res))
        out = dict(row)
        if res["ok"]:
            n_ok += 1
            reported = res["type"]
            note = ""
            if res["type"] in ("lfm", "hfm") and res["margin"] < args.min_margin:
                reported = "swept"
                note = "curvature not resolved"
            out.update({"refined": 1, "refined_type": reported,
                        "refined_f1_hz": f"{res['f1']:.1f}", "refined_f2_hz": f"{res['f2']:.1f}",
                        "refined_f1_se_hz": f"{res['f1_se']:.1f}",
                        "refined_f2_se_hz": f"{res['f2_se']:.1f}",
                        "refined_bandwidth_hz": f"{res['bandwidth']:.1f}",
                        "refined_direction": res["direction"],
                        "refined_slope_hz_per_s": f"{res['slope']:.1f}",
                        "refined_rmse_hz": f"{res['rmse']:.1f}",
                        "refined_n_points": res["n_points"],
                        "refined_note": note})
            print(f"  {i:>3d} {row['type']:>8s} {reported:>8s} {det['f1']:>8.0f} "
                  f"{res['f1']:>8.0f} {det['f2']:>8.0f} {res['f2']:>8.0f} "
                  f"{max(res['f1_se'], res['f2_se']):>7.0f} {res['direction']:>5s} "
                  f"{res['n_points']:>4d}  {note}")
            if args.plot and n_plots < args.max_plots:
                plot_refinement(data, args.fs, det, res,
                                os.path.join(args.outdir, f"refined_{i:03d}.png"))
                n_plots += 1
        else:
            out.update({"refined": 0, "refined_type": "", "refined_f1_hz": "",
                        "refined_f2_hz": "", "refined_f1_se_hz": "", "refined_f2_se_hz": "",
                        "refined_bandwidth_hz": "", "refined_direction": "",
                        "refined_slope_hz_per_s": "", "refined_rmse_hz": "",
                        "refined_n_points": res.get("n_points", 0),
                        "refined_note": res["reason"]})
            print(f"  {i:>3d} {row['type']:>8s} {'-':>8s} {det['f1']:>8.0f} {'-':>8s} "
                  f"{det['f2']:>8.0f} {'-':>8s} {'-':>7s} {'-':>5s} "
                  f"{res.get('n_points', 0):>4d}  not refined: {res['reason']}")
        out_rows.append(out)

    path = args.out or os.path.splitext(args.detections)[0] + "_refined.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nRefined {n_ok} of {len(rows)} detections; the rest keep the network's values "
          f"and carry a reason in refined_note.")
    print(f"Saved {path}")
    if args.plot:
        plot_overview(results, duration, os.path.join(args.outdir, "refined_overview.png"), args.fs)
        print(f"Saved {n_plots} pulse figures to '{args.outdir}/'")


if __name__ == "__main__":
    main()
