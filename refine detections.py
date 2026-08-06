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

The pulse's start and stop times are refined as well, since the tracked ridge shows where the
pulse actually begins and ends; the fitted sweep is only ever evaluated between those points. The
ridge is cut where a reverberation echo takes over, so that the stop time and the fitted sweep
describe the direct arrival rather than the last thing to arrive in the same band.

The two ridge thresholds are not set by hand: several values are tried per pulse and the one whose
sweep fit has the lowest residual is kept, so a faint pulse and a loud one next to an interferer
are each tracked at the setting that suits them. A CW pulse is reported as one centre frequency,
since its two fitted endpoints estimate the same tone - unlike the network, whose four regression
heads are independent and can place them hundreds of hertz apart.

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
from table_io import save_table_xlsx
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


def classify(fits, n_points, z=2.0, min_bandwidth=50.0):

    """
    Chooses the pulse type from the two fits. A pulse whose total frequency change is below
    min_bandwidth is a tone regardless of what the fit says, and a slope that is not significantly
    different from zero means a constant tone as well, so either makes the pulse CW;
    otherwise the law with the smaller residual wins. The two laws are nearly identical over a
    narrow fractional bandwidth, so the margin between them is returned and should be checked
    before the LFM/HFM distinction is quoted.

    ----------

    Parameters:
        fits (dict) - output of fit_sweeps.
        n_points (int) - number of ridge points the fits are based on.
        z (float) - default 2.0. Significance factor for calling the slope non-zero.
        min_bandwidth (float) - default 50.0. Frequency change in Hz below which the pulse is a
                                tone, whatever the fit says. Roughly two spectrogram bins at the
                                pipeline's settings.

    Returns:
        kind (str) - 'cw', 'lfm', 'hfm', or '' when no fit was possible.
        best (dict) - the winning fit, or None.
        margin (float) - relative residual gap between the two laws; 0 when only one fit exists.
    """

    lin, hyp = fits["linear"], fits["hyperbolic"]
    if lin is None:
        return "", None, 0.0

    # A sweep narrower than min_bandwidth is a tone. The statistical test alone is not enough here:
    # the slope is fitted from dozens of ridge points, so its standard error can fall to a fraction
    # of a hertz, and a drift of twenty or thirty hertz across the pulse then counts as
    # "significantly non-zero" and is reported as a sweep. That drift is below the width of a
    # single spectrogram bin - it is not a sweep anyone can measure, and calling it LFM or HFM
    # gives the reader a sweep direction and a bandwidth that mean nothing. The physical floor is
    # applied first, and the significance test only afterwards.
    bandwidth = abs(lin["f2"] - lin["f1"])
    if bandwidth < min_bandwidth or n_points < 3:
        return "cw", lin, 0.0

    if abs(lin["slope"]) < z * lin["slope_se"]:
        return "cw", lin, 0.0

    if hyp is None:
        return "lfm", lin, 0.0

    total = lin["rss"] + hyp["rss"]
    margin = abs(lin["rss"] - hyp["rss"]) / total if total > 0 else 0.0
    return ("lfm", lin, margin) if lin["rss"] <= hyp["rss"] else ("hfm", hyp, margin)


def truncate_at_echo(t, f, level, df, min_points=8, reversal_frac=0.25, gap_factor=4.0,
                     min_sweep_bins=8.0):

    """
    Cuts the tracked ridge where a reverberation echo takes over from the direct arrival.

    An echo is the same pulse arriving again by a longer path, so it puts energy in the same band
    and the tracker follows it straight on from the direct arrival. The ridge then runs past the
    end of the pulse and everything read from it is spoiled: the stop time belongs to the echo, and
    for a swept pulse the echo RESTARTS the sweep, so the fitted slope - and with it the endpoint
    frequencies - is pulled towards a line through two sweeps instead of one.

    Two tests are applied, because each covers what the other cannot.

    A gap in time. An echo arrives after the direct pulse has ended, so the ridge usually stops and
    resumes, leaving empty columns between them. This test does not care what the pulse is doing in
    frequency, so it is the one that works for a CW tone, whose echo sits at the same frequency and
    is invisible to any test based on frequency alone.

    A reversal in the sweep. A swept pulse moves its frequency one way throughout; an echo begins
    the sweep again, which appears as a large step back against that direction. Small reversals are
    ordinary noise, so a cut is made only when the step back exceeds a fraction of the frequency
    span the ridge has covered.

    An earlier version also compared the later points with a straight-line fit to the early ones.
    That was removed: a hyperbolic sweep departs from a straight line by construction, so the test
    cut clean HFM pulses that had no echo at all. Only tests that cannot be confused by the pulse's
    own shape are used here.

    ----------

    Parameters:
        t (ndarray) - ridge times, ascending.
        f (ndarray) - ridge frequencies.
        level (ndarray) - ridge peak levels in dB.
        df (float) - frequency bin width in Hz.
        min_points (int) - default 8. Never cut below this many points.
        reversal_frac (float) - default 0.25. Step back against the sweep direction, as a fraction
                                of the frequency span covered, that counts as an echo restart.
        gap_factor (float) - default 4.0. Multiple of the usual spacing between ridge points that
                             counts as the pulse having stopped.
        min_sweep_bins (float) - default 8.0. Frequency span, in bins, below which the pulse is
                                 treated as unswept and the reversal test is not applied.

    Returns:
        t, f, level (ndarray) - the ridge truncated at the echo.
        cut (int or None) - index the ridge was cut at, or None when nothing was cut.
    """

    n = len(t)
    if n < min_points + 2:
        return t, f, level, None

    cut = n

    # --- test 1: the ridge stops and resumes ---
    dt = np.diff(t)
    if len(dt) >= 3:
        step = float(np.median(dt))
        if step > 0:
            big = np.nonzero(dt > gap_factor * step)[0]
            big = big[big + 1 >= min_points]
            if len(big):
                cut = min(cut, int(big[0]) + 1)

    # --- test 2: the sweep turns back on itself ---
    head = max(min_points, n // 3)
    span = float(np.max(f) - np.min(f))
    # Only a genuinely swept pulse has a direction to reverse. A CW tone's span is its own jitter,
    # a few tens of hertz, and a threshold set as a fraction of that would be smaller than the
    # noise - so the test would cut every CW pulse at the first wobble. Requiring the span to
    # cover several frequency bins keeps the test to the pulses it was meant for.
    if head >= 2 and head < n and span > min_sweep_bins * df:
        slope = float(np.polyfit(t[:head], f[:head], 1)[0])
        if slope != 0.0:
            direction = 1.0 if slope > 0 else -1.0
            threshold = reversal_frac * span
            extreme = f[0]
            for i in range(1, n):
                extreme = max(extreme, f[i]) if direction > 0 else min(extreme, f[i])
                if direction * (extreme - f[i]) > threshold and i >= min_points:
                    cut = min(cut, i)
                    break

    if cut >= n or cut < min_points:
        return t, f, level, None
    return t[:cut], f[:cut], level[:cut], int(cut)


def _attempt(seg, fs, det, t0, lo, hi, snr_db, rel_db, max_step, min_points, df,
             truncate=True, min_bandwidth=50.0):

    """
    Tracks and fits one pulse at a single pair of ridge thresholds. Returns the fitted result with
    its residual, or None when the ridge is too short or no fit is possible. Used by refine_one to
    compare several threshold settings.

    ----------

    Parameters:
        seg (ndarray) - the cropped signal.
        fs (int) - sampling frequency in Hz.
        det (dict) - the detection being refined.
        t0 (float) - start of the crop, in seconds from the start of the recording.
        lo (float) - lower edge of the search band in Hz.
        hi (float) - upper edge of the search band in Hz.
        snr_db (float) - ridge threshold above the median level of a time bin.
        rel_db (float) - level range kept below the strongest column.
        max_step (float) - per-point band half-width in Hz.
        min_points (int) - fewest ridge points accepted.
        df (float) - frequency bin width in Hz.

    Returns:
        (dict or None) - the fitted result, with the thresholds it used.
    """

    t, f, level = track_ridge(seg, fs, lo, hi, snr_db=snr_db, rel_db=rel_db,
                              max_step_hz=max_step)
    if len(t) < min_points:
        return None

    # Keep only points inside the detected span, but do not force the fit to span it: the ridge
    # marks where the pulse actually is, and the network's span is itself an estimate.
    inside = (t >= det["t_start"] - t0) & (t <= det["t_stop"] - t0)
    if inside.sum() >= min_points:
        t, f, level = t[inside], f[inside], level[inside]

    # The fit is evaluated at the FIRST and LAST tracked point, never outside them. Evaluating it
    # at the network's span would extrapolate whenever that span is wider than the ridge, and the
    # extrapolation error grows with the sweep rate, so it damages exactly the fastest sweeps.
    # Cut the ridge where a reverberation echo takes over, so the stop time and the fitted sweep
    # describe the direct arrival only.
    raw_span = float(t[-1] - t[0])
    cut = None
    if truncate:
        t, f, level, cut = truncate_at_echo(t, f, level, df, min_points=min_points)

    ridge_start, ridge_stop = float(t[0]), float(t[-1])
    fits = fit_sweeps(t, f, ridge_start, ridge_stop)
    kind, best, margin = classify(fits, len(t), min_bandwidth=min_bandwidth)
    if best is None:
        return None
    return {"t": t, "f": f, "fit": best, "kind": kind, "margin": margin,
            "rmse": best["rmse"], "n_points": len(t), "snr_db": snr_db, "rel_db": rel_db,
            "t_start": ridge_start + t0, "t_stop": ridge_stop + t0,
            "echo_cut": cut is not None, "raw_span": raw_span}


def refine_one(data, fs, det, context=0.5, pad_hz=1500.0, snr_db=None, rel_db=None,
               min_points=8, max_rmse_bins=3.0, min_coverage=0.3, truncate_echo=True,
               min_bandwidth=50.0):

    """
    Refines a single detection. The search band is taken from the network's estimate, widened by
    pad_hz and clamped to the physical range - the regression heads are unbounded, so an estimate
    can fall outside it, and a wider search is the right response to an implausible one.

    The two ridge thresholds are chosen per pulse rather than set by hand. A pulse just above the
    noise needs a permissive threshold to yield any ridge at all, while a loud pulse next to an
    interfering tone needs a strict one; no single pair of values suits both, which is why tuning
    them manually is unrewarding. Several pairs are tried and the one whose sweep fit has the
    lowest residual per point is kept. The residual is an internal measure of how well the tracked
    points follow a sweep law, so no labels are needed to make the choice. Passing snr_db or
    rel_db explicitly disables the search for that threshold.

    The pulse's start and stop times are refined too: they are taken from the first and last tracked
    point, and the sweep is only ever evaluated between them. Reading the fit at the network's own
    span would extrapolate beyond the tracked points whenever that span is the wider of the two,
    which both distorts the reported endpoint frequencies and draws a sweep across parts of the
    figure where no pulse was measured.

    A pulse whose fitted frequency change is under a spectrogram bin or two is reported as CW
    rather than as a very narrow sweep, since a bandwidth smaller than the resolution carries no
    usable direction and no usable width.

    A CW pulse is a single tone, so its two fitted endpoints estimate the same quantity: they are
    replaced by their mean, and the standard error of that mean is reported for both. The network
    cannot do this - its four regression heads are independent, so a CW detection there can carry
    two endpoint estimates hundreds of hertz apart.

    ----------

    Parameters:
        data (ndarray) - the full resampled signal.
        fs (int) - sampling frequency in Hz.
        det (dict) - one detection with keys t_start, t_stop, f1, f2.
        context (float) - default 0.5. Seconds of signal kept on each side of the pulse.
        pad_hz (float) - default 1500.0. Width added to each side of the predicted band.
        snr_db (float or None) - default None (search 4, 6, 8, 12, 16). Ridge threshold above the
                                 median level of a time bin.
        rel_db (float or None) - default None (search 12, 20, 30). Level range kept below the
                                 strongest column.
        min_points (int) - default 8. Fewest ridge points accepted for a refinement.
        max_rmse_bins (float) - default 3.0. Largest fit residual accepted, in frequency bins.
        min_coverage (float) - default 0.3. Smallest fraction of the detected span the ridge must
                               cover; below this the tracker probably caught only a fragment.
        min_bandwidth (float) - default 50.0. Frequency change in Hz below which a pulse is
                                reported as CW rather than as a very narrow sweep.
        truncate_echo (bool) - default True. Cut the ridge where a reverberation echo takes over
                               (see truncate_at_echo). Turn off to measure the whole ridge,
                               including any echo.

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

    snr_grid = [snr_db] if snr_db is not None else [4.0, 6.0, 8.0, 12.0, 16.0]
    rel_grid = [rel_db] if rel_db is not None else [12.0, 20.0, 30.0]

    best_attempt, tried = None, 0
    for s in snr_grid:
        for r in rel_grid:
            tried += 1
            attempt = _attempt(seg, fs, det, t0, lo, hi, s, r, max_step, min_points, df,
                               truncate=truncate_echo, min_bandwidth=min_bandwidth)
            if attempt is None:
                continue
            # Compare on residual per point, then prefer the longer ridge on a tie: a fit over
            # more of the pulse is better supported at the same scatter.
            if (best_attempt is None
                    or (attempt["rmse"], -attempt["n_points"])
                    < (best_attempt["rmse"], -best_attempt["n_points"])):
                best_attempt = attempt

    if best_attempt is None:
        return {"ok": False, "reason": f"no ridge long enough at any threshold ({tried} tried)",
                "n_points": 0}
    if best_attempt["rmse"] > max_rmse_bins * df:
        return {"ok": False,
                "reason": f"ridge scatter {best_attempt['rmse']:.0f} Hz too large",
                "n_points": best_attempt["n_points"]}

    # A ridge covering only a fragment of the detected span usually means the tracker locked onto
    # part of the pulse, or onto something else entirely; the measurement is then not comparable
    # with the detection it claims to refine.
    # Coverage is judged on the ridge as tracked, before any echo was trimmed: trimming is a
    # correction, not evidence that the tracker caught only a fragment.
    ridge_span = best_attempt.get("raw_span", best_attempt["t_stop"] - best_attempt["t_start"])
    if ridge_span < min_coverage * duration:
        return {"ok": False,
                "reason": f"ridge covers {ridge_span / duration * 100:.0f}% of the detected span",
                "n_points": best_attempt["n_points"]}

    best = best_attempt["fit"]
    kind = best_attempt["kind"]
    f1, f2 = best["f1"], best["f2"]
    f1_se, f2_se = best["f1_se"], best["f2_se"]

    if kind == "cw":
        # One tone, two estimates of it: the mean is the better estimate, and its standard error
        # is smaller than either endpoint's.
        centre = 0.5 * (f1 + f2)
        centre_se = 0.5 * float(np.hypot(f1_se, f2_se))
        f1 = f2 = centre
        f1_se = f2_se = centre_se

    direction = "flat" if kind == "cw" else ("up" if best["slope"] > 0 else "down")
    return {"ok": True, "type": kind, "f1": f1, "f2": f2,
            "t_start": best_attempt["t_start"], "t_stop": best_attempt["t_stop"],
            "duration": best_attempt["t_stop"] - best_attempt["t_start"],
            "f1_se": f1_se, "f2_se": f2_se,
            "bandwidth": f2 - f1, "slope": best["slope"],
            "slope_se": best["slope_se"], "direction": direction,
            "rmse": best_attempt["rmse"], "margin": best_attempt["margin"],
            "echo_cut": best_attempt.get("echo_cut", False),
            "n_points": best_attempt["n_points"],
            "snr_db": best_attempt["snr_db"], "rel_db": best_attempt["rel_db"],
            "t": best_attempt["t"] + t0, "f": best_attempt["f"],
            "crop": (t0, t1, lo, hi), "fit": best, "kind": kind}


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
        if res and res["ok"]:
            mid = 0.5 * (res["t_start"] + res["t_stop"])
        else:
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
    # The curve is drawn only between the first and last tracked point, which is also where it
    # was fitted; drawing it across the network's span would show a sweep where nothing was
    # measured.
    tau = np.linspace(0, 1, 100)
    curve = (res["fit"]["f1"] + (res["fit"]["f2"] - res["fit"]["f1"]) * tau
             if res["kind"] != "hfm" else
             1.0 / (1.0 / res["fit"]["f1"]
                    + (1.0 / res["fit"]["f2"] - 1.0 / res["fit"]["f1"]) * tau))
    ax.plot(res["t_start"] + tau * (res["t_stop"] - res["t_start"]), curve,
            color=colour, lw=1.6, label=f"{type_name(res['type'])} fit")
    ax.axvline(det["t_start"], color=REFERENCE, ls=":", lw=1.2, label="Detected span")
    ax.axvline(det["t_stop"], color=REFERENCE, ls=":", lw=1.2)
    ax.axvline(res["t_start"], color=colour, ls="-", lw=1.0, alpha=0.7, label="Measured span")
    ax.axvline(res["t_stop"], color=colour, ls="-", lw=1.0, alpha=0.7)
    ax.set_ylim(max(0, lo - 500), min(fs / 2, hi + 500))
    finish(ax, "Time [s]", "Frequency [Hz]",
           f"{hms(res['t_start'])} to {hms(res['t_stop'])}: {type_name(res['type'])}, "
           f"{res['f1']:.0f} to {res['f2']:.0f} Hz\n"
           f"network: {hms(det['t_start'])} to {hms(det['t_stop'])}, "
           f"{det['f1']:.0f} to {det['f2']:.0f} Hz", legend=True)
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
    ap.add_argument("--snr-db", type=float, default=None,
                    help="fix the ridge threshold above the median level of a time bin "
                         "(default: choose per pulse from 4, 6, 8, 12, 16 dB by fit residual)")
    ap.add_argument("--rel-db", type=float, default=None,
                    help="fix the level range kept below the strongest column "
                         "(default: choose per pulse from 12, 20, 30 dB by fit residual)")
    ap.add_argument("--min-points", type=int, default=8,
                    help="fewest ridge points accepted for a refinement")
    ap.add_argument("--max-rmse-bins", type=float, default=3.0,
                    help="largest accepted fit residual, in frequency bins")
    ap.add_argument("--min-coverage", type=float, default=0.3,
                    help="smallest fraction of the detected span the tracked ridge must cover")
    ap.add_argument("--min-margin", type=float, default=0.1,
                    help="smallest residual gap between the two sweep laws for the LFM/HFM "
                         "distinction to be trusted; below this the type is reported as swept")
    ap.add_argument("--min-bandwidth", type=float, default=50.0,
                    help="frequency change in Hz below which a pulse is reported as CW rather "
                         "than as a very narrow sweep; roughly two spectrogram bins")
    ap.add_argument("--keep-echo", action="store_true",
                    help="do not cut the ridge where a reverberation echo takes over; the stop "
                         "time then describes the last arrival rather than the direct pulse")
    ap.add_argument("--xlsx", action="store_true",
                    help="also write the table as an Excel file beside the CSV")
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
          f"{'net f2':>8s} {'new f2':>8s} {'+/- Hz':>7s} {'dir':>5s} {'dur s':>7s} {'pts':>4s}  note")
    for i, row in enumerate(rows):
        det = {"t_start": float(row["t_start_s"]), "t_stop": float(row["t_stop_s"]),
               "f1": float(row["f_start_hz"]), "f2": float(row["f_stop_hz"])}
        res = refine_one(data, args.fs, det, context=args.context, pad_hz=args.pad_hz,
                         snr_db=args.snr_db, rel_db=args.rel_db, min_points=args.min_points,
                         max_rmse_bins=args.max_rmse_bins, min_coverage=args.min_coverage,
                         truncate_echo=not args.keep_echo,
                         min_bandwidth=args.min_bandwidth)

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
                        "refined_t_start_s": f"{res['t_start']:.3f}",
                        "refined_t_stop_s": f"{res['t_stop']:.3f}",
                        "refined_t_start_hms": hms(res["t_start"]),
                        "refined_t_stop_hms": hms(res["t_stop"]),
                        "refined_duration_s": f"{res['duration']:.3f}",
                        "refined_f1_hz": f"{res['f1']:.1f}", "refined_f2_hz": f"{res['f2']:.1f}",
                        "refined_f1_se_hz": f"{res['f1_se']:.1f}",
                        "refined_f2_se_hz": f"{res['f2_se']:.1f}",
                        "refined_bandwidth_hz": f"{res['bandwidth']:.1f}",
                        "refined_direction": res["direction"],
                        "refined_slope_hz_per_s": f"{res['slope']:.1f}",
                        "refined_rmse_hz": f"{res['rmse']:.1f}",
                        "refined_n_points": res["n_points"],
                        "refined_snr_db": res["snr_db"], "refined_rel_db": res["rel_db"],
                        "refined_echo_trimmed": int(res.get("echo_cut", False)),
                        "refined_note": note})
            if res["type"] == "cw" and not note:
                note = f"centre {res['f1']:.0f} Hz"
            if res.get("echo_cut"):
                note = (note + "; " if note else "") + "echo trimmed"
            print(f"  {i:>3d} {row['type']:>8s} {reported:>8s} {det['f1']:>8.0f} "
                  f"{res['f1']:>8.0f} {det['f2']:>8.0f} {res['f2']:>8.0f} "
                  f"{max(res['f1_se'], res['f2_se']):>7.1f} {res['direction']:>5s} "
                  f"{res['duration']:>7.2f} {res['n_points']:>4d}  {note}")
            if args.plot and n_plots < args.max_plots:
                plot_refinement(data, args.fs, det, res,
                                os.path.join(args.outdir, f"refined_{i:03d}.png"))
                n_plots += 1
        else:
            out.update({"refined": 0, "refined_type": "",
                        "refined_t_start_s": "", "refined_t_stop_s": "",
                        "refined_t_start_hms": "", "refined_t_stop_hms": "",
                        "refined_duration_s": "", "refined_f1_hz": "",
                        "refined_f2_hz": "", "refined_f1_se_hz": "", "refined_f2_se_hz": "",
                        "refined_bandwidth_hz": "", "refined_direction": "",
                        "refined_slope_hz_per_s": "", "refined_rmse_hz": "",
                        "refined_n_points": res.get("n_points", 0),
                        "refined_snr_db": "", "refined_rel_db": "",
                        "refined_echo_trimmed": "",
                        "refined_note": res["reason"]})
            print(f"  {i:>3d} {row['type']:>8s} {'-':>8s} {det['f1']:>8.0f} {'-':>8s} "
                  f"{det['f2']:>8.0f} {'-':>8s} {'-':>7s} {'-':>5s} {'-':>7s} "
                  f"{res.get('n_points', 0):>4d}  not refined: {res['reason']}")
        out_rows.append(out)

    path = args.out or os.path.splitext(args.detections)[0] + "_refined.csv"
    header = list(out_rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(out_rows)
    if args.xlsx:
        save_table_xlsx(header, [[r[k] for k in header] for r in out_rows],
                        os.path.splitext(path)[0] + ".xlsx", sheet="refined")

    print(f"\nRefined {n_ok} of {len(rows)} detections; the rest keep the network's values "
          f"and carry a reason in refined_note.")
    print(f"Saved {path}")
    if args.plot:
        plot_overview(results, duration, os.path.join(args.outdir, "refined_overview.png"), args.fs)
        print(f"Saved {n_plots} pulse figures to '{args.outdir}/'")


if __name__ == "__main__":
    main()
