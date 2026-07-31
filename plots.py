"""
Evaluation and training figures.

Evaluation set (from evaluate.py --plots, written to eval_plots/):
  count_confusion    - predicted vs true number of pulses per example
  type_confusion     - predicted vs true pulse type, on matched pulses
  detection          - true positives, false alarms and misses, with precision/recall/F1
  error_histograms   - signed error of each regressed quantity
  pred_vs_true       - predicted against true value of each regressed quantity
  by_n_pulses        - metrics against the number of pulses in the example

Training set (from plots.py run directly, or at the end of training):
  loss_curves        - teacher-forced training and validation loss per epoch
  ar_vs_tf           - AR score against teacher-forced loss per epoch
  components         - the individual terms of the AR score per epoch

Run directly to draw the training figures from a finished run:
    python plots.py
    python plots.py --components val_components.csv --outdir figures --prefix train
"""

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from metrics_core import match_pulses, evaluate, gate
from plot_style import (FIG, FIG_GRID, FONT, PRIMARY, ACCENT, REFERENCE,
                        LABELS, UNITS, axis_label, type_name, finish, save)


TYPES = ["cw", "lfm", "hfm"]
COUNTS = [0, 1, 2, 3, 4]
REG_KEYS = ["t_start", "t_stop", "f1", "f2"]


def collect(all_pred, all_truth):

    """
    Collects everything the evaluation figures need: the pulse-count confusion, the type confusion
    on matched pulses, and the predicted/true value pairs of each regressed quantity.

    ----------

    Parameters:
        all_pred (list) - predicted pulses per example, each a dict with keys type, t_start,
                          t_stop, f1, f2.
        all_truth (list) - true pulses per example, in the same form.

    Returns:
        count_cm (ndarray) - 5x5 array, rows true count, columns predicted count.
        type_cm (ndarray) - 3x3 array, rows true type, columns predicted type.
        reg (dict) - per quantity, {'pred': array, 'true': array} over matched pulses.
    """

    count_cm = np.zeros((len(COUNTS), len(COUNTS)), dtype=int)
    type_cm = np.zeros((len(TYPES), len(TYPES)), dtype=int)
    reg = {k: {"pred": [], "true": []} for k in REG_KEYS}
    t_index = {t: i for i, t in enumerate(TYPES)}

    for pred, truth in zip(all_pred, all_truth):
        count_cm[min(len(truth), COUNTS[-1]), min(len(pred), COUNTS[-1])] += 1
        for i, j in match_pulses(pred, truth)[0]:
            type_cm[t_index[truth[j]["type"]], t_index[pred[i]["type"]]] += 1
            for key in reg:
                reg[key]["pred"].append(pred[i][key])
                reg[key]["true"].append(truth[j][key])

    for key in reg:
        reg[key]["pred"] = np.array(reg[key]["pred"], float)
        reg[key]["true"] = np.array(reg[key]["true"], float)
    return count_cm, type_cm, reg


def plot_confusion(cm, labels, title, axis_name, path):

    """
    Draws a confusion matrix as a row-normalised heatmap, annotated with the count and the row
    percentage.

    ----------

    Parameters:
        cm (ndarray) - confusion matrix, rows true, columns predicted.
        labels (list) - tick labels for both axes.
        title (str) - figure title.
        axis_name (str) - what the axes count, e.g. 'pulses' or 'type'.
        path (str) - output file path.

    Returns:
        path (str) - the file written.
    """

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, float), where=row_sums > 0)
    annot = np.array([[f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%" if row_sums[i] else f"{cm[i, j]}"
                       for j in range(cm.shape[1])] for i in range(cm.shape[0])], dtype=object)

    side = 1.1 * len(labels) + 2.6
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    sns.heatmap(norm, ax=ax, cmap="Blues", vmin=0, vmax=1, annot=annot, fmt="",
                annot_kws={"fontsize": FONT["annot"]}, xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Fraction of true row", "fraction": 0.046, "pad": 0.04})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    finish(ax, f"Predicted {axis_name}", f"True {axis_name}", title)
    return save(fig, path)


def plot_detection(metrics, path):

    """
    Draws the detection outcome counts, with precision, recall and F1 in the title.

    ----------

    Parameters:
        metrics (dict) - output of metrics_core.evaluate.
        path (str) - output file path.

    Returns:
        path (str) - the file written.
    """

    d = metrics["detection"]
    names = ["True positives", "False alarms", "Misses"]
    values = [d["TP"], d["FP"], d["FN"]]

    fig, ax = plt.subplots(figsize=FIG)
    bars = ax.bar(names, values, color=[PRIMARY, ACCENT, REFERENCE], width=0.6)
    ax.bar_label(bars, fontsize=FONT["annot"], padding=2)
    ax.margins(y=0.12)
    finish(ax, None, "Pulses",
           f"Detection outcomes  |  precision {d['precision']:.3f}, "
           f"recall {d['recall']:.3f}, F1 {d['f1']:.3f}\n"
           f"a prediction matches a true pulse within {gate:.1f} s and overlapping bands")
    return save(fig, path)


def plot_error_histograms(reg, path):

    """
    Draws the signed error distribution of each regressed quantity, with zero and the mean marked.

    ----------

    Parameters:
        reg (dict) - regression pairs from collect.
        path (str) - output file path.

    Returns:
        path (str) - the file written.
    """

    fig, axes = plt.subplots(2, 2, figsize=FIG_GRID)
    for ax, key in zip(axes.ravel(), REG_KEYS):
        err = reg[key]["pred"] - reg[key]["true"]
        if err.size == 0:
            ax.set_axis_off()
            ax.set_title(f"{LABELS[key]} (no matched pulses)")
            continue
        sns.histplot(err, bins=40, ax=ax, color=PRIMARY, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=REFERENCE, lw=1.4, label="No error")
        ax.axvline(err.mean(), color=ACCENT, ls="--", lw=1.6,
                   label=f"Mean {err.mean():+.3g} {UNITS[key]}")
        finish(ax, f"Predicted - true [{UNITS[key]}]", "Pulses",
               f"{LABELS[key]}  (n = {err.size})", legend=True)
    fig.suptitle("Regression error on matched pulses", fontsize=FONT["title"])
    return save(fig, path)


def plot_pred_vs_true(reg, path):

    """
    Draws predicted against true values for each regressed quantity, with the identity line.

    ----------

    Parameters:
        reg (dict) - regression pairs from collect.
        path (str) - output file path.

    Returns:
        path (str) - the file written.
    """

    fig, axes = plt.subplots(2, 2, figsize=FIG_GRID)
    for ax, key in zip(axes.ravel(), REG_KEYS):
        p, t = reg[key]["pred"], reg[key]["true"]
        if p.size == 0:
            ax.set_axis_off()
            ax.set_title(f"{LABELS[key]} (no matched pulses)")
            continue
        ax.scatter(t, p, s=12, alpha=0.45, color=PRIMARY, edgecolors="none")
        lo, hi = float(min(p.min(), t.min())), float(max(p.max(), t.max()))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.4, color=REFERENCE, label="Perfect prediction")
        ax.set_aspect("equal", adjustable="datalim")
        finish(ax, f"True [{UNITS[key]}]", f"Predicted [{UNITS[key]}]",
               f"{LABELS[key]}  (n = {p.size})", legend=True)
    fig.suptitle("Predicted against true value on matched pulses", fontsize=FONT["title"])
    return save(fig, path)


def plot_metrics_by_n_pulses(all_pred, all_truth, path):

    """
    Draws the metrics against the number of true pulses in an example. Flat curves mean the
    encoder copes with busier scenes; curves that degrade as pulses are added indicate the fixed
    size encoder state is the limit. The zero pulse column reports false alarms on empty examples.

    ----------

    Parameters:
        all_pred (list) - predicted pulses per example.
        all_truth (list) - true pulses per example.
        path (str) - output file path.

    Returns:
        path (str) - the file written.
    """

    ks = sorted({len(t) for t in all_truth})
    rows = []
    for k in ks:
        idx = [i for i, t in enumerate(all_truth) if len(t) == k]
        m = evaluate([all_pred[i] for i in idx], [all_truth[i] for i in idx])
        reg = m["regression"]
        pick = lambda keys: ([reg[q]["mae"] for q in keys if reg[q]] or [np.nan])
        rows.append({"k": k, "n": len(idx),
                     "recall": m["detection"]["recall"],
                     "count_exact": m["count"]["exact_match_rate"],
                     "fp": m["detection"]["FP"] / len(idx),
                     "mae_t": float(np.mean(pick(("t_start", "t_stop")))),
                     "mae_f": float(np.mean(pick(("f1", "f2"))))})

    x_all = [r["k"] for r in rows]
    x_pulse = [r["k"] for r in rows if r["k"] > 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_GRID[0], FIG[1]))

    ax1.plot(x_all, [r["count_exact"] for r in rows], marker="o", color=PRIMARY,
             label="Exact count")
    ax1.plot(x_pulse, [r["recall"] for r in rows if r["k"] > 0], marker="s", color="#55A868",
             label="Recall")
    ax1.plot(x_all, [r["fp"] for r in rows], marker="^", ls="--", color=ACCENT,
             label="False alarms per example")
    for r in rows:
        ax1.annotate(f"n={r['n']}", (r["k"], -0.005), ha="center", va="top",
                     fontsize=FONT["annot"], color="0.4")
    ax1.set_xticks(x_all)
    finish(ax1, "True pulses in example", "Rate", "Detection", legend=True, rate_axis=True)

    ax2.plot(x_pulse, [r["mae_t"] for r in rows if r["k"] > 0], marker="o", color=PRIMARY,
             label="Time")
    ax2.set_xticks(x_pulse)
    ax2.tick_params(axis="y", labelcolor=PRIMARY)
    ax2b = ax2.twinx()
    ax2b.plot(x_pulse, [r["mae_f"] for r in rows if r["k"] > 0], marker="s", ls="--",
              color=ACCENT, label="Frequency")
    ax2b.set_ylabel("Frequency MAE [Hz]", color=ACCENT)
    ax2b.tick_params(axis="y", labelcolor=ACCENT)
    ax2b.grid(False)
    finish(ax2, "True pulses in example", "Time MAE [s]", "Localisation")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.suptitle("Performance against scene complexity", fontsize=FONT["title"])
    return save(fig, path)


def plot_loss_curves(outdir, prefix, train_path="train_losses.npy", val_path="val_losses.npy"):

    """
    Draws the teacher-forced training and validation loss per epoch. The validation minimum is
    marked for reference only: the saved checkpoint is chosen by the AR score, not by this loss.

    ----------

    Parameters:
        outdir (str) - directory the figure is written to.
        prefix (str) - filename prefix.
        train_path (str) - default 'train_losses.npy'.
        val_path (str) - default 'val_losses.npy'.

    Returns:
        path (str) - the file written, or None if the arrays were not found.
    """

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"  [plots] {train_path}/{val_path} not found, skipping loss curves")
        return None

    train, val = np.load(train_path), np.load(val_path)
    fig, ax = plt.subplots(figsize=FIG)
    ax.plot(range(len(train)), train, marker="o", color=PRIMARY, label="Training")
    ax.plot(range(len(val)), val, marker="s", color=ACCENT, label="Validation")
    if len(val):
        best = int(np.argmin(val))
        ax.axvline(best, color=REFERENCE, ls=":", lw=1.4,
                   label=f"Validation minimum (epoch {best})")
    finish(ax, "Epoch", "Teacher-forced loss", "Training and validation loss", legend=True)
    return save(fig, os.path.join(outdir, f"{prefix}_loss_curves.png"))


def plot_training_curves(outdir=".", prefix="train", components_csv="val_components.csv",
                         ar_path="val_ar_scores.npy", val_path="val_losses.npy"):

    """
    Draws the two training diagnostics from the per-epoch log written by train_model.py:

      ar_vs_tf    - the AR score (the selection criterion, from free-running generation) against
                    the teacher-forced loss (the training surrogate), each with its own axis and
                    its best epoch marked. Their rank correlation says how far the surrogate
                    drifts from the objective.
      components  - the individual terms behind the AR score, so a drift in one of them is
                    visible instead of hidden inside the total.

    If the component log is missing, ar_vs_tf is still drawn from the saved .npy series.

    ----------

    Parameters:
        outdir (str) - default '.'. Directory the figures are written to.
        prefix (str) - default 'train'. Filename prefix.
        components_csv (str) - default 'val_components.csv'.
        ar_path (str) - default 'val_ar_scores.npy'. Fallback AR series.
        val_path (str) - default 'val_losses.npy'. Fallback teacher-forced series.

    Returns:
        paths (list) - the files written.
    """

    data = None
    if os.path.exists(components_csv):
        data = np.atleast_1d(np.genfromtxt(components_csv, delimiter=",", names=True))
        epoch, ar, tf = data["epoch"], data["ar_score"], data["tf_val_loss"]
    elif os.path.exists(ar_path) and os.path.exists(val_path):
        ar, tf = np.load(ar_path), np.load(val_path)
        n = min(len(ar), len(tf))
        ar, tf, epoch = ar[:n], tf[:n], np.arange(n)
        print(f"  [plots] {components_csv} not found, drawing ar_vs_tf from the .npy series only")
    else:
        print(f"  [plots] no training log found, skipping training curves")
        return []
    if len(epoch) < 2:
        print("  [plots] fewer than two epochs, skipping training curves")
        return []

    ar_best, tf_best = int(np.nanargmin(ar)), int(np.nanargmin(tf))
    paths = []

    fig, ax = plt.subplots(figsize=FIG)
    ax.plot(epoch, ar, marker="o", color=PRIMARY, label="AR score (selection criterion)")
    ax.axvline(epoch[ar_best], color=PRIMARY, ls=":", lw=1.4,
               label=f"Best AR score (epoch {int(epoch[ar_best])})")
    ax.set_ylabel("AR score", color=PRIMARY)
    ax.tick_params(axis="y", labelcolor=PRIMARY)
    ax2 = ax.twinx()
    ax2.plot(epoch, tf, marker="s", color=ACCENT, label="Teacher-forced loss (surrogate)")
    ax2.axvline(epoch[tf_best], color=ACCENT, ls=":", lw=1.4,
                label=f"Lowest teacher-forced loss (epoch {int(epoch[tf_best])})")
    ax2.set_ylabel("Teacher-forced validation loss", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)
    ax2.grid(False)

    title = f"Selection criterion against training surrogate"
    if len(epoch) >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr(ar, tf).statistic
        title += (f"\nrank correlation {rho:.2f}; selecting on the surrogate would cost "
                  f"{ar[tf_best] - ar[ar_best]:+.3f} AR")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    paths.append(save(fig, os.path.join(outdir, f"{prefix}_ar_vs_tf.png")))

    if data is None:
        return paths

    fig, axes = plt.subplots(2, 2, figsize=FIG_GRID)
    for ax in axes.ravel():
        ax.axvline(epoch[ar_best], color=REFERENCE, ls=":", lw=1.2)

    ax = axes[0, 0]
    for key, name, colour, marker in (("precision", "Precision", PRIMARY, "o"),
                                      ("recall", "Recall", "#55A868", "s"),
                                      ("f1", "F1", ACCENT, "^")):
        ax.plot(epoch, data[key], marker=marker, color=colour, label=name)
    finish(ax, None, "Rate", "Detection", legend=True, rate_axis=True)

    ax = axes[0, 1]
    ax.plot(epoch, data["count_exact"], marker="o", color=PRIMARY, label="Exact count")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Exact count rate", color=PRIMARY)
    ax.tick_params(axis="y", labelcolor=PRIMARY)
    axb = ax.twinx()
    axb.plot(epoch, data["count_bias"], marker="s", ls="--", color=ACCENT, label="Count bias")
    axb.axhline(0.0, color=REFERENCE, lw=1.0)
    axb.set_ylabel("Count bias [pulses]", color=ACCENT)
    axb.tick_params(axis="y", labelcolor=ACCENT)
    axb.grid(False)
    ax.set_title("Pulse count (bias above zero means too many pulses)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    ax = axes[1, 0]
    ax.plot(epoch, data["type_acc"], marker="o", color=PRIMARY)
    finish(ax, "Epoch", "Accuracy", "Type accuracy on matched pulses", rate_axis=True)

    ax = axes[1, 1]
    ax.plot(epoch, data["mae_t_start"], marker="o", color=PRIMARY, label="Start time")
    ax.plot(epoch, data["mae_t_stop"], marker="s", color="#55A868", label="Stop time")
    ax.set_ylabel("Time MAE [s]")
    axb = ax.twinx()
    axb.plot(epoch, data["mae_f1"], marker="^", ls="--", color=ACCENT, label="Start frequency")
    axb.plot(epoch, data["mae_f2"], marker="v", ls="--", color="#C44E52", label="Stop frequency")
    axb.set_ylabel("Frequency MAE [Hz]")
    axb.grid(False)
    ax.set_xlabel("Epoch")
    ax.set_title("Regression error")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2)

    fig.suptitle(f"AR score components per epoch (dotted line: selected epoch "
                 f"{int(epoch[ar_best])})", fontsize=FONT["title"])
    paths.append(save(fig, os.path.join(outdir, f"{prefix}_components.png")))
    return paths


def make_all_plots(all_pred, all_truth, metrics, outdir=".", prefix="eval"):

    """
    Draws the full evaluation figure set, plus the training figures when their logs are present.

    ----------

    Parameters:
        all_pred (list) - predicted pulses per example.
        all_truth (list) - true pulses per example.
        metrics (dict) - output of metrics_core.evaluate.
        outdir (str) - default '.'. Directory the figures are written to.
        prefix (str) - default 'eval'. Filename prefix.

    Returns:
        paths (list) - the files written.
    """

    os.makedirs(outdir, exist_ok=True)
    count_cm, type_cm, reg = collect(all_pred, all_truth)
    at = lambda name: os.path.join(outdir, f"{prefix}_{name}.png")

    paths = [plot_confusion(count_cm, [str(c) for c in COUNTS],
                            "Pulses per example", "pulses", at("count_confusion"))]
    if type_cm.sum():
        paths.append(plot_confusion(type_cm, [type_name(t) for t in TYPES],
                                    "Pulse type on matched pulses", "type", at("type_confusion")))
    else:
        print("  [plots] no matched pulses, skipping the type confusion matrix")

    paths.append(plot_detection(metrics, at("detection")))
    paths.append(plot_error_histograms(reg, at("error_histograms")))
    paths.append(plot_pred_vs_true(reg, at("pred_vs_true")))
    paths.append(plot_metrics_by_n_pulses(all_pred, all_truth, at("by_n_pulses")))

    loss = plot_loss_curves(outdir, prefix)
    if loss:
        paths.append(loss)
    paths += plot_training_curves(outdir, prefix)

    print(f"  [plots] wrote {len(paths)} figures to '{outdir}/'")
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Draw the training figures from a finished run.")
    ap.add_argument("--components", default="val_components.csv")
    ap.add_argument("--ar-scores", default="val_ar_scores.npy")
    ap.add_argument("--val-losses", default="val_losses.npy")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--prefix", default="train")
    a = ap.parse_args()
    plot_training_curves(a.outdir, a.prefix, a.components, a.ar_scores, a.val_losses)
    plot_loss_curves(a.outdir, a.prefix)
