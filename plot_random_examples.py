"""
Sanity-check visualization of the generated dataset: draws a grid of randomly chosen example
spectrograms with their ground-truth pulses overlaid as boxes (colored by type, dashed and
edge-only, and expanded to a minimum height so a CW pulse stays visible inside its own box), plus the SNR and
pulse count from META in each title. Use it after generating or merging a dataset to eyeball that
pulses sit where the targets say, that the SNR range looks right, and that noise-only examples are
present.

Works with either the raw variable-length TARGET.npy or the fixed-length padded_sequences.npy
(rows after the EOS token are ignored). Targets are read in physical units, as stored on disk.

Run:
    python plot_random_examples.py --n 8 --seed 0
    python plot_random_examples.py --input hat01_INPUT.npy --target hat01_TARGET.npy --meta hat01_META.npy --axes hat01_AXES.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_style import FONT, TYPE_COLORS, type_name, finish, save


CLASS_NAMES = {0: "cw", 1: "lfm", 2: "hfm", 3: "eos"}


def read_pulses(row_seq):

    """
    Reads the pulses of one target sequence, stopping at the first EOS row. Accepts both raw
    variable-length sequences and padded fixed-length ones.

    ----------

    Parameters:
        row_seq (ndarray) - target rows [cw, lfm, hfm, eos, t_start, t_stop, f1, f2].

    Returns:
        pulses (list) - one dict per pulse, keys type, t_start, t_stop, f1, f2.
    """

    pulses = []
    for row in np.asarray(row_seq, dtype=np.float64):
        k = int(np.argmax(row[:4]))
        if k == 3:
            break
        pulses.append({"type": CLASS_NAMES[k], "t_start": row[4], "t_stop": row[5],
                       "f1": row[6], "f2": row[7]})
    return pulses


def main():

    """
    Parses arguments, samples random examples, and saves the annotated grid figure.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="INPUT.npy")
    ap.add_argument("--target", default="padded_sequences.npy",
                    help="padded_sequences.npy or the raw TARGET.npy")
    ap.add_argument("--meta", default="META.npy")
    ap.add_argument("--axes", default="AXES.npz")
    ap.add_argument("--n", type=int, default=8, help="number of examples to plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="random_examples.png")
    args = ap.parse_args()

    X = np.load(args.input, mmap_mode="r")
    y = np.load(args.target, allow_pickle=True)
    meta = np.load(args.meta)
    axes = np.load(args.axes)
    t_axis, f_axis = axes["t"], axes["f"]
    assert len(X) == len(y) == len(meta), "INPUT, targets, and META must be index-aligned"

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.n, len(X)), replace=False)

    ncols = min(4, len(idx))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)

    for panel, i in enumerate(idx):
        ax = axs[panel // ncols][panel % ncols]
        ax.pcolormesh(t_axis, f_axis, np.array(X[i]), shading="gouraud", cmap="plasma")
        f_lo, f_hi = float(f_axis[0]), float(f_axis[-1])
        f_range = f_hi - f_lo
        for p in read_pulses(y[i]):
            lo, hi = min(p["f1"], p["f2"]), max(p["f1"], p["f2"])
            # A CW pulse has f1 == f2, so its box would collapse onto the ridge and hide the very
            # thing it marks. Expand symmetrically to a minimum height of 2% of the visible
            # frequency range, leaving the pulse visible inside the box.
            min_h = max(f_range * 0.02, 1.0)
            if hi - lo < min_h:
                pad = 0.5 * (min_h - (hi - lo))
                lo, hi = lo - pad, hi + pad
            colour = TYPE_COLORS[p["type"]]
            ax.add_patch(Rectangle((p["t_start"], lo), p["t_stop"] - p["t_start"], hi - lo,
                                   fill=False, edgecolor=colour, linewidth=1.4, linestyle="--"))
            ax.text(p["t_start"], hi + f_range * 0.015, type_name(p["type"]),
                    color=colour, fontsize=FONT["annot"], va="bottom")
        snr, n_pulses = meta[i, 0], int(meta[i, 2])
        title = (f"Example {i}: noise only" if n_pulses == 0
                 else f"Example {i}: {n_pulses} pulse(s), SNR {snr:.1f} dB")
        if meta.shape[1] > 4:
            title += f", recording {int(meta[i, 4])}"
        finish(ax, "Time [s]", "Frequency [Hz]", title)

    for panel in range(len(idx), nrows * ncols):
        axs[panel // ncols][panel % ncols].axis("off")

    save(fig, args.out)


if __name__ == "__main__":
    main()
