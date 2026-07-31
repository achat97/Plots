"""
Shared figure conventions, so every script in the pipeline produces figures that look and read the
same way.

Conventions applied everywhere:
  - axis labels are sentence case with the unit in brackets: "Start frequency [Hz]"
  - titles are sentence case and state what is shown, not how it was computed
  - pulse types are displayed as the uppercase acronyms CW / LFM / HFM, always in the same colour
  - rates are plotted on a linear 0-1 axis; log scales are used only where the data spans decades
  - legends are smaller than the axis labels and never sit on top of the data
  - every reference line carries a legend entry saying what it is

Import `LABELS`/`UNITS` for the canonical name of a quantity instead of hard-coding strings, so a
rename happens in one place.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Figure sizes (inches). Chosen so the default font sizes stay legible when a figure is embedded
# at roughly half a page; scale FONT with the figure if you need larger text in a printed report.
FIG = (8.0, 5.0)          # single panel
FIG_WIDE = (12.0, 5.0)    # time series spanning a long recording
FIG_GRID = (11.0, 8.0)    # 2 x 2 panels

FONT = {"title": 13, "label": 12, "tick": 11, "legend": 10, "annot": 10}

# One colour per pulse type, used by every script.
TYPE_COLORS = {"cw": "#4C72B0", "lfm": "#55A868", "hfm": "#C44E52"}
TYPE_NAMES = {"cw": "CW", "lfm": "LFM", "hfm": "HFM"}

PRIMARY = "#4C72B0"
ACCENT = "#DD8452"
REFERENCE = "0.35"        # colour for reference lines (chance level, thresholds, ...)

# Canonical display name and unit for every regression quantity. The code keeps its internal
# names (f1, f2); figures always show these.
LABELS = {"t_start": "Start time", "t_stop": "Stop time",
          "f1": "Start frequency", "f2": "Stop frequency"}
UNITS = {"t_start": "s", "t_stop": "s", "f1": "Hz", "f2": "Hz"}


def use_style():

    """
    Applies the shared rcParams. Called on import by every plotting module.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.figsize": FIG,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"],
        "axes.titleweight": "normal",
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "legend.framealpha": 0.9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "grid.alpha": 0.3,
    })


def axis_label(key):

    """
    Returns the canonical axis label for a regression quantity, e.g. 'Start frequency [Hz]'.

    ----------

    Parameters:
        key (str) - one of 't_start', 't_stop', 'f1', 'f2'.

    Returns:
        (str) - the label.
    """

    return f"{LABELS[key]} [{UNITS[key]}]"


def type_name(key):

    """
    Returns the display name of a pulse type, e.g. 'lfm' -> 'LFM'.

    ----------

    Parameters:
        key (str) - internal type name.

    Returns:
        (str) - the display name.
    """

    return TYPE_NAMES.get(key, key.upper())


def finish(ax, xlabel=None, ylabel=None, title=None, legend=False, rate_axis=False,
           legend_loc="best"):

    """
    Applies labels, an optional legend, and the standard rate axis limits to one set of axes.

    ----------

    Parameters:
        ax (Axes) - the axes to finish.
        xlabel (str or None) - x-axis label.
        ylabel (str or None) - y-axis label.
        title (str or None) - axes title.
        legend (bool) - default False. Draw a legend if the axes has labelled artists.
        rate_axis (bool) - default False. Fix the y-axis to the 0-1 range used for rates.
        legend_loc (str) - default 'best'. Legend location.

    Returns:
        None
    """

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if rate_axis:
        ax.set_ylim(-0.03, 1.03)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)


def save(fig, path, quiet=False):

    """
    Saves and closes a figure, and reports the path.

    ----------

    Parameters:
        fig (Figure) - the figure to save.
        path (str) - output file path.
        quiet (bool) - default False. Suppress the printed confirmation.

    Returns:
        path (str) - the path written.
    """

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    if not quiet:
        print(f"Saved {path}")
    return path


def hms(seconds, decimals=2):

    """
    Formats seconds as clock time, e.g. 3725.4 -> '1:02:05.40'. Long recordings are easier to
    navigate in clock time than in raw seconds.

    ----------

    Parameters:
        seconds (float) - time offset from the start of the recording, in seconds.
        decimals (int) - default 2. Digits kept on the seconds field.

    Returns:
        (str) - the formatted timestamp.
    """

    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    # round BEFORE splitting, or a value like 59.999 s formats as the invalid "0:00:60.00"
    seconds = round(abs(seconds), decimals)
    h, rem = divmod(seconds, 3600.0)
    m, s = divmod(rem, 60.0)
    width = 2 if decimals == 0 else decimals + 3
    return f"{sign}{int(h)}:{int(m):02d}:{s:0{width}.{decimals}f}"


def hms_axis(ax):

    """
    Formats an axis of time offsets in seconds as clock time ticks.

    ----------

    Parameters:
        ax (Axes) - axes whose x-axis holds time offsets in seconds.

    Returns:
        None
    """

    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: hms(v, decimals=0)))


use_style()
