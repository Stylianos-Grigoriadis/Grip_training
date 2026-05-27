import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# =========================
# Selected indices
# Write here which indices you want to show
# Indices are based on the visible 50-point signal
# =========================
BLOCKED_SELECTED_INDICES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
PINK_SELECTED_INDICES = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
RANDOM_SELECTED_INDICES = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# =========================
# Global font settings
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# =========================
# Global settings
# =========================
DOT_ZOOM = 0.1
LOWEST_DOT_Y = 0.06
MIDDLE_YLIM = (-0.015, 0.22)
LOWEST_YLIM = (-0.015, 0.14)
LOWER_XLIM = (0, 13)

LABEL_BASE_Y = 0.04
LABEL_STEP_Y = 0.04
LABEL_CLOSE_THRESHOLD = 0.65

# =========================
# Helper function:
# rescale a signal so that its min becomes new_min
# and its max becomes new_max
# =========================
def rescale_to_range(signal, new_min=3, new_max=12):
    signal = np.asarray(signal, dtype=float)
    old_min = np.min(signal)
    old_max = np.max(signal)

    if old_max == old_min:
        return np.full_like(signal, (new_min + new_max) / 2, dtype=float)

    scaled = (signal - old_min) / (old_max - old_min)
    scaled = scaled * (new_max - new_min) + new_min
    return scaled

# =========================
# Helper function:
# select values using 1-based graph indices
# =========================
def select_values_from_indices(signal, selected_indices):
    selected_indices = np.asarray(selected_indices, dtype=int)

    if np.any(selected_indices < 1):
        raise ValueError("Selected indices must start from 1, not 0.")

    if np.any(selected_indices > len(signal)):
        raise ValueError(f"Selected indices cannot be larger than signal length: {len(signal)}")

    selected_values = signal[selected_indices - 1].copy()
    return selected_indices, selected_values

# =========================
# Helper function:
# stack label heights for points that are close in x
# lower index stays lower, higher index goes higher
# =========================
def stacked_label_positions(x_values, indices, base_y=0.04, step_y=0.025, close_threshold=0.65):
    x_values = np.asarray(x_values)
    indices = np.asarray(indices)

    order = np.argsort(indices)
    y_positions = np.full(len(x_values), base_y, dtype=float)

    placed_x = []
    placed_y = []

    for k in order:
        x = x_values[k]

        nearby_y_values = [
            py for px, py in zip(placed_x, placed_y)
            if abs(px - x) <= close_threshold
        ]

        if len(nearby_y_values) == 0:
            y = base_y
        else:
            y = max(nearby_y_values) + step_y

        y_positions[k] = y
        placed_x.append(x)
        placed_y.append(y)

    return y_positions

# =========================
# Helper function:
# create grouped labels for blocked practice
# Example: 36-40 and 41-45
# =========================
def create_blocked_group_labels(selected_indices, selected_values, base_y=0.055, step_y=0.04):
    selected_indices = np.asarray(selected_indices)
    selected_values = np.asarray(selected_values)

    labels = []
    unique_values = np.unique(selected_values)

    for i, value in enumerate(unique_values):
        indices_for_value = selected_indices[selected_values == value]

        if len(indices_for_value) == 1:
            label_text = str(indices_for_value[0])
        else:
            label_text = f"{indices_for_value[0]}-{indices_for_value[-1]}"

        labels.append((label_text, float(value), base_y + i * step_y))

    return labels

# =========================
# Helper function:
# draw lower graph
# =========================
def draw_lower_graph(
    ax,
    x_values,
    indices,
    red_dot,
    y_text_positions=None,
    xlim=(3, 12),
    xticks=None,
    ylim=(-0.015, 0.14),
    background_img=None,
    background_alpha=1.0,
    show_index_labels=True,
    custom_labels=None,
    show_reference_line=True,
    dot_y=0.00,
    dot_zoom=0.055
):
    xmin, xmax = xlim

    if background_img is not None:
        ax.imshow(
            background_img,
            extent=[xmin, xmax, ylim[0], ylim[1]],
            aspect="auto",
            alpha=background_alpha,
            zorder=0
        )

    if show_reference_line:
        ax.hlines(
            y=dot_y,
            xmin=xmin,
            xmax=xmax,
            color="black",
            linewidth=1,
            zorder=1
        )

    for x in x_values:
        imagebox = OffsetImage(red_dot, zoom=dot_zoom)
        ab = AnnotationBbox(
            imagebox,
            (x, dot_y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=3
        )
        ax.add_artist(ab)

    if show_index_labels and custom_labels is None and y_text_positions is not None:
        for x, idx, y_text in zip(x_values, indices, y_text_positions):
            ax.text(
                x,
                y_text,
                str(idx),
                ha="center",
                va="bottom",
                zorder=4
            )

    if show_index_labels and custom_labels is not None:
        for label_text, label_x, label_y in custom_labels:
            ax.text(
                label_x,
                label_y,
                label_text,
                ha="center",
                va="bottom",
                zorder=4
            )

    ax.set_xlabel("Distance from the basket")
    ax.set_yticks([])
    ax.set_ylim(*ylim)
    ax.set_xlim(xmin, xmax)

    if xticks is not None:
        ax.set_xticks(xticks)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# =========================
# Pink signal
# Keep only values 31 to 80 -> total 50 values
# Then rescale so min=3 and max=12
# =========================
pink = pd.read_excel("Pink signal.xlsx")
pink_signal_full = pink["Pink"].copy().to_numpy()

pink_signal_raw = pink_signal_full[30:80].copy()
pink_signal = rescale_to_range(pink_signal_raw, new_min=3, new_max=12)
time_pink = np.arange(1, len(pink_signal) + 1)

pink_selected_indices, pink_selected_values = select_values_from_indices(
    pink_signal,
    PINK_SELECTED_INDICES
)

# =========================
# Blocked practice signal
# 5 repetitions of each value from 3 to 12 -> total 50 values
# =========================
blocked_practice = np.repeat(np.arange(3, 13), 5)
time_blocked = np.arange(1, len(blocked_practice) + 1)

blocked_selected_indices, blocked_selected_values = select_values_from_indices(
    blocked_practice,
    BLOCKED_SELECTED_INDICES
)

blocked_lower_x = blocked_selected_values.astype(float)

# =========================
# Random practice signal
# 50 decimal-valued points, always the same because of seed
# Then rescale so min=3 and max=12
# =========================
rng = np.random.default_rng(42)
random_raw = rng.random(50)
random_practice = np.round(rescale_to_range(random_raw, new_min=3, new_max=12), 2)
time_random = np.arange(1, len(random_practice) + 1)

random_selected_indices, random_selected_values = select_values_from_indices(
    random_practice,
    RANDOM_SELECTED_INDICES
)

# =========================
# Label heights
# =========================
pink_label_y = stacked_label_positions(
    pink_selected_values,
    pink_selected_indices,
    base_y=LABEL_BASE_Y,
    step_y=LABEL_STEP_Y,
    close_threshold=LABEL_CLOSE_THRESHOLD
)

random_label_y = stacked_label_positions(
    random_selected_values,
    random_selected_indices,
    base_y=LABEL_BASE_Y,
    step_y=LABEL_STEP_Y,
    close_threshold=LABEL_CLOSE_THRESHOLD
)

blocked_custom_labels = create_blocked_group_labels(
    blocked_selected_indices,
    blocked_selected_values,
    base_y=0.055,
    step_y=0.04
)

# =========================
# Images
# =========================
red_dot = mpimg.imread("red dots for figures.png")
court_img = mpimg.imread("Court.png")

# =========================
# Figure
# height_ratios:
# first number  -> upper graphs height
# second number -> middle graphs height
# third number  -> lowest graphs height
# =========================
fig, axes = plt.subplots(
    3,
    3,
    figsize=(16, 9),
    gridspec_kw={"height_ratios": [2, 0.75, 2]},
    constrained_layout=False
)

ax_blocked_top = axes[0, 0]
ax_pink_top = axes[0, 1]
ax_random_top = axes[0, 2]

ax_blocked_middle = axes[1, 0]
ax_pink_middle = axes[1, 1]
ax_random_middle = axes[1, 2]

ax_blocked_lowest = axes[2, 0]
ax_pink_lowest = axes[2, 1]
ax_random_lowest = axes[2, 2]

# =========================================================
# TOP LEFT: Blocked practice
# =========================================================
ax_blocked_top.scatter(
    time_blocked,
    blocked_practice,
    color="grey",
    s=45,
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
    alpha=0.5
)

ax_blocked_top.plot(
    time_blocked,
    blocked_practice,
    color="darkred",
    alpha=0.35,
    linewidth=1.5,
    zorder=2
)

ax_blocked_top.scatter(
    blocked_selected_indices,
    blocked_selected_values,
    color="red",
    s=70,
    edgecolor="black",
    linewidth=0.7,
    zorder=4
)

ax_blocked_top.set_title("Blocked Practice", fontweight="bold")
ax_blocked_top.set_xlabel("Index")
ax_blocked_top.set_ylabel("Signal Magnitude")
ax_blocked_middle.set_ylabel("Practice Order")
ax_blocked_lowest.set_ylabel("Practice Location")
ax_blocked_top.set_xlim(0, 52)
ax_blocked_top.set_ylim(2.5, 12.5)

ax_blocked_top.spines["top"].set_visible(False)
ax_blocked_top.spines["right"].set_visible(False)

# =========================================================
# TOP MIDDLE: Pink signal
# =========================================================
ax_pink_top.scatter(
    time_pink,
    pink_signal,
    color="grey",
    s=45,
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
    alpha=0.5
)

ax_pink_top.plot(
    time_pink,
    pink_signal,
    color="darkred",
    alpha=0.35,
    linewidth=1.5,
    zorder=2
)

ax_pink_top.scatter(
    pink_selected_indices,
    pink_selected_values,
    color="red",
    s=70,
    edgecolor="black",
    linewidth=0.7,
    zorder=4
)

ax_pink_top.set_title("Structured Practice", fontweight="bold")
ax_pink_top.set_xlabel("Index")
ax_pink_top.set_ylabel("")
ax_pink_top.set_yticks([])
ax_pink_top.set_xlim(0, 52)
ax_pink_top.set_ylim(2.5, 12.5)

ax_pink_top.spines["top"].set_visible(False)
ax_pink_top.spines["right"].set_visible(False)
ax_pink_top.spines["left"].set_visible(False)

# =========================================================
# TOP RIGHT: Random practice
# =========================================================
ax_random_top.scatter(
    time_random,
    random_practice,
    color="grey",
    s=45,
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
    alpha=0.5
)

ax_random_top.plot(
    time_random,
    random_practice,
    color="darkred",
    alpha=0.35,
    linewidth=1.5,
    zorder=2
)

ax_random_top.scatter(
    random_selected_indices,
    random_selected_values,
    color="red",
    s=70,
    edgecolor="black",
    linewidth=0.7,
    zorder=4
)

ax_random_top.set_title("Contextual Interference", fontweight="bold")
ax_random_top.set_xlabel("Index")
ax_random_top.set_ylabel("")
ax_random_top.set_yticks([])
ax_random_top.set_xlim(0, 52)
ax_random_top.set_ylim(2.5, 12.5)

ax_random_top.spines["top"].set_visible(False)
ax_random_top.spines["right"].set_visible(False)
ax_random_top.spines["left"].set_visible(False)

# =========================================================
# MIDDLE ROW: lower graphs without background
# =========================================================
draw_lower_graph(
    ax=ax_blocked_middle,
    x_values=blocked_lower_x,
    indices=blocked_selected_indices,
    red_dot=red_dot,
    y_text_positions=None,
    xlim=LOWER_XLIM,
    xticks=np.arange(1, 13),
    ylim=MIDDLE_YLIM,
    background_img=None,
    show_index_labels=True,
    custom_labels=blocked_custom_labels,
    show_reference_line=True,
    dot_y=0.00,
    dot_zoom=DOT_ZOOM
)

draw_lower_graph(
    ax=ax_pink_middle,
    x_values=pink_selected_values,
    indices=pink_selected_indices,
    red_dot=red_dot,
    y_text_positions=pink_label_y,
    xlim=LOWER_XLIM,
    xticks=np.arange(1, 13),
    ylim=MIDDLE_YLIM,
    background_img=None,
    show_index_labels=True,
    custom_labels=None,
    show_reference_line=True,
    dot_y=0.00,
    dot_zoom=DOT_ZOOM
)

draw_lower_graph(
    ax=ax_random_middle,
    x_values=random_selected_values,
    indices=random_selected_indices,
    red_dot=red_dot,
    y_text_positions=random_label_y,
    xlim=LOWER_XLIM,
    xticks=np.arange(1, 13),
    ylim=MIDDLE_YLIM,
    background_img=None,
    show_index_labels=True,
    custom_labels=None,
    show_reference_line=True,
    dot_y=0.00,
    dot_zoom=DOT_ZOOM
)

# =========================================================
# LOWEST ROW: same lower graphs with Court background
# =========================================================
draw_lower_graph(
    ax=ax_blocked_lowest,
    x_values=blocked_lower_x,
    indices=blocked_selected_indices,
    red_dot=red_dot,
    y_text_positions=None,
    xlim=LOWER_XLIM,
    xticks=np.arange(1, 13),
    ylim=LOWEST_YLIM,
    background_img=court_img,
    background_alpha=1.0,
    show_index_labels=False,
    custom_labels=None,
    show_reference_line=False,
    dot_y=LOWEST_DOT_Y,
    dot_zoom=DOT_ZOOM
)

draw_lower_graph(
    ax=ax_pink_lowest,
    x_values=pink_selected_values,
    indices=pink_selected_indices,
    red_dot=red_dot,
    y_text_positions=None,
    xlim=LOWER_XLIM,
    xticks=np.arange(1, 13),
    ylim=LOWEST_YLIM,
    background_img=court_img,
    background_alpha=1.0,
    show_index_labels=False,
    custom_labels=None,
    show_reference_line=False,
    dot_y=LOWEST_DOT_Y,
    dot_zoom=DOT_ZOOM
)

draw_lower_graph(
    ax=ax_random_lowest,
    x_values=random_selected_values,
    indices=random_selected_indices,
    red_dot=red_dot,
    y_text_positions=None,
    xlim=LOWER_XLIM,
    xticks=np.arange(1, 13),
    ylim=LOWEST_YLIM,
    background_img=court_img,
    background_alpha=1.0,
    show_index_labels=False,
    custom_labels=None,
    show_reference_line=False,
    dot_y=LOWEST_DOT_Y,
    dot_zoom=DOT_ZOOM
)

# =========================
# Manual spacing
# =========================
fig.subplots_adjust(
    left=0.05,
    right=0.98,
    bottom=0.06,
    top=0.93,
    wspace=0.10,
    hspace=0.12
)

ax_blocked_top.yaxis.set_label_coords(-0.09, 0.5)
ax_blocked_middle.yaxis.set_label_coords(-0.09, 0.5)
ax_blocked_lowest.yaxis.set_label_coords(-0.09, 0.5)

plt.show()