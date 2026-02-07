import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch


def plot_saen_training_sets_all(results, box_width=0.08, jitter=0.015):
    """
    Boxplots with TRUE spacing between groups, outlined datapoints,
    readable legend labels, hatching for Slow groups (including legend),
    and zoomed x-axis.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing SaEn values per training set.

    box_width : float
        Width of each boxplot.

    jitter : float
        Horizontal jitter for datapoints.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    # -------------------------------
    # SaEn columns
    # -------------------------------
    saen_cols = [
        'SaEn training set 1', 'SaEn training set 2',
        'SaEn training set 3', 'SaEn training set 4',
        'SaEn training set 5', 'SaEn training set 6',
        'SaEn training set 7', 'SaEn training set 8',
        'SaEn training set 9', 'SaEn training set 10'
    ]

    # -------------------------------
    # Melt dataframe
    # -------------------------------
    df_long = results.melt(
        id_vars=['Group ID'],
        value_vars=saen_cols,
        var_name='Training Set',
        value_name='SaEn'
    )

    # -------------------------------
    # Orders and labels
    # -------------------------------
    set_order = saen_cols
    set_labels = [f'Set {i}' for i in range(1, 11)]

    group_order = [
        "Sine_100", "Pink_100", "White_100",
        "Sine_65", "Pink_65", "White_65"
    ]

    legend_labels = {
        "Sine_100": "Sine Fast",
        "Pink_100": "Pink Fast",
        "White_100": "White Fast",
        "Sine_65": "Sine Slow",
        "Pink_65": "Pink Slow",
        "White_65": "White Slow",
    }

    palette = {
        "Sine_100": "#4F4F4F",
        "Pink_100": "#FFC0CB",
        "White_100": "#D3D3D3",
        "Sine_65": "#4F4F4F",
        "Pink_65": "#FFC0CB",
        "White_65": "#D3D3D3",
    }

    # -------------------------------
    # X positions
    # -------------------------------
    base_positions = np.arange(len(set_order))
    group_offsets = np.linspace(-0.30, 0.30, len(group_order))

    plt.figure(figsize=(15, 7))
    ax = plt.gca()

    # -------------------------------
    # Draw boxplots + datapoints
    # -------------------------------
    for gi, group in enumerate(group_order):
        for si, set_name in enumerate(set_order):

            data = df_long.loc[
                (df_long['Group ID'] == group) &
                (df_long['Training Set'] == set_name),
                'SaEn'
            ].dropna()

            if data.empty:
                continue

            pos = base_positions[si] + group_offsets[gi]

            bp = ax.boxplot(
                data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )

            for patch in bp['boxes']:
                patch.set_facecolor(palette[group])
                patch.set_edgecolor('black')
                patch.set_alpha(0.9)

                # Hatch Slow groups
                if group.endswith('_65'):
                    patch.set_hatch('////')

            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1.2)

            # Raw datapoints
            x_jitter = np.random.normal(pos, jitter, size=len(data))
            ax.scatter(
                x_jitter,
                data,
                s=35,
                facecolor=palette[group],
                edgecolor='black',
                linewidth=0.8,
                alpha=0.75,
                zorder=3
            )

    # -------------------------------
    # Axes formatting
    # -------------------------------
    ax.set_xticks(base_positions)
    ax.set_xticklabels(set_labels)
    ax.set_ylabel("Sample Entropy (SaEn)")
    ax.set_title("Sample Entropy Across Training Sets")
    ax.set_xlabel("")

    # Zoom x-axis to remove whitespace
    ax.set_xlim(
        base_positions[0] - 0.45,
        base_positions[-1] + 0.45
    )

    # -------------------------------
    # Legend with texture
    # -------------------------------
    legend_handles = []
    legend_texts = []

    for g in group_order:
        patch = Patch(
            facecolor=palette[g],
            edgecolor='black',
            label=legend_labels[g]
        )

        if g.endswith('_65'):
            patch.set_hatch('////')

        legend_handles.append(patch)
        legend_texts.append(legend_labels[g])

    ax.legend(
        handles=legend_handles,
        labels=legend_texts,
        title='Group',
        bbox_to_anchor=(0.5, -0.18),
        loc='upper center',
        ncol=6
    )

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def plot_saen_training_sets_with_slopes(results, box_width=0.08, jitter=0.015, show_slopes=True, show_points=True):
    """
    Boxplots with TRUE spacing between groups and outlined datapoints.
    Optionally overlays group-level linear regression slopes computed
    from the mean SaEn across training sets (1–10).

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing SaEn values per training set.

    box_width : float
        Width of each boxplot.

    jitter : float
        Horizontal jitter for datapoints.

    show_slopes : bool
        Whether to plot group-level regression slopes.

    show_points : bool
        Whether to show individual datapoints.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    # -------------------------------
    # Exclude participants
    # -------------------------------
    results = results[results['Exclude'] == 0].copy()

    # -------------------------------
    # SaEn columns
    # -------------------------------
    saen_cols = [
        'SaEn training set 1', 'SaEn training set 2',
        'SaEn training set 3', 'SaEn training set 4',
        'SaEn training set 5', 'SaEn training set 6',
        'SaEn training set 7', 'SaEn training set 8',
        'SaEn training set 9', 'SaEn training set 10'
    ]

    # -------------------------------
    # Melt dataframe
    # -------------------------------
    df_long = results.melt(
        id_vars=['Signal'],
        value_vars=saen_cols,
        var_name='Training Set',
        value_name='SaEn'
    )

    # -------------------------------
    # Orders and labels
    # -------------------------------
    set_order = saen_cols
    set_labels = [f'Set {i}' for i in range(1, 11)]

    group_order = [
        "Sine",
        "Pink",
        "White"
    ]

    legend_labels = {
        "Sine": "Sine",
        "Pink": "Pink",
        "White": "White",
    }

    palette = {
        "Sine": "#4F4F4F",
        "Pink": "#FFC0CB",
        "White": "#D3D3D3",
    }

    # -------------------------------
    # X positions
    # -------------------------------
    base_positions = np.arange(len(set_order))
    group_offsets = np.linspace(-0.20, 0.20, len(group_order))

    plt.figure(figsize=(15, 7))
    ax = plt.gca()

    # -------------------------------
    # Draw boxplots (+ optional datapoints)
    # -------------------------------
    for gi, group in enumerate(group_order):
        for si, set_name in enumerate(set_order):

            data = df_long.loc[
                (df_long['Signal'] == group) &
                (df_long['Training Set'] == set_name),
                'SaEn'
            ].dropna()

            if data.empty:
                continue

            pos = base_positions[si] + group_offsets[gi]

            bp = ax.boxplot(
                data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )

            for patch in bp['boxes']:
                patch.set_facecolor(palette[group])
                patch.set_edgecolor('black')
                patch.set_alpha(0.9)

            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1.2)

            # ---- Raw datapoints (optional)
            if show_points:
                x_jitter = np.random.normal(pos, jitter, size=len(data))
                ax.scatter(
                    x_jitter,
                    data,
                    s=35,
                    facecolor=palette[group],
                    edgecolor='black',
                    linewidth=0.8,
                    alpha=0.75,
                    zorder=3
                )

    # -------------------------------
    # Group-level regression slopes
    # -------------------------------
    if show_slopes:
        for gi, group in enumerate(group_order):

            means = []
            x_positions = []

            for si, set_name in enumerate(set_order):
                values = df_long.loc[
                    (df_long['Signal'] == group) &
                    (df_long['Training Set'] == set_name),
                    'SaEn'
                ].dropna()

                if not values.empty:
                    means.append(values.mean())
                    x_positions.append(base_positions[si] + group_offsets[gi])

            means = np.array(means)
            x_num = np.arange(1, len(means) + 1)

            if len(means) >= 2:
                slope, intercept = np.polyfit(x_num, means, 1)
                y_fit = slope * x_num + intercept

                ax.plot(
                    x_positions,
                    y_fit,
                    color='k',
                    linewidth=5,
                    zorder=4
                )

                ax.plot(
                    x_positions,
                    y_fit,
                    color=palette[group],
                    linewidth=3,
                    zorder=5
                )

    # -------------------------------
    # Axes formatting
    # -------------------------------
    ax.set_xticks(base_positions)
    ax.set_xticklabels(set_labels)
    ax.set_ylabel("Sample Entropy (SaEn)")
    ax.set_title("Sample Entropy Across Training Sets")
    ax.set_xlabel("")

    ax.set_xlim(
        base_positions[0] - 0.35,
        base_positions[-1] + 0.35
    )

    # -------------------------------
    # Legend
    # -------------------------------
    legend_handles = []
    legend_texts = []

    for g in group_order:
        patch = Patch(
            facecolor=palette[g],
            edgecolor='black',
            label=legend_labels[g]
        )
        legend_handles.append(patch)
        legend_texts.append(legend_labels[g])

    ax.legend(
        handles=legend_handles,
        labels=legend_texts,
        title='Group',
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=3
    )

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def compute_saen_slope(results):
    """
    Calculate the slope of SaEn across training sets (1–10)
    for each participant (row).

    Returns
    -------
    pandas.Series
        One slope value per participant.
    """

    import numpy as np
    import pandas as pd

    saen_cols = [
        'SaEn training set 1', 'SaEn training set 2',
        'SaEn training set 3', 'SaEn training set 4',
        'SaEn training set 5', 'SaEn training set 6',
        'SaEn training set 7', 'SaEn training set 8',
        'SaEn training set 9', 'SaEn training set 10'
    ]

    x = np.arange(1, 11)
    slopes = []

    for i in range(len(results)):
        y = results.iloc[i][saen_cols].values.astype(float)
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)

    return pd.Series(slopes, index=results.index, name='SaEn_slope')

def plot_DFA_training_sets_with_slopes(results, box_width=0.08, jitter=0.015, show_slopes=True, show_points=True):
    """
    Boxplots with TRUE spacing between groups and outlined datapoints.
    Optionally overlays group-level linear regression slopes computed
    from the mean SaEn across training sets (1–10).

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing SaEn values per training set.

    box_width : float
        Width of each boxplot.

    jitter : float
        Horizontal jitter for datapoints.

    show_slopes : bool
        Whether to plot group-level regression slopes.

    show_points : bool
        Whether to show individual datapoints.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    # -------------------------------
    # Exclude participants
    # -------------------------------
    results = results[results['Exclude'] == 0].copy()

    # -------------------------------
    # SaEn columns
    # -------------------------------
    saen_cols = [
        'DFA training set 1', 'DFA training set 2',
        'DFA training set 3', 'DFA training set 4',
        'DFA training set 5', 'DFA training set 6',
        'DFA training set 7', 'DFA training set 8',
        'DFA training set 9', 'DFA training set 10'
    ]

    # -------------------------------
    # Melt dataframe
    # -------------------------------
    df_long = results.melt(
        id_vars=['Signal'],
        value_vars=saen_cols,
        var_name='Training Set',
        value_name='DFA'
    )

    # -------------------------------
    # Orders and labels
    # -------------------------------
    set_order = saen_cols
    set_labels = [f'Set {i}' for i in range(1, 11)]

    group_order = [
        "Sine",
        "Pink",
        "White"
    ]

    legend_labels = {
        "Sine": "Sine",
        "Pink": "Pink",
        "White": "White",
    }

    palette = {
        "Sine": "#4F4F4F",
        "Pink": "#FFC0CB",
        "White": "#D3D3D3",
    }

    # -------------------------------
    # X positions
    # -------------------------------
    base_positions = np.arange(len(set_order))
    group_offsets = np.linspace(-0.20, 0.20, len(group_order))

    plt.figure(figsize=(15, 7))
    ax = plt.gca()

    # -------------------------------
    # Draw boxplots (+ optional datapoints)
    # -------------------------------
    for gi, group in enumerate(group_order):
        for si, set_name in enumerate(set_order):

            data = df_long.loc[
                (df_long['Signal'] == group) &
                (df_long['Training Set'] == set_name),
                'DFA'
            ].dropna()

            if data.empty:
                continue

            pos = base_positions[si] + group_offsets[gi]

            bp = ax.boxplot(
                data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )

            for patch in bp['boxes']:
                patch.set_facecolor(palette[group])
                patch.set_edgecolor('black')
                patch.set_alpha(0.9)

            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1.2)

            # ---- Raw datapoints (optional)
            if show_points:
                x_jitter = np.random.normal(pos, jitter, size=len(data))
                ax.scatter(
                    x_jitter,
                    data,
                    s=35,
                    facecolor=palette[group],
                    edgecolor='black',
                    linewidth=0.8,
                    alpha=0.75,
                    zorder=3
                )

    # -------------------------------
    # Group-level regression slopes
    # -------------------------------
    if show_slopes:
        for gi, group in enumerate(group_order):

            means = []
            x_positions = []

            for si, set_name in enumerate(set_order):
                values = df_long.loc[
                    (df_long['Signal'] == group) &
                    (df_long['Training Set'] == set_name),
                    'DFA'
                ].dropna()

                if not values.empty:
                    means.append(values.mean())
                    x_positions.append(base_positions[si] + group_offsets[gi])

            means = np.array(means)
            x_num = np.arange(1, len(means) + 1)

            if len(means) >= 2:
                slope, intercept = np.polyfit(x_num, means, 1)
                y_fit = slope * x_num + intercept

                ax.plot(
                    x_positions,
                    y_fit,
                    color='k',
                    linewidth=5,
                    zorder=4
                )

                ax.plot(
                    x_positions,
                    y_fit,
                    color=palette[group],
                    linewidth=3,
                    zorder=5
                )

    # -------------------------------
    # Axes formatting
    # -------------------------------
    ax.set_xticks(base_positions)
    ax.set_xticklabels(set_labels)
    ax.set_ylabel("Detrended Fluctuation Analysis (DFA)")
    ax.set_title("Exponent α Across Training Sets")
    ax.set_xlabel("")

    ax.set_xlim(
        base_positions[0] - 0.35,
        base_positions[-1] + 0.35
    )

    # -------------------------------
    # Legend
    # -------------------------------
    legend_handles = []
    legend_texts = []

    for g in group_order:
        patch = Patch(
            facecolor=palette[g],
            edgecolor='black',
            label=legend_labels[g]
        )
        legend_handles.append(patch)
        legend_texts.append(legend_labels[g])

    ax.legend(
        handles=legend_handles,
        labels=legend_texts,
        title='Group',
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=3
    )

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def compute_DFA_slope(results):
    """
    Calculate the slope of SaEn across training sets (1–10)
    for each participant (row).

    Returns
    -------
    pandas.Series
        One slope value per participant.
    """

    saen_cols = [
        'DFA training set 1', 'DFA training set 2',
        'DFA training set 3', 'DFA training set 4',
        'DFA training set 5', 'DFA training set 6',
        'DFA training set 7', 'DFA training set 8',
        'DFA training set 9', 'DFA training set 10'
    ]

    x = np.arange(1, 11)
    slopes = []

    for i in range(len(results)):
        y = results.iloc[i][saen_cols].values.astype(float)
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)

    return pd.Series(slopes, index=results.index, name='DFA_slope')



directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results'
os.chdir(directory)
# results = pd.read_excel('Training_trials_SaEn_results.xlsx')
results_time_delay = pd.read_excel('Training_trials_SaEn_time_delay_results.xlsx')
results_DFA = pd.read_excel('Training_trials_DFA_10_N10_results.xlsx')
print(results_time_delay.columns)


# plot_saen_training_sets_all(results)
# plot_saen_training_sets_with_slopes(results, box_width=0.15)
# plot_saen_training_sets_with_slopes(results_time_delay, box_width=0.15, show_points=True)

plot_DFA_training_sets_with_slopes(results_DFA, box_width=0.15, show_points=True)

