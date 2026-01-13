import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

def plot_time_to_adapt_seaborn(results, mode='min', box_width=0.08, jitter=0.015):
    """
    Boxplots with TRUE spacing between groups, outlined datapoints,
    readable legend labels, hatching for Slow groups (including legend),
    and zoomed x-axis.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with adaptation results.

    mode : {'min', 'avg', 'trial1', 'trial2', 'trial3'}
        Which dependent variables to plot.

    box_width : float
        Width of each boxplot.

    jitter : float
        Horizontal jitter for datapoints.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    mode = mode.lower()

    # -------------------------------
    # Column mapping
    # -------------------------------
    column_map = {
        'min': [
            'Min Time to adapt before down',
            'Min Time to adapt after down',
            'Min Time to adapt before up',
            'Min Time to adapt after up'
        ],
        'avg': [
            'Average Time to adapt before down',
            'Average Time to adapt after down',
            'Average Time to adapt before up',
            'Average Time to adapt after up'
        ],
        'trial1': [
            'Time_to_adapt_before_down_1',
            'Time_to_adapt_after_down_1',
            'Time_to_adapt_before_up_1',
            'Time_to_adapt_after_up_1'
        ],
        'trial2': [
            'Time_to_adapt_before_down_2',
            'Time_to_adapt_after_down_2',
            'Time_to_adapt_before_up_2',
            'Time_to_adapt_after_up_2'
        ],
        'trial3': [
            'Time_to_adapt_before_down_3',
            'Time_to_adapt_after_down_3',
            'Time_to_adapt_before_up_3',
            'Time_to_adapt_after_up_3'
        ]
    }

    if mode not in column_map:
        raise ValueError(
            "mode must be one of: 'min', 'avg', 'trial1', 'trial2', 'trial3'"
        )

    value_cols = column_map[mode]

    # -------------------------------
    # Melt dataframe
    # -------------------------------
    df_long = results.melt(
        id_vars=['Group ID'],
        value_vars=value_cols,
        var_name='Set',
        value_name='Time to Adapt'
    )

    # -------------------------------
    # Orders and labels
    # -------------------------------
    set_order = value_cols
    set_labels = ['Before Down', 'After Down', 'Before Up', 'After Up']

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

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # -------------------------------
    # Draw boxplots + datapoints
    # -------------------------------
    for gi, group in enumerate(group_order):
        for si, set_name in enumerate(set_order):

            data = df_long.loc[
                (df_long['Group ID'] == group) &
                (df_long['Set'] == set_name),
                'Time to Adapt'
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
    ax.set_ylabel("Time to Adapt (s)")
    ax.set_title(f"Time to Adapt ({mode.upper()})")
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
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=6
    )

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def plot_difference_time_to_adapt(results, mode='min', box_width=0.08, jitter=0.015):
    """
    Plot difference in time to adapt (After - Before) for Down and Up,
    with true spacing between groups, hatching for Slow groups,
    outlined datapoints, and a 1x6 legend.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with difference columns.

    mode : {'min', 'avg'}
        Which difference metric to plot.

    box_width : float
        Width of boxplots.

    jitter : float
        Horizontal jitter for datapoints.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    mode = mode.lower()

    # -------------------------------
    # Column mapping
    # -------------------------------
    column_map = {
        'min': {
            'Down': 'Difference min time to adapt before after down',
            'Up': 'Difference min time to adapt before after up'
        },
        'avg': {
            'Down': 'Difference average time to adapt before after down',
            'Up': 'Difference average time to adapt before after up'
        }
    }

    if mode not in column_map:
        raise ValueError("mode must be 'min' or 'avg'")

    # -------------------------------
    # Build long dataframe
    # -------------------------------
    df_long = []

    for direction, col in column_map[mode].items():
        tmp = results[['Group ID', col]].copy()
        tmp['Direction'] = direction
        tmp = tmp.rename(columns={col: 'Difference'})
        df_long.append(tmp)

    df_long = pd.concat(df_long, ignore_index=True)

    # -------------------------------
    # Orders and labels
    # -------------------------------
    directions = ['Down', 'Up']
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
    base_positions = np.arange(len(directions))
    group_offsets = np.linspace(-0.30, 0.30, len(group_order))

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # -------------------------------
    # Draw boxplots + datapoints
    # -------------------------------
    for gi, group in enumerate(group_order):
        for di, direction in enumerate(directions):

            data = df_long.loc[
                (df_long['Group ID'] == group) &
                (df_long['Direction'] == direction),
                'Difference'
            ].dropna()

            if data.empty:
                continue

            pos = base_positions[di] + group_offsets[gi]

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
    ax.set_xticklabels(directions)
    ax.set_ylabel("Δ Time to Adapt (After − Before) [s]")
    ax.set_title(f"Difference in Time to Adapt ({mode.upper()})")
    ax.set_xlabel("")

    # Zoom x-axis
    ax.set_xlim(
        base_positions[0] - 0.45,
        base_positions[-1] + 0.45
    )

    # Zero reference line (important for differences)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)

    # -------------------------------
    # Legend (1 row, 6 columns)
    # -------------------------------
    legend_handles = []
    legend_texts = []

    for g in group_order:
        p = Patch(
            facecolor=palette[g],
            edgecolor='black'
        )
        if g.endswith('_65'):
            p.set_hatch('////')

        legend_handles.append(p)
        legend_texts.append(legend_labels[g])

    ax.legend(
        handles=legend_handles,
        labels=legend_texts,
        title='Group',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
        frameon=True
    )

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def plot_difference_min_time_boxplot(df, box_width=0.15, group_spacing=0.25):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    box_width : float
        Width of each box
    group_spacing : float
        Spacing between boxplots within the same tick
    """

    # -------------------------------------------------
    # 1. Filter excluded rows
    # -------------------------------------------------
    df = df[df['Exclude'] == 0].copy()

    # -------------------------------------------------
    # 2. Long format
    # -------------------------------------------------
    df_long = pd.melt(
        df,
        id_vars=['Signal'],
        value_vars=[
            'Difference min time to adapt before after up',
            'Difference min time to adapt before after down'
        ],
        var_name='Direction',
        value_name='Difference'
    )

    df_long['Direction'] = df_long['Direction'].map({
        'Difference min time to adapt before after up': 'Up',
        'Difference min time to adapt before after down': 'Down'
    })

    # -------------------------------------------------
    # 3. Plot order & spacing
    # -------------------------------------------------
    direction_order = ['Up', 'Down']
    signal_order = ['Sine', 'Pink', 'White']

    base_positions = np.arange(len(direction_order))
    group_offsets = np.linspace(
        -group_spacing,
        group_spacing,
        len(signal_order)
    )

    # -------------------------------------------------
    # 4. COLOR CONTROL (EDIT ONLY THIS)
    # -------------------------------------------------
    signal_colors = {
        'Sine':  '#4F4F4F',   # blue
        'Pink':  '#FFC0CB',   # orange/pink
        'White': '#D3D3D3'    # green
    }

    # -------------------------------------------------
    # 5. Plot
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    for di, direction in enumerate(direction_order):
        for si, signal in enumerate(signal_order):

            data = df_long.loc[
                (df_long['Direction'] == direction) &
                (df_long['Signal'] == signal),
                'Difference'
            ].dropna()

            if data.empty:
                continue

            pos = base_positions[di] + group_offsets[si]

            bp = ax.boxplot(
                data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )

            # Apply color
            for patch in bp['boxes']:
                patch.set_facecolor(signal_colors[signal])
                patch.set_edgecolor('black')
                patch.set_alpha(0.85)

            for whisker in bp['whiskers']:
                whisker.set_color('black')

            for cap in bp['caps']:
                cap.set_color('black')

            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

    # -------------------------------------------------
    # 6. Axis formatting
    # -------------------------------------------------
    ax.set_xticks(base_positions)
    ax.set_xticklabels(direction_order)
    ax.set_xlabel('')
    ax.set_ylabel('Time to Adapt (Before − After)')
    ax.set_title('Difference in Minimum Time to Adapt\nBefore vs After')
    ax.set_ylim(-2.2, 1.5)
    # ax.set_xlim(0.45, 0.45)
    ax.grid(axis='y', alpha=0.3)

    # -------------------------------------------------
    # 7. Custom legend
    # -------------------------------------------------
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=signal)
        for signal, color in signal_colors.items()
    ]

    ax.legend(
        handles=legend_handles,
        title='Group',
        frameon=True,
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=3
    )


    plt.tight_layout()
    plt.show()



directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results\Perturbation results'
os.chdir(directory)
results_sd = pd.read_excel('Sd Method Perturbation_results_3_sd_after_max_threshold.xlsx')
results_asymp = pd.read_excel('Asymptote Method Perturbation results 0.95.xlsx')
print(results_sd.columns)
print(results_asymp.columns)

# plot_time_to_adapt_seaborn(results_sd, mode='min')
# plot_time_to_adapt_seaborn(results_sd, mode='avg')

# plot_difference_time_to_adapt(results_sd, mode='min')
# plot_difference_time_to_adapt(results_sd, mode='avg')

# plot_time_to_adapt_seaborn(results_asymp, mode='min')
# plot_time_to_adapt_seaborn(results_asymp, mode='avg')
#
# plot_difference_time_to_adapt(results_asymp, mode='min')
# plot_difference_time_to_adapt(results_asymp, mode='avg')

plot_difference_min_time_boxplot(results_sd)

