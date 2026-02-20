import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16


def plot_error_boxplots(
    df,
    error_type='spatial',
    group_by='all',
    plot_type='box',
    show_points=True
):
    """
    Creates grouped box or violin plots with jittered raw data points.
    """

    error_type = error_type.lower()
    group_by = group_by.lower()
    plot_type = plot_type.lower()

    # -------------------------------
    # Select dependent variable
    # -------------------------------
    if error_type == 'spatial':
        cols = [c for c in df.columns if 'Mean Spatial error' in c]
        y_label = 'Spatial Error'
        title_base = 'Spatial Error'

    elif error_type == 'variable':
        cols = [c for c in df.columns if 'Variable Error trial' in c]
        y_label = 'Variable Error'
        title_base = 'Variable Error'

    else:
        raise ValueError("error_type must be 'spatial' or 'variable'")

    # -------------------------------
    # Melt to long format
    # -------------------------------
    df_long = df.melt(
        id_vars=['Signal', 'Speed'],
        value_vars=cols,
        var_name='Set',
        value_name=y_label
    )

    df_long['Set'] = (
        df_long['Set']
        .str.extract(r'(\d+)')
        .astype(int)
    )

    # -------------------------------
    # Define grouping logic
    # -------------------------------
    if group_by == 'all':
        df_long['Group'] = df_long['Signal'] + '_' + df_long['Speed']
        hue_order = [
            'Pink_Fast', 'Pink_Slow',
            'Sine_Fast', 'Sine_Slow',
            'White_Fast', 'White_Slow'
        ]
        title = f'{title_base} across Sets (All Groups)'

    elif group_by == 'speed':
        df_long['Group'] = df_long['Speed']
        hue_order = ['Fast', 'Slow']
        title = f'{title_base} across Sets (Speed)'

    elif group_by == 'signal':
        df_long['Group'] = df_long['Signal']
        hue_order = ['Pink', 'Sine', 'White']
        title = f'{title_base} across Sets (Signal)'

    else:
        raise ValueError("group_by must be 'all', 'speed', or 'signal'")

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(16, 6))

    if plot_type == 'box':
        sns.boxplot(
            data=df_long,
            x='Set',
            y=y_label,
            hue='Group',
            hue_order=hue_order,
            order=sorted(df_long['Set'].unique()),
            width=0.8,
            fliersize=0  # hide default outliers (we plot raw points instead)
        )

    elif plot_type == 'violin':
        sns.violinplot(
            data=df_long,
            x='Set',
            y=y_label,
            hue='Group',
            hue_order=hue_order,
            order=sorted(df_long['Set'].unique()),
            cut=0,
            scale='width',
            inner='quartile',
            linewidth=1,
            inner_kws={'linewidth': 3}
        )

    else:
        raise ValueError("plot_type must be 'box' or 'violin'")

    # -------------------------------
    # Overlay jittered datapoints
    # -------------------------------
    if show_points:
        sns.stripplot(
            data=df_long,
            x='Set',
            y=y_label,
            hue='Group',
            hue_order=hue_order,
            order=sorted(df_long['Set'].unique()),
            dodge=True,
            jitter=0.2,
            alpha=0.6,
            size=4,
            edgecolor='black',
            linewidth=0.8
        )

    # -------------------------------
    # Legend handling (avoid duplicates)
    # -------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    n_groups = len(hue_order)
    plt.legend(
        handles[:n_groups],
        labels[:n_groups],
        title=group_by.capitalize(),
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    plt.xlabel('Set')
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_error_boxplots_only_slow(
    df,
    error_type='spatial',
    plot_type='box',
    show_points=True
):
    """
    Creates grouped box or violin plots with jittered raw data points.
    Excludes rows where Exclude == 1 and plots only Signal (Pink, Sine, White),
    collapsing across Speed.
    """

    error_type = error_type.lower()
    plot_type = plot_type.lower()

    # -------------------------------
    # Exclude participants
    # -------------------------------
    df = df[df['Exclude'] == 0].copy()

    # -------------------------------
    # Select dependent variable
    # -------------------------------
    if error_type == 'spatial':
        cols = [c for c in df.columns if 'Mean Spatial error' in c]
        y_label = 'Spatial Error'
        title_base = 'Spatial Error'

    elif error_type == 'variable':
        cols = [c for c in df.columns if 'Variable Error trial' in c]
        y_label = 'Variable Error'
        title_base = 'Variable Error'

    else:
        raise ValueError("error_type must be 'spatial' or 'variable'")

    # -------------------------------
    # Melt to long format
    # -------------------------------
    df_long = df.melt(
        id_vars=['Signal'],
        value_vars=cols,
        var_name='Set',
        value_name=y_label
    )

    df_long['Set'] = (
        df_long['Set']
        .str.extract(r'(\d+)')
        .astype(int)
    )

    # -------------------------------
    # Keep only desired signals
    # -------------------------------
    signal_order = ['Pink', 'Sine', 'White']
    df_long = df_long[df_long['Signal'].isin(signal_order)]

    title = f'{title_base} across Sets (Signal)'

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(16, 6))

    if plot_type == 'box':
        sns.boxplot(
            data=df_long,
            x='Set',
            y=y_label,
            hue='Signal',
            hue_order=signal_order,
            order=sorted(df_long['Set'].unique()),
            width=0.8,
            fliersize=0
        )

    elif plot_type == 'violin':
        sns.violinplot(
            data=df_long,
            x='Set',
            y=y_label,
            hue='Signal',
            hue_order=signal_order,
            order=sorted(df_long['Set'].unique()),
            cut=0,
            scale='width',
            inner='quartile',
            linewidth=1,
            inner_kws={'linewidth': 3}
        )

    else:
        raise ValueError("plot_type must be 'box' or 'violin'")

    # -------------------------------
    # Overlay jittered datapoints
    # -------------------------------
    if show_points:
        sns.stripplot(
            data=df_long,
            x='Set',
            y=y_label,
            hue='Signal',
            hue_order=signal_order,
            order=sorted(df_long['Set'].unique()),
            dodge=True,
            jitter=0.2,
            alpha=0.6,
            size=4,
            edgecolor='black',
            linewidth=0.8
        )

    # -------------------------------
    # Legend handling (avoid duplicates)
    # -------------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    n_groups = len(signal_order)

    plt.legend(
        handles[:n_groups],
        labels[:n_groups],
        title='Signal',
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    plt.xlabel('Set')
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_means_plotly(df, error_type='spatial', group_by='all', dodge=0.12):
    """
    Interactive Plotly line plot showing mean ± SD across sets,
    with horizontal dodging to avoid overlapping error bars.
    """

    error_type = error_type.lower()
    group_by = group_by.lower()

    # -----------------------------
    # Select error columns
    # -----------------------------
    if error_type == 'spatial':
        cols = [c for c in df.columns if 'Mean Spatial error' in c]
        y_label = 'Spatial Error'
        title_base = 'Mean Spatial Error'
    elif error_type == 'variable':
        cols = [c for c in df.columns if 'Variable Error trial' in c]
        y_label = 'Variable Error'
        title_base = 'Mean Variable Error'
    else:
        raise ValueError("error_type must be 'spatial' or 'variable'")

    # -----------------------------
    # Reshape
    # -----------------------------
    df_long = df.melt(
        id_vars=['Signal', 'Speed'],
        value_vars=cols,
        var_name='Set',
        value_name=y_label
    )

    df_long['Set'] = df_long['Set'].str.extract(r'(\d+)').astype(int)

    # -----------------------------
    # Grouping
    # -----------------------------
    if group_by == 'all':
        df_long['Group'] = df_long['Signal'] + '_' + df_long['Speed']
        title = f'{title_base} across Sets (All Groups)'
    elif group_by == 'signal':
        df_long['Group'] = df_long['Signal']
        title = f'{title_base} across Sets (Signal)'
    elif group_by == 'speed':
        df_long['Group'] = df_long['Speed']
        title = f'{title_base} across Sets (Speed)'
    else:
        raise ValueError("group_by must be 'all', 'signal', or 'speed'")

    # -----------------------------
    # Mean & SD
    # -----------------------------
    df_stats = (
        df_long
        .groupby(['Set', 'Group'], as_index=False)
        .agg(
            mean_error=(y_label, 'mean'),
            sd_error=(y_label, 'std')
        )
    )

    # -----------------------------
    # Create deterministic x-offsets
    # -----------------------------
    groups = df_stats['Group'].unique()
    offsets = np.linspace(
        -dodge, dodge, len(groups)
    )

    offset_map = dict(zip(groups, offsets))
    df_stats['Set_jittered'] = df_stats['Set'] + df_stats['Group'].map(offset_map)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = px.line(
        df_stats,
        x='Set_jittered',
        y='mean_error',
        color='Group',
        error_y='sd_error',
        markers=True,
        title=title
    )

    # Fix x-axis ticks to show original Set numbers
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=sorted(df_stats['Set'].unique()),
            ticktext=[str(i) for i in sorted(df_stats['Set'].unique())],
            title='Set'
        ),
        yaxis_title=f'{y_label} (mean ± SD)',
        template='plotly_white'
    )

    fig.show()

def plot_error_mean_sd(df, error_type='spatial'):
    """
    Plots mean ± 1 SD of error across sets for each Signal (Pink, Sine, White),
    excluding rows where Exclude == 1 and collapsing across Speed.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    error_type = error_type.lower()

    # -------------------------------
    # Exclude participants
    # -------------------------------
    df = df[df['Exclude'] == 0].copy()

    # -------------------------------
    # Select dependent variable
    # -------------------------------
    if error_type == 'spatial':
        cols = [c for c in df.columns if 'Mean Spatial error' in c]
        y_label = 'Spatial Error'
        title = 'Spatial Error across Sets'

    elif error_type == 'variable':
        cols = [c for c in df.columns if 'Variable Error trial' in c]
        y_label = 'Variable Error'
        title = 'Variable Error across Sets'

    else:
        raise ValueError("error_type must be 'spatial' or 'variable'")

    # -------------------------------
    # Melt to long format
    # -------------------------------
    df_long = df.melt(
        id_vars=['Signal'],
        value_vars=cols,
        var_name='Set',
        value_name=y_label
    )

    df_long['Set'] = (
        df_long['Set']
        .str.extract(r'(\d+)')
        .astype(int)
    )

    # -------------------------------
    # Keep only desired signals
    # -------------------------------
    signal_order = ['Pink', 'Sine', 'White']
    df_long = df_long[df_long['Signal'].isin(signal_order)]

    # -------------------------------
    # Compute mean & SD
    # -------------------------------
    summary = (
        df_long
        .groupby(['Signal', 'Set'])[y_label]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # -------------------------------
    # Color control (edit here)
    # -------------------------------
    signal_colors = {
        'Pink':  '#E75480',
        'Sine':  '#4F4F4F',
        'White': '#BFBFBF'
    }

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(14, 6))

    for signal in signal_order:
        sub = summary[summary['Signal'] == signal]
        x = sub['Set']
        y = sub['mean']
        sd = sub['std']

        plt.plot(
            x,
            y,
            lw=3,
            color=signal_colors[signal],
            label=signal
        )

        plt.fill_between(
            x,
            y - sd,
            y + sd,
            color=signal_colors[signal],
            alpha=0.25
        )

    plt.xlabel('Set')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title='Signal')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_error_mean_sem(df, error_type='spatial'):
    """
    Plots mean ± SEM of error across sets for each Signal (Pink, Sine, White),
    excluding rows where Exclude == 1 and collapsing across Speed.

    SEM is shown as vertical error bars (not shaded areas).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    error_type = error_type.lower()

    # -------------------------------
    # Exclude participants
    # -------------------------------
    df = df[df['Exclude'] == 0].copy()

    # -------------------------------
    # Select dependent variable
    # -------------------------------
    if error_type == 'spatial':
        cols = [c for c in df.columns if 'Mean Spatial error' in c]
        y_label = 'Spatial Error'
        title = 'Spatial Error across Sets (Mean ± SEM)'

    elif error_type == 'variable':
        cols = [c for c in df.columns if 'Variable Error trial' in c]
        y_label = 'Variable Error'
        title = 'Variable Error across Sets (Mean ± SEM)'

    else:
        raise ValueError("error_type must be 'spatial' or 'variable'")

    # -------------------------------
    # Melt to long format
    # -------------------------------
    df_long = df.melt(
        id_vars=['Signal'],
        value_vars=cols,
        var_name='Set',
        value_name=y_label
    )

    df_long['Set'] = (
        df_long['Set']
        .str.extract(r'(\d+)')
        .astype(int)
    )

    # -------------------------------
    # Keep only desired signals
    # -------------------------------
    signal_order = ['Pink', 'Sine', 'White']
    df_long = df_long[df_long['Signal'].isin(signal_order)]

    # -------------------------------
    # Compute mean, SD, N, SEM
    # -------------------------------
    summary = (
        df_long
        .groupby(['Signal', 'Set'])[y_label]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    summary['sem'] = summary['std'] / np.sqrt(summary['count'])

    # -------------------------------
    # COLOR CONTROL (EDIT / IMPORT THIS)
    # -------------------------------
    signal_colors = {
        'Pink':  '#E75480',
        'Sine':  '#4F4F4F',
        'White': '#BFBFBF'
    }

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(14, 6))

    for signal in signal_order:
        sub = summary[summary['Signal'] == signal]

        x = sub['Set']
        y = sub['mean']
        sem = sub['sem']

        plt.errorbar(
            x,
            y,
            yerr=sem,
            fmt='-o',
            lw=3,
            capsize=4,
            markersize=5,
            color=signal_colors[signal],
            label=signal
        )

    # -------------------------------
    # Axis formatting
    # -------------------------------
    plt.xlabel('Set')
    plt.ylabel(y_label)
    plt.title(title)

    plt.xticks(np.arange(1, 11))  # Force ticks 1–10
    plt.grid(axis='y', alpha=0.3)

    plt.legend(title='Signal')
    plt.tight_layout()
    plt.show()

def plot_error_mean_sd_with_jitter_and_points(
    df,
    error_type='spatial',
    line_jitter=0.18,
    point_jitter=0.04,
    show_points=True
):
    """
    Plots mean ± SD of error across sets for each Signal
    (Pink, Sine, White), excluding rows where Exclude == 1
    and collapsing across Speed.

    - Mean lines use larger fixed x-jitter
    - Raw datapoints use smaller random x-jitter
    - Data are converted from kg to Newtons
    """

    error_type = error_type.lower()

    # -------------------------------
    # Exclude participants
    # -------------------------------
    df = df[df['Exclude'] == 0].copy()

    # -------------------------------
    # Select dependent variable
    # -------------------------------
    if error_type == 'spatial':
        cols = [c for c in df.columns if 'Mean Spatial error' in c]
        y_label = 'Spatial Error'
    elif error_type == 'variable':
        cols = [c for c in df.columns if 'Variable Error trial' in c]
        y_label = 'Variable Error'
    else:
        raise ValueError("error_type must be 'spatial' or 'variable'")

    # -------------------------------
    # Melt to long format
    # -------------------------------
    df_long = df.melt(
        id_vars=['Signal'],
        value_vars=cols,
        var_name='Set',
        value_name=y_label
    )

    df_long['Set'] = (
        df_long['Set']
        .str.extract(r'(\d+)')
        .astype(int)
    )

    # -------------------------------
    # Convert kg to Newtons
    # -------------------------------
    df_long[y_label] = df_long[y_label] * 9.81

    # -------------------------------
    # Keep only desired signals
    # -------------------------------
    signal_order = ['Sine', 'Pink', 'White']
    df_long = df_long[df_long['Signal'].isin(signal_order)]

    # -------------------------------
    # Summary statistics (mean, SD)
    # -------------------------------
    summary = (
        df_long
        .groupby(['Signal', 'Set'])[y_label]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # -------------------------------
    # COLOR CONTROL
    # -------------------------------
    signal_colors = {
        'Sine':  '#4F4F4F',
        'Pink': '#E75480',
        'White': '#BFBFBF'
    }

    # -------------------------------
    # Jitter offsets for mean lines
    # -------------------------------
    line_offsets = {
        'Sine':  -line_jitter,
        'Pink':   0.0,
        'White':  line_jitter
    }

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(14, 6))

    # ---- Mean ± SD lines ----
    for signal in signal_order:
        sub = summary[summary['Signal'] == signal]

        x = sub['Set'].to_numpy() + line_offsets[signal]
        y = sub['mean']
        sd = sub['std']

        plt.errorbar(
            x,
            y,
            yerr=sd,
            fmt='-o',
            lw=3,
            capsize=4,
            markersize=5,
            color=signal_colors[signal],
            label={
                'Sine': 'Non-variable',
                'Pink': 'Structured',
                'White': 'Non-structured'
            }[signal],
            zorder=3
        )

    # ---- Raw datapoints ----
    if show_points:
        for signal in signal_order:
            sub = df_long[df_long['Signal'] == signal]

            x = (
                sub['Set'].to_numpy()
                + line_offsets[signal]
                + np.random.uniform(-point_jitter, point_jitter, size=len(sub))
            )

            plt.scatter(
                x,
                sub[y_label],
                s=18,
                alpha=0.5,
                color=signal_colors[signal],
                edgecolor='black',
                linewidth=0.4,
                zorder=2
            )

    # -------------------------------
    # Axis formatting
    # -------------------------------
    plt.xlabel('Training Set')
    plt.ylabel(f'{y_label} (N)')

    if error_type == 'spatial':
        plt.title('Average Spatial Error across Training Sets')
    elif error_type == 'variable':
        plt.title('Average Variable Error across Training Sets')

    plt.ylim(0.3 * 9.81, 2.6 * 9.81)
    plt.xticks(np.arange(1, 11))
    plt.grid(axis='y', alpha=0.3)

    # -------------------------------
    # Legend (centered below plot)
    # -------------------------------
    plt.legend(
        title='Group',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True
    )

    plt.tight_layout()
    plt.show()




Stylianos = True
if Stylianos:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results'
else:
    directory = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Results'
os.chdir(directory)
results = pd.read_excel('Training_trials_results.xlsx')
print(results.columns)




# plot_error_boxplots(results, error_type='spatial', group_by='all', plot_type='box')
# plot_error_boxplots(results, error_type='spatial', group_by='speed', plot_type='box')
# plot_error_boxplots(results, error_type='spatial', group_by='signal', plot_type='box')
# plot_error_boxplots(results, error_type='variable', group_by='all', plot_type='box')
# plot_error_boxplots(results, error_type='variable', group_by='speed', plot_type='box')
# plot_error_boxplots(results, error_type='variable', group_by='signal', plot_type='box')

# plot_error_means_plotly(results, error_type='spatial', group_by='all')
# plot_error_means_plotly(results, error_type='spatial', group_by='signal')
# plot_error_means_plotly(results, error_type='spatial', group_by='speed')

# plot_error_boxplots_only_slow(results, error_type='spatial', plot_type='box')
# plot_error_boxplots_only_slow(results, error_type='variable', plot_type='box')

# plot_error_mean_sd(results, error_type='spatial')
# plot_error_mean_sd(results, error_type='variable')

# plot_error_mean_sem(results, error_type='spatial')
# plot_error_mean_sem(results, error_type='variable')

plot_error_mean_sd_with_jitter_and_points(results, error_type='spatial', show_points=False)
plot_error_mean_sd_with_jitter_and_points(results, error_type='variable', show_points=False)