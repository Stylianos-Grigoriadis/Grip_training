import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from Lib_grip import spatial_error
import glob
import lib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_error_boxplots(df, error_type='spatial', group_by='all'):
    """
    Creates grouped box plots of performance error metrics across experimental sets.

    The function visualizes either Spatial Error or Variable Error across multiple
    sets (e.g., trials 1–10) using box plots. The x-axis represents the set number,
    while the y-axis represents the selected error metric. Data are grouped
    according to the experimental conditions specified by the `group_by` argument.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing columns for Signal, Speed, and error values
        stored in a wide format (one column per set).

    error_type : {'spatial', 'variable'}, optional
        Specifies the dependent variable to be plotted.
        - 'spatial'  : Mean Spatial Error
        - 'variable' : Variable Error

    group_by : {'all', 'speed', 'signal'}, optional
        Defines how box plots are grouped:
        - 'all'    : Six groups (Signal × Speed combinations)
        - 'speed'  : Two groups (Fast vs Slow)
        - 'signal' : Three groups (Pink, Sine, White)

    Returns
    -------
    None
        Displays the box plot figure.
    """

    error_type = error_type.lower()
    group_by = group_by.lower()

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
    sns.boxplot(
        data=df_long,
        x='Set',
        y=y_label,
        hue='Group',
        hue_order=hue_order,
        order=sorted(df_long['Set'].unique()),
        width=0.8,
        fliersize=2
    )

    plt.xlabel('Set')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title=group_by.capitalize(), bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import plotly.express as px

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



Stylianos = True
if Stylianos:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results'
else:
    directory = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Results'
os.chdir(directory)
results = pd.read_excel('Training_trials_results.xlsx')
print(results)



# plot_error_boxplots(results, error_type='spatial', group_by='all')
# plot_error_boxplots(results, error_type='spatial', group_by='speed')
# plot_error_boxplots(results, error_type='spatial', group_by='signal')
# plot_error_boxplots(results, error_type='variable', group_by='all')
# plot_error_boxplots(results, error_type='variable', group_by='speed')
# plot_error_boxplots(results, error_type='variable', group_by='signal')

plot_error_means_plotly(results, error_type='spatial', group_by='all')
plot_error_means_plotly(results, error_type='spatial', group_by='signal')
plot_error_means_plotly(results, error_type='spatial', group_by='speed')