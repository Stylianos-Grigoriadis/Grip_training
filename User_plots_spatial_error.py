import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
import glob
import lib
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

stylianos = True
if stylianos == True:
    directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Results'
else:
    directory_path = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Pilot study 4\Results'

os.chdir(directory_path)
results = pd.read_excel('Mean_spatial_error_results_with_15_low_pass_filter.xlsx')


################### Plot for Set*Signal ###################
df_long = results.melt(id_vars=['Signal ID'],
                  value_vars=[  'Mean Spatial error trail 1',
                                'Mean Spatial error trail 2',
                                'Mean Spatial error trail 3',
                                'Mean Spatial error trail 4',
                                'Mean Spatial error trail 5',
                                'Mean Spatial error trail 6',
                                'Mean Spatial error trail 7',
                                'Mean Spatial error trail 8',
                                'Mean Spatial error trail 9',
                                'Mean Spatial error trail 10'],
                  var_name='Set', value_name='Spatial Error')  # Ensure correct name

df_long['ID'] = pd.Categorical(df_long['Signal ID'], categories=["Sine", "Pink", "White"], ordered=True)

custom_palette = {
    "Sine": "#4F4F4F",    # Dark gray (first)
    "Pink": "#FFC0CB",      # Soft pink (second)
    "White": "#D3D3D3",      # Light gray (third)
}

plt.figure(figsize=(12, 6))
bar = sns.boxplot(x='Set', y='Spatial Error', hue='Signal ID', data=df_long, palette=custom_palette, showfliers=False)




# Customize plot
custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6', 'Set 7', 'Set 8', 'Set 9', 'Set 10']
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels=custom_labels)
plt.ylim(0.2, 5)
plt.title('Spatial Error Per Group for each Set')
plt.ylabel('Spatial Error')
plt.xlabel('')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

################### Plot for Set*Speed ###################
df_long = results.melt(id_vars=['Speed ID'],
                  value_vars=[  'Mean Spatial error trail 1',
                                'Mean Spatial error trail 2',
                                'Mean Spatial error trail 3',
                                'Mean Spatial error trail 4',
                                'Mean Spatial error trail 5',
                                'Mean Spatial error trail 6',
                                'Mean Spatial error trail 7',
                                'Mean Spatial error trail 8',
                                'Mean Spatial error trail 9',
                                'Mean Spatial error trail 10'],
                  var_name='Set', value_name='Spatial Error')  # Ensure correct name

df_long['ID'] = pd.Categorical(df_long['Speed ID'], categories=["Slow", "Fast"], ordered=True)

custom_palette = {
    "Slow": "#D3D3D3",      # Light gray(first)
    "Fast": "#4F4F4F",      # Dark gray (second)
}

plt.figure(figsize=(12, 6))
bar = sns.boxplot(x='Set', y='Spatial Error', hue='Speed ID', data=df_long, palette=custom_palette, showfliers=False)




# Customize plot
custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6', 'Set 7', 'Set 8', 'Set 9', 'Set 10']
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels=custom_labels)
plt.ylim(0.2, 5)
plt.title('Spatial Error Per Speed for each Set')
plt.ylabel('Spatial Error')
plt.xlabel('')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

################### Plot for Signal*Speed ###################
df_long = results.melt(id_vars=['Group ID'],
                  value_vars=['Mean Spatial error for each set'],
                  var_name='Set', value_name='Spatial Error')  # Ensure correct name

df_long['ID'] = pd.Categorical(df_long['Group ID'], categories=["Sine_100", "Sine_65", "Pink_100", "Pink_65", "White_100", "White_65"], ordered=True)

custom_palette = {
    "Sine_100": "#4F4F4F",    # Dark gray (first)
    "Pink_100": "#FFC0CB",      # Soft pink (second)
    "White_100": "#D3D3D3",      # Light gray (third)
    "Sine_65": "#4F4F4F",  # Dark gray (first)
    "Pink_65": "#FFC0CB",  # Soft pink (second)
    "White_65": "#D3D3D3",  # Light gray (third)
    }



plt.figure(figsize=(12, 6))
bar = sns.boxplot(x='Set', y='Spatial Error', hue='ID', data=df_long, palette=custom_palette, showfliers=False)
print(bar.patches)
hatches = ['', '////', '', '////', '', '////',
           '', '////', '', '////', '', '////',
           ]

for i, thisbar in enumerate(bar.patches):
    print(i)
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])

# Customize legend
handles, _ = bar.get_legend_handles_labels()
custom_labels = ['Sine Fast', 'Sine Slow', 'Pink Fast', 'Pink Slow', 'White Fast', 'White Slow']
plt.legend(handles=handles, labels=custom_labels, title='Group', loc='upper right')

# Customize plot
custom_labels = ['Average Spatial Error Across All Sets']
plt.xticks(ticks=[0], labels=custom_labels)
plt.ylim(0.2, 5)
plt.title('Average Spatial Error For each group across all Sets')
plt.ylabel('Spatial Error')
plt.xlabel('')

# Show plot
plt.tight_layout()
plt.show()