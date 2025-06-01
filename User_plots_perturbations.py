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

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Results'
os.chdir(directory)
results = pd.read_excel('Perturbation_results_2_sd.xlsx')

# Plot of Minimum adaptation times
df_long = results.melt(id_vars=['Group ID'],
                  value_vars=['Min Time to adapt before down', 'Min Time to adapt after down',
                              'Min Time to adapt before up', 'Min Time to adapt after up'],
                  var_name='Set', value_name='Min Time to adapt')  # Ensure correct name



# Rename set names for readability

df_long['ID'] = pd.Categorical(df_long['Group ID'], categories=["Sine_100", "Pink_100", "White_100", "Sine_65", "Pink_65", "White_65"], ordered=True)

# Custom color palette
custom_palette = {
    "Sine_100": "#4F4F4F",    # Dark gray (first)
    "Pink_100": "#FFC0CB",      # Soft pink (second)
    "White_100": "#D3D3D3",      # Light gray (third)
    "Sine_65": "#4F4F4F",  # Dark gray (first)
    "Pink_65": "#FFC0CB",  # Soft pink (second)
    "White_65": "#D3D3D3",  # Light gray (third)
    }

# Create the boxplot with the correct order
plt.figure(figsize=(12, 6))
bar = sns.boxplot(x='Set', y='Min Time to adapt', hue='Group ID', data=df_long, palette=custom_palette, showfliers=True)

hatches = ['', '', '', '',
           '////', '////', '////', '////',
           '', '', '', '',
           '////', '////', '////', '////',
           '', '', '', '',
           '////', '////', '////', '////',
           '', '////', '', '////', '', '////',
           ]

for i, thisbar in enumerate(bar.patches):
    print(i)
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])


# Customize plot
custom_labels = ['Before Down', 'After Down', 'Before Up', 'After Up']
plt.xticks(ticks=[0, 1, 2, 3], labels=custom_labels)
plt.title('Time to Adapt')
plt.ylabel('Minimum Time to Adapt')
plt.xlabel('')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Plot of Average adaptation times
df_long = results.melt(id_vars=['Group ID'],
                  value_vars=['Average Time to adapt before down', 'Average Time to adapt after down',
                              'Average Time to adapt before up', 'Average Time to adapt after up'],
                  var_name='Set', value_name='Average Time to adapt')  # Ensure correct name



# Rename set names for readability
df_long['Set'] = df_long['Set'].str.replace('Average Spatial error set ', 'Set ')

df_long['ID'] = pd.Categorical(df_long['Group ID'], categories=["Sine_100", "Pink_100", "White_100", "Sine_65", "Pink_65", "White_65"], ordered=True)

# Custom color palette
custom_palette = {
    "Sine_100": "#4F4F4F",    # Dark gray (first)
    "Pink_100": "#FFC0CB",      # Soft pink (second)
    "White_100": "#D3D3D3",      # Light gray (third)
    "Sine_65": "#4F4F4F",  # Dark gray (first)
    "Pink_65": "#FFC0CB",  # Soft pink (second)
    "White_65": "#D3D3D3",  # Light gray (third)
    }

# Create the boxplot with the correct order
plt.figure(figsize=(12, 6))
bar = sns.boxplot(x='Set', y='Average Time to adapt', hue='Group ID', data=df_long, palette=custom_palette, showfliers=True)

hatches = ['', '', '', '',
           '////', '////', '////', '////',
           '', '', '', '',
           '////', '////', '////', '////',
           '', '', '', '',
           '////', '////', '////', '////',
           '', '////', '', '////', '', '////',
           ]

for i, thisbar in enumerate(bar.patches):
    print(i)
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])


# Customize plot
custom_labels = ['Before Down', 'After Down', 'Before Up', 'After Up']
plt.xticks(ticks=[0, 1, 2, 3], labels=custom_labels)
plt.title('Time to Adapt')
plt.ylabel('Average Time to Adapt')
plt.xlabel('')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()