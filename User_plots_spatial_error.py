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
# df_long = results.melt(id_vars=['Signal ID'],
#                   value_vars=[  'Mean Spatial error trail 1',
#                                 'Mean Spatial error trail 2',
#                                 'Mean Spatial error trail 3',
#                                 'Mean Spatial error trail 4',
#                                 'Mean Spatial error trail 5',
#                                 'Mean Spatial error trail 6',
#                                 'Mean Spatial error trail 7',
#                                 'Mean Spatial error trail 8',
#                                 'Mean Spatial error trail 9',
#                                 'Mean Spatial error trail 10'],
#                   var_name='Set', value_name='Spatial Error')  # Ensure correct name
#
# df_long['ID'] = pd.Categorical(df_long['Signal ID'], categories=["Sine", "Pink", "White"], ordered=True)
#
# custom_palette = {
#     "Sine": "#4F4F4F",    # Dark gray (first)
#     "Pink": "#FFC0CB",      # Soft pink (second)
#     "White": "#D3D3D3",      # Light gray (third)
# }
#
# plt.figure(figsize=(12, 6))
# bar = sns.boxplot(x='Set', y='Spatial Error', hue='Signal ID', data=df_long, palette=custom_palette, showfliers=False)
#
#
#
#
# # Customize plot
# custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6', 'Set 7', 'Set 8', 'Set 9', 'Set 10']
# plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels=custom_labels)
# plt.ylim(0.2, 5)
# plt.title('Spatial Error Per Group for each Set')
# plt.ylabel('Spatial Error')
# plt.xlabel('')
# plt.legend()
#
# # Show plot
# plt.tight_layout()
# plt.show()
#
# ################### Plot for Set*Speed ###################
# df_long = results.melt(id_vars=['Speed ID'],
#                   value_vars=[  'Mean Spatial error trail 1',
#                                 'Mean Spatial error trail 2',
#                                 'Mean Spatial error trail 3',
#                                 'Mean Spatial error trail 4',
#                                 'Mean Spatial error trail 5',
#                                 'Mean Spatial error trail 6',
#                                 'Mean Spatial error trail 7',
#                                 'Mean Spatial error trail 8',
#                                 'Mean Spatial error trail 9',
#                                 'Mean Spatial error trail 10'],
#                   var_name='Set', value_name='Spatial Error')  # Ensure correct name
#
# df_long['ID'] = pd.Categorical(df_long['Speed ID'], categories=["Slow", "Fast"], ordered=True)
#
# custom_palette = {
#     "Slow": "#D3D3D3",      # Light gray(first)
#     "Fast": "#4F4F4F",      # Dark gray (second)
# }
#
# plt.figure(figsize=(12, 6))
# bar = sns.boxplot(x='Set', y='Spatial Error', hue='Speed ID', data=df_long, palette=custom_palette, showfliers=False)
#
#
#
#
# # Customize plot
# custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6', 'Set 7', 'Set 8', 'Set 9', 'Set 10']
# plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels=custom_labels)
# plt.ylim(0.2, 5)
# plt.title('Spatial Error Per Speed for each Set')
# plt.ylabel('Spatial Error')
# plt.xlabel('')
# plt.legend()
#
# # Show plot
# plt.tight_layout()
# plt.show()
#
# ################### Plot for Signal*Speed ###################
# df_long = results.melt(id_vars=['Group ID'],
#                   value_vars=['Mean Spatial error for each set'],
#                   var_name='Set', value_name='Spatial Error')  # Ensure correct name
#
# df_long['ID'] = pd.Categorical(df_long['Group ID'], categories=["Sine_100", "Sine_65", "Pink_100", "Pink_65", "White_100", "White_65"], ordered=True)
#
# custom_palette = {
#     "Sine_100": "#4F4F4F",    # Dark gray (first)
#     "Pink_100": "#FFC0CB",      # Soft pink (second)
#     "White_100": "#D3D3D3",      # Light gray (third)
#     "Sine_65": "#4F4F4F",  # Dark gray (first)
#     "Pink_65": "#FFC0CB",  # Soft pink (second)
#     "White_65": "#D3D3D3",  # Light gray (third)
#     }
#
#
#
# plt.figure(figsize=(12, 6))
# bar = sns.boxplot(x='Set', y='Spatial Error', hue='ID', data=df_long, palette=custom_palette, showfliers=False)
# print(bar.patches)
# hatches = ['', '////', '', '////', '', '////',
#            '', '////', '', '////', '', '////',
#            ]
#
# for i, thisbar in enumerate(bar.patches):
#     print(i)
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
#
# # Customize legend
# handles, _ = bar.get_legend_handles_labels()
# custom_labels = ['Sine Fast', 'Sine Slow', 'Pink Fast', 'Pink Slow', 'White Fast', 'White Slow']
# plt.legend(handles=handles, labels=custom_labels, title='Group', loc='upper right')
#
# # Customize plot
# custom_labels = ['Average Spatial Error Across All Sets']
# plt.xticks(ticks=[0], labels=custom_labels)
# plt.ylim(0.2, 5)
# plt.title('Average Spatial Error For each group across all Sets')
# plt.ylabel('Spatial Error')
# plt.xlabel('')
#
# # Show plot
# plt.tight_layout()
# plt.show()


print(results.columns)
SE_Sine_average_slow = []
SE_Pink_average_slow = []
SE_White_average_slow = []
SE_Sine_average_fast = []
SE_Pink_average_fast = []
SE_White_average_fast = []
SE_Sine_average = []
SE_Pink_average = []
SE_White_average = []
SE_Sine_sd_slow = []
SE_Pink_sd_slow = []
SE_White_sd_slow = []
SE_Sine_sd_fast = []
SE_Pink_sd_fast = []
SE_White_sd_fast = []
SE_Sine_sd = []
SE_Pink_sd = []
SE_White_sd = []

SE_slow_average = []
SE_fast_average = []
SE_slow_sd = []
SE_fast_sd = []

for i in range(1,11):
    
    SE_Sine_slow = results.loc[(results['Signal ID'] == 'Sine') & (results['Speed ID'] == 'Slow') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_Sine_fast = results.loc[(results['Signal ID'] == 'Sine') & (results['Speed ID'] == 'Fast') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_Sine = results.loc[(results['Signal ID'] == 'Sine'), f'Mean Spatial error trail {i}'].to_numpy()
    SE_Sine_average_slow.append(float(np.mean(SE_Sine_slow)))
    SE_Sine_average_fast.append(float(np.mean(SE_Sine_fast)))
    SE_Sine_average.append(float(np.mean(SE_Sine)))
    SE_Sine_sd_slow.append(float(np.std(SE_Sine_slow)))
    SE_Sine_sd_fast.append(float(np.std(SE_Sine_fast)))
    SE_Sine_sd.append(float(np.std(SE_Sine)))
    
    SE_Pink_slow = results.loc[(results['Signal ID'] == 'Pink') & (results['Speed ID'] == 'Slow') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_Pink_fast = results.loc[(results['Signal ID'] == 'Pink') & (results['Speed ID'] == 'Fast') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_Pink = results.loc[(results['Signal ID'] == 'Pink'), f'Mean Spatial error trail {i}'].to_numpy()
    SE_Pink_average_slow.append(float(np.mean(SE_Pink_slow)))
    SE_Pink_average_fast.append(float(np.mean(SE_Pink_fast)))
    SE_Pink_average.append(float(np.mean(SE_Pink)))
    SE_Pink_sd_slow.append(float(np.std(SE_Pink_slow)))
    SE_Pink_sd_fast.append(float(np.std(SE_Pink_fast)))
    SE_Pink_sd.append(float(np.std(SE_Pink)))
    
    SE_White_slow = results.loc[(results['Signal ID'] == 'White') & (results['Speed ID'] == 'Slow') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_White_fast = results.loc[(results['Signal ID'] == 'White') & (results['Speed ID'] == 'Fast') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_White = results.loc[(results['Signal ID'] == 'White'), f'Mean Spatial error trail {i}'].to_numpy()
    SE_White_average_slow.append(float(np.mean(SE_White_slow)))
    SE_White_average_fast.append(float(np.mean(SE_White_fast)))
    SE_White_average.append(float(np.mean(SE_White)))
    SE_White_sd_slow.append(float(np.std(SE_White_slow)))
    SE_White_sd_fast.append(float(np.std(SE_White_fast)))
    SE_White_sd.append(float(np.std(SE_White)))

    SE_slow = results.loc[(results['Speed ID'] == 'Slow') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_slow_average.append(float(np.mean(SE_White_slow)))
    SE_slow_sd.append(float(np.std(SE_White_slow)))
    SE_fast = results.loc[(results['Speed ID'] == 'Fast') , f'Mean Spatial error trail {i}'].to_numpy()
    SE_fast_average.append(float(np.mean(SE_White_fast)))
    SE_fast_sd.append(float(np.std(SE_White_fast)))


    


Sets = range(1,11)
# Sine
# plt.plot(Sets, SE_Sine_average, label='Sine', c='#4F4F4F')
# plt.plot(Sets, SE_Pink_average, label='Pink', c='#FFC0CB')
# plt.plot(Sets, SE_White_average, label='White', c='#D3D3D3')
# plt.fill_between(
#     Sets,
#     np.array(SE_Sine_average) - np.array(SE_Sine_sd),
#     np.array(SE_Sine_average) + np.array(SE_Sine_sd),
#     color='#4F4F4F',
#     alpha=0.2
# )
# plt.fill_between(
#     Sets,
#     np.array(SE_Pink_average) - np.array(SE_Pink_sd),
#     np.array(SE_Pink_average) + np.array(SE_Pink_sd),
#     color='#FFC0CB',
#     alpha=0.2
# )
# plt.fill_between(
#     Sets,
#     np.array(SE_White_average) - np.array(SE_White_sd),
#     np.array(SE_White_average) + np.array(SE_White_sd),
#     color='#D3D3D3',
#     alpha=0.2
# )



# plt.plot(Sets, SE_Sine_average_slow, label='Sine_slow', c='#4F4F4F', ls='--')
# plt.plot(Sets, SE_Pink_average_slow, label='Pink_slow', c='#FFC0CB', ls='--')
# plt.plot(Sets, SE_White_average_slow, label='White_slow', c='#D3D3D3', ls='--')
# plt.fill_between(
#     Sets,
#     np.array(SE_Sine_average_slow) - np.array(SE_Sine_sd_slow),
#     np.array(SE_Sine_average_slow) + np.array(SE_Sine_sd_slow),
#     color='#4F4F4F',
#     alpha=0.2
# )
# plt.fill_between(
#     Sets,
#     np.array(SE_Pink_average_slow) - np.array(SE_Pink_sd_slow),
#     np.array(SE_Pink_average_slow) + np.array(SE_Pink_sd_slow),
#     color='#FFC0CB',
#     alpha=0.2
# )
# plt.fill_between(
#     Sets,
#     np.array(SE_White_average_slow) - np.array(SE_White_sd_slow),
#     np.array(SE_White_average_slow) + np.array(SE_White_sd_slow),
#     color='#D3D3D3',
#     alpha=0.2
# )



# plt.fill_between(
#     Sets,
#     np.array(SE_Sine_average_fast) - np.array(SE_Sine_sd_fast),
#     np.array(SE_Sine_average_fast) + np.array(SE_Sine_sd_fast),
#     color='#4F4F4F',
#     alpha=0.2
# )
# plt.fill_between(
#     Sets,
#     np.array(SE_Pink_average_fast) - np.array(SE_Pink_sd_fast),
#     np.array(SE_Pink_average_fast) + np.array(SE_Pink_sd_fast),
#     color='#FFC0CB',
#     alpha=0.2
# )
# plt.fill_between(
#     Sets,
#     np.array(SE_White_average_fast) - np.array(SE_White_sd_fast),
#     np.array(SE_White_average_fast) + np.array(SE_White_sd_fast),
#     color='#D3D3D3',
#     alpha=0.2
# )
# plt.plot(Sets, SE_Sine_average_fast, label='Sine_fast', c='#4F4F4F', ls=':')
# plt.plot(Sets, SE_Pink_average_fast, label='Pink_fast', c='#FFC0CB', ls=':')
# plt.plot(Sets, SE_White_average_fast, label='White_fast', c='#D3D3D3', ls=':')
#
# plt.xticks(Sets)
# plt.legend()
# plt.show()
#
# plt.plot(Sets, SE_slow_average, label='slow', c='red')
# plt.fill_between(
#     Sets,
#     np.array(SE_slow_average) - np.array(SE_slow_sd),
#     np.array(SE_slow_average) + np.array(SE_slow_sd),
#     color='red',
#     alpha=0.2
# )
# plt.plot(Sets, SE_fast_average, label='fast', c='blue')
# # plt.fill_between(
# #     Sets,
# #     np.array(SE_fast_average) - np.array(SE_fast_sd),
# #     np.array(SE_fast_average) + np.array(SE_fast_sd),
# #     color='blue',
# #     alpha=0.2
# # )
# plt.legend()
# plt.xticks(Sets)
# plt.show()


# color_background = '#E8E8E8'
#
# fig, ax = plt.subplots(figsize=(8, 5), facecolor=color_background)  # dark grey figure
# ax.set_facecolor(color_background)  # dark grey plot background
fig, ax = plt.subplots(figsize=(8, 5))  # dark grey figure
white_color = 'slategray'
pink_color = 'lightpink'
sine_color = 'black'
ax.fill_between(
    Sets,
    np.array(SE_Sine_average_slow),
    np.array(SE_Sine_average_fast),
    color=sine_color,
    alpha=0.6
)
ax.fill_between(
    Sets,
    np.array(SE_Pink_average_slow),
    np.array(SE_Pink_average_fast),
    color=pink_color,
    alpha=0.25
)
ax.fill_between(
    Sets,
    np.array(SE_White_average_slow),
    np.array(SE_White_average_fast),
    color=white_color,
    alpha=0.2
)

ax.plot(Sets, SE_Sine_average_slow, label='Sine Slow', c=sine_color, ls='--', lw=3)
ax.plot(Sets, SE_Sine_average_fast, label='Sine Fast', c=sine_color, ls=':', lw=3)
ax.plot(Sets, SE_Pink_average_slow, label='Pink Slow', c=pink_color, ls='--', lw=3)
ax.plot(Sets, SE_Pink_average_fast, label='Pink Fast', c=pink_color, ls=':', lw=3)
# ax.plot(Sets, SE_White_average_slow, c='k', ls='--', lw=3)
# ax.plot(Sets, SE_White_average_fast, c='k', ls=':', lw=3)
ax.plot(Sets, SE_White_average_slow, label='White Slow', c=white_color, ls='--', lw=3)
ax.plot(Sets, SE_White_average_fast, label='White Fast', c=white_color, ls=':', lw=3)

ax.plot(Sets, SE_Sine_average, label='Sine', c=sine_color, lw=3)
ax.plot(Sets, SE_Pink_average, label='Pink', c=pink_color, lw=3)
ax.plot(Sets, SE_White_average, label='White', c=white_color, lw=3)

ax.legend()
ax.set_xticks(Sets)
ax.set_xlabel('Set')
ax.set_ylabel('Average Spatial Error')
# ax.spines['bottom'].set_color('white')
# ax.spines['left'].set_color('white')

plt.show()

