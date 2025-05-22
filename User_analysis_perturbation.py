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



directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Data'
files = glob.glob(os.path.join(directory_path, "*"))
information_excel = pd.read_excel(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Participants.xlsx')

list_time_to_adapt_before_down_1 = []
list_time_to_adapt_before_down_2 = []
list_time_to_adapt_before_down_3 = []
list_time_to_adapt_before_up_1 = []
list_time_to_adapt_before_up_2 = []
list_time_to_adapt_before_up_3 = []
list_time_to_adapt_after_down_1 = []
list_time_to_adapt_after_down_2 = []
list_time_to_adapt_after_down_3 = []
list_time_to_adapt_after_up_1 = []
list_time_to_adapt_after_up_2 = []
list_time_to_adapt_after_up_3 = []
list_average_time_to_adapt_before_down = []
list_average_time_to_adapt_before_up = []
list_average_time_to_adapt_after_down = []
list_average_time_to_adapt_after_up = []
list_min_time_to_adapt_before_down = []
list_min_time_to_adapt_before_up = []
list_min_time_to_adapt_after_down = []
list_min_time_to_adapt_after_up = []



list_ID = []
list_ID_team = []

sd_factor = 2
consecutive_values = 36

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    list_ID.append(ID)
    ID_team = ID[:-2]
    list_ID_team.append(ID_team)

    print(ID)

    #########################################################################################################
    # Isometric trials analysis for calculation of mean and sd of spatial error to use at time to adapt later
    file_isometrics = file+r'\Isometric_trials'
    os.chdir(file_isometrics)

    Isometric_trial_50 = pd.read_csv(r'Isometric_trial_50.csv', skiprows=2)
    Isometric_trial_10 = pd.read_csv(r'Isometric_trial_10.csv', skiprows=2)
    # print(Isometric_trial_10.columns)

    Isometric_trial_50['Performance'] = lib.Butterworth(75, 50, Isometric_trial_50['Performance'])
    Isometric_trial_10['Performance'] = lib.Butterworth(75, 50, Isometric_trial_10['Performance'])

    Isometric_50_force = Isometric_trial_50['Performance'].to_numpy()
    Isometric_10_force = Isometric_trial_10['Performance'].to_numpy()

    perc_50 = Isometric_trial_50['Target'][0]
    perc_10 = Isometric_trial_10['Target'][0]

    # Plot the generated target vs the force applied by the player
    # perc_50_numpy = np.full(len(Isometric_trial_50['Performance']), perc_50)
    # perc_10_numpy = np.full(len(Isometric_trial_10['Performance']), perc_10)
    #
    # plt.plot(Isometric_trial_50['Time'], perc_50_numpy, label='Generated targets')
    # plt.plot(Isometric_trial_50['Time'], Isometric_trial_50['Performance'], label='Player')
    # plt.title('Isometric 50% of MVC')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(Isometric_trial_10['Time'], perc_10_numpy, label='Generated targets')
    # plt.plot(Isometric_trial_10['Time'], Isometric_trial_10['Performance'], label='Player')
    # plt.title('Isometric 10% of MVC')
    # plt.legend()
    # plt.show()
    # End plot

    spatial_error_50 = abs(Isometric_50_force-perc_50)
    spatial_error_10 = abs(Isometric_10_force-perc_10)

    # Assess which 5-second window has the lowest average spatial error
    min_average_spatial_error_50 = 10000
    min_average_spatial_error_10 = 10000

    window_seconds = 5
    sampling_frequency = 75
    for i in range(len(spatial_error_50)-window_seconds*sampling_frequency):
        average_of_window_50 = np.mean(spatial_error_50[i: i+(window_seconds*sampling_frequency)])
        if min_average_spatial_error_50 > average_of_window_50:
            min_average_spatial_error_50 = average_of_window_50
            window_index_with_min_average_spatial_error_50 = [i,i+(window_seconds*sampling_frequency)]

    for i in range(len(spatial_error_10)-window_seconds*sampling_frequency):
        average_of_window_10 = np.mean(spatial_error_10[i: i+(window_seconds*sampling_frequency)])
        if min_average_spatial_error_10 > average_of_window_10:
            min_average_spatial_error_10 = average_of_window_10
            window_index_with_min_average_spatial_error_10 = [i,i+(window_seconds*sampling_frequency)]

    # This is what I needed for the calculation of time to adapt
    mean_spatial_error_50 = np.mean(spatial_error_50[window_index_with_min_average_spatial_error_50[0]: window_index_with_min_average_spatial_error_50[1]])
    mean_spatial_error_10 = np.mean(spatial_error_10[window_index_with_min_average_spatial_error_10[0]: window_index_with_min_average_spatial_error_10[1]])

    sd_spatial_error_50 = np.std(spatial_error_50[window_index_with_min_average_spatial_error_50[0]: window_index_with_min_average_spatial_error_50[1]])
    sd_spatial_error_10 = np.std(spatial_error_10[window_index_with_min_average_spatial_error_10[0]: window_index_with_min_average_spatial_error_10[1]])

    #########################################################################################################

    # Read the perturbation trials
    file_perturbations_before = file + r'\Perturbation_trials\before'
    os.chdir(file_perturbations_before)
    Perturbation_before_down_1 = pd.read_csv(r'pertr_down_1.csv', skiprows=2)
    Perturbation_before_down_2 = pd.read_csv(r'pertr_down_2.csv', skiprows=2)
    Perturbation_before_down_3 = pd.read_csv(r'pertr_down_3.csv', skiprows=2)
    Perturbation_before_up_1 = pd.read_csv(r'pertr_up_1.csv', skiprows=2)
    Perturbation_before_up_2 = pd.read_csv(r'pertr_up_2.csv', skiprows=2)
    Perturbation_before_up_3 = pd.read_csv(r'pertr_up_3.csv', skiprows=2)

    file_perturbations_after = file + r'\Perturbation_trials\after'
    os.chdir(file_perturbations_after)
    Perturbation_after_down_1 = pd.read_csv(r'pertr_down_1.csv', skiprows=2)
    Perturbation_after_down_2 = pd.read_csv(r'pertr_down_2.csv', skiprows=2)
    Perturbation_after_down_3 = pd.read_csv(r'pertr_down_3.csv', skiprows=2)
    Perturbation_after_up_1 = pd.read_csv(r'pertr_up_1.csv', skiprows=2)
    Perturbation_after_up_2 = pd.read_csv(r'pertr_up_2.csv', skiprows=2)
    Perturbation_after_up_3 = pd.read_csv(r'pertr_up_3.csv', skiprows=2)

    # plt.plot(Perturbation_after_down_1['ClosestSampleTime'], Perturbation_after_down_1['Target'], label='Targets')
    # plt.plot(Perturbation_after_down_1['Time'], Perturbation_after_down_1['Performance'], label='Player')
    # plt.legend()
    # plt.show()

    time_to_adapt_before_down_1 = lb.adaptation_time_using_sd(Perturbation_before_down_1, sd_factor, consecutive_values, ID, mean_spatial_error_10, sd_spatial_error_10, plot=True)
    time_to_adapt_before_down_2 = lb.adaptation_time_using_sd(Perturbation_before_down_2, sd_factor, consecutive_values, ID, mean_spatial_error_10, sd_spatial_error_10, plot=True)
    time_to_adapt_before_down_3 = lb.adaptation_time_using_sd(Perturbation_before_down_3, sd_factor, consecutive_values, ID, mean_spatial_error_10, sd_spatial_error_10, plot=True)
    time_to_adapt_before_up_1 = lb.adaptation_time_using_sd(Perturbation_before_up_1, sd_factor, consecutive_values, ID, mean_spatial_error_50, sd_spatial_error_50, plot=True)
    time_to_adapt_before_up_2 = lb.adaptation_time_using_sd(Perturbation_before_up_2, sd_factor, consecutive_values, ID, mean_spatial_error_50, sd_spatial_error_50, plot=True)
    time_to_adapt_before_up_3 = lb.adaptation_time_using_sd(Perturbation_before_up_3, sd_factor, consecutive_values, ID, mean_spatial_error_50, sd_spatial_error_50, plot=True)
    time_to_adapt_after_down_1 = lb.adaptation_time_using_sd(Perturbation_after_down_1, sd_factor, consecutive_values, ID, mean_spatial_error_10, sd_spatial_error_10, plot=True)
    time_to_adapt_after_down_2 = lb.adaptation_time_using_sd(Perturbation_after_down_2, sd_factor, consecutive_values, ID, mean_spatial_error_10, sd_spatial_error_10, plot=True)
    time_to_adapt_after_down_3 = lb.adaptation_time_using_sd(Perturbation_after_down_3, sd_factor, consecutive_values, ID, mean_spatial_error_10, sd_spatial_error_10, plot=True)
    time_to_adapt_after_up_1 = lb.adaptation_time_using_sd(Perturbation_after_up_1, sd_factor, consecutive_values, ID, mean_spatial_error_50, sd_spatial_error_50, plot=True)
    time_to_adapt_after_up_2 = lb.adaptation_time_using_sd(Perturbation_after_up_2, sd_factor, consecutive_values, ID, mean_spatial_error_50, sd_spatial_error_50, plot=True)
    time_to_adapt_after_up_3 = lb.adaptation_time_using_sd(Perturbation_after_up_3, sd_factor, consecutive_values, ID, mean_spatial_error_50, sd_spatial_error_50, plot=True)


    def safe_mean(*args):
        valid = [x for x in args if x is not None]
        return np.mean(valid) if valid else None


    def safe_min(*args):
        valid = [x for x in args if x is not None]
        return np.min(valid) if valid else None


    average_time_to_adapt_before_down = safe_mean(time_to_adapt_before_down_1, time_to_adapt_before_down_2, time_to_adapt_before_down_3)
    average_time_to_adapt_before_up = safe_mean(time_to_adapt_before_up_1, time_to_adapt_before_up_2, time_to_adapt_before_up_3)
    average_time_to_adapt_after_down = safe_mean(time_to_adapt_after_down_1, time_to_adapt_after_down_2, time_to_adapt_after_down_3)
    average_time_to_adapt_after_up = safe_mean(time_to_adapt_after_up_1, time_to_adapt_after_up_2, time_to_adapt_after_up_3)
    min_time_to_adapt_before_down = safe_min(time_to_adapt_before_down_1, time_to_adapt_before_down_2, time_to_adapt_before_down_3)
    min_time_to_adapt_before_up = safe_min(time_to_adapt_before_up_1, time_to_adapt_before_up_2, time_to_adapt_before_up_3)
    min_time_to_adapt_after_down = safe_min(time_to_adapt_after_down_1, time_to_adapt_after_down_2, time_to_adapt_after_down_3)
    min_time_to_adapt_after_up = safe_min(time_to_adapt_after_up_1, time_to_adapt_after_up_2, time_to_adapt_after_up_3)


    list_time_to_adapt_before_down_1.append(time_to_adapt_before_down_1)
    list_time_to_adapt_before_down_2.append(time_to_adapt_before_down_2)
    list_time_to_adapt_before_down_3.append(time_to_adapt_before_down_3)
    list_time_to_adapt_before_up_1.append(time_to_adapt_before_up_1)
    list_time_to_adapt_before_up_2.append(time_to_adapt_before_up_2)
    list_time_to_adapt_before_up_3.append(time_to_adapt_before_up_3)
    list_time_to_adapt_after_down_1.append(time_to_adapt_after_down_1)
    list_time_to_adapt_after_down_2.append(time_to_adapt_after_down_2)
    list_time_to_adapt_after_down_3.append(time_to_adapt_after_down_3)
    list_time_to_adapt_after_up_1.append(time_to_adapt_after_up_1)
    list_time_to_adapt_after_up_2.append(time_to_adapt_after_up_2)
    list_time_to_adapt_after_up_3.append(time_to_adapt_after_up_3)
    list_average_time_to_adapt_before_down.append(average_time_to_adapt_before_down)
    list_average_time_to_adapt_before_up.append(average_time_to_adapt_before_up)
    list_average_time_to_adapt_after_down.append(average_time_to_adapt_after_down)
    list_average_time_to_adapt_after_up.append(average_time_to_adapt_after_up)
    list_min_time_to_adapt_before_down.append(min_time_to_adapt_before_down)
    list_min_time_to_adapt_before_up.append(min_time_to_adapt_before_up)
    list_min_time_to_adapt_after_down.append(min_time_to_adapt_after_down)
    list_min_time_to_adapt_after_up.append(min_time_to_adapt_after_up)
    print(average_time_to_adapt_before_down)
    print(average_time_to_adapt_before_up)
    print(average_time_to_adapt_after_down)
    print(average_time_to_adapt_after_up)
    print(min_time_to_adapt_before_down)
    print(min_time_to_adapt_before_up)
    print(min_time_to_adapt_after_down)
    print(min_time_to_adapt_after_up)



dist = {'ID'        : list_ID,
        'Group ID'  : list_ID_team,
        'Time_to_adapt_before_down_1'  : list_time_to_adapt_before_down_1,
        'Time_to_adapt_before_down_2'  : list_time_to_adapt_before_down_2,
        'Time_to_adapt_before_down_3'  : list_time_to_adapt_before_down_3,
        'Average Time to adapt before down': list_average_time_to_adapt_before_down,
        'Min Time to adapt before down': list_min_time_to_adapt_before_down,
        'Time_to_adapt_before_up_1'  : list_time_to_adapt_before_up_1,
        'Time_to_adapt_before_up_2'  : list_time_to_adapt_before_up_2,
        'Time_to_adapt_before_up_3'  : list_time_to_adapt_before_up_3,
        'Average Time to adapt before up': list_average_time_to_adapt_before_up,
        'Min Time to adapt before up': list_min_time_to_adapt_before_up,
        'Time_to_adapt_after_down_1'  : list_time_to_adapt_after_down_1,
        'Time_to_adapt_after_down_2'  : list_time_to_adapt_after_down_2,
        'Time_to_adapt_after_down_3'  : list_time_to_adapt_after_down_3,
        'Average Time to adapt after down': list_average_time_to_adapt_after_down,
        'Min Time to adapt after down': list_min_time_to_adapt_after_down,
        'Time_to_adapt_after_up_1'  : list_time_to_adapt_after_up_1,
        'Time_to_adapt_after_up_2'  : list_time_to_adapt_after_up_2,
        'Time_to_adapt_after_up_3'  : list_time_to_adapt_after_up_3,
        'Average Time to adapt after up': list_average_time_to_adapt_after_up,
        'Min Time to adapt after up': list_min_time_to_adapt_after_up,
        }

results = pd.DataFrame(dist)
print(results)
# directory_to_save = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Results'
# os.chdir(directory_to_save)
# results.to_excel('Perturbation_results_2_sd.xlsx')

# Plot of Minimum adaptation times
df_long = results.melt(id_vars=['Group ID'],
                  value_vars=['Min Time to adapt before down', 'Min Time to adapt after down',
                              'Min Time to adapt before up', 'Min Time to adapt after up'],
                  var_name='Set', value_name='Min Time to adapt')  # Ensure correct name



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