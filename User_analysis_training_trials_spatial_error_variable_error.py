import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
from Lib_grip import spatial_error
import glob
import lib

# def variable_error(spatial_error, avg_spatial_error, trials):
#     variable_error = np.sqrt(np.sum((spatial_error - avg_spatial_error) ** 2) / (trials - 1))
#     # variable_error = np.array(variable_error)
#     return variable_error

#def variable_error(spatial_error):
    # mean_error = np.mean(spatial_error)
    # trials = len(spatial_error)
    # ve = np.sqrt(np.sum((spatial_error - mean_error) ** 2) / (trials - 1))
    # return ve


directory_path = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Pilot study 4\Data'

files = glob.glob(os.path.join(directory_path, "*"))


list_avg_spatial_error_1 = []
list_avg_spatial_error_2 = []
list_avg_spatial_error_3 = []
list_avg_spatial_error_4 = []
list_avg_spatial_error_5 = []
list_avg_spatial_error_6 = []
list_avg_spatial_error_7 = []
list_avg_spatial_error_8 = []
list_avg_spatial_error_9 = []
list_avg_spatial_error_10 = []

# Variable error is SD of (mean) Spatial Error
list_variable_error_1 = []
list_variable_error_2 = []
list_variable_error_3 = []
list_variable_error_4 = []
list_variable_error_5 = []
list_variable_error_6 = []
list_variable_error_7 = []
list_variable_error_8 = []
list_variable_error_9 = []
list_variable_error_10 = []

list_ID = []
list_ID_team = []

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    list_ID.append(ID)
    ID_team = ID.split(".")
    #print(ID_team)
    list_ID_team.append(ID_team[0])
    #print(ID) # We keep this so that we know which participant is assessed during the run of the code

    # Trial analysis for calculation of mean and sd of spatial error
    file_training_trials = file + r'\Training_trials'
    os.chdir(file_training_trials)

    df1 = pd.read_csv(r'Trial_1.csv', skiprows=2)
    df2 = pd.read_csv(r'Trial_2.csv', skiprows=2)
    df3 = pd.read_csv(r'Trial_3.csv', skiprows=2)
    df4 = pd.read_csv(r'Trial_4.csv', skiprows=2)
    df5 = pd.read_csv(r'Trial_5.csv', skiprows=2)
    df6 = pd.read_csv(r'Trial_6.csv', skiprows=2)
    df7 = pd.read_csv(r'Trial_7.csv', skiprows=2)
    df8 = pd.read_csv(r'Trial_8.csv', skiprows=2)
    df9 = pd.read_csv(r'Trial_9.csv', skiprows=2)
    df10 = pd.read_csv(r'Trial_10.csv', skiprows=2)

    # Filtering process
    Performance_1 = df1['Performance']
    Performance_2 = df2['Performance']
    Performance_3 = df3['Performance']
    Performance_4 = df4['Performance']
    Performance_5 = df5['Performance']
    Performance_6 = df6['Performance']
    Performance_7 = df7['Performance']
    Performance_8 = df8['Performance']
    Performance_9 = df9['Performance']
    Performance_10 = df10['Performance']

    low_pass_filter_frequency = 15
    Performance_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_1)
    Performance_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_2)
    Performance_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_3)
    Performance_4_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_4)
    Performance_5_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_5)
    Performance_6_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_6)
    Performance_7_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_7)
    Performance_8_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_8)
    Performance_9_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_9)
    Performance_10_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_10)

    df1['Performance'] = Performance_1_filtered
    df2['Performance'] = Performance_2_filtered
    df3['Performance'] = Performance_3_filtered
    df4['Performance'] = Performance_4_filtered
    df5['Performance'] = Performance_5_filtered
    df6['Performance'] = Performance_6_filtered
    df7['Performance'] = Performance_7_filtered
    df8['Performance'] = Performance_8_filtered
    df9['Performance'] = Performance_9_filtered
    df10['Performance'] = Performance_10_filtered

    # Synchronise training trials
    synch_df1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df1)
    synch_df2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df2)
    synch_df3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df3)
    synch_df4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df4)
    synch_df5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df5)
    synch_df6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df6)
    synch_df7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df7)
    synch_df8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df8)
    synch_df9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df9)
    synch_df10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df10)

    # Calculation of Spatial Error for each trial
    spatial_errors_1 = spatial_error(synch_df1)
    spatial_errors_2 = spatial_error(synch_df2)
    spatial_errors_3 = spatial_error(synch_df3)
    spatial_errors_4 = spatial_error(synch_df4)
    spatial_errors_5 = spatial_error(synch_df5)
    spatial_errors_6 = spatial_error(synch_df6)
    spatial_errors_7 = spatial_error(synch_df7)
    spatial_errors_8 = spatial_error(synch_df8)
    spatial_errors_9 = spatial_error(synch_df9)
    spatial_errors_10 = spatial_error(synch_df10)

    # Calculation of Mean Spatial Error for each trial
    avg_spatial_error_1 = np.mean(spatial_errors_1)
    avg_spatial_error_2 = np.mean(spatial_errors_2)
    avg_spatial_error_3 = np.mean(spatial_errors_3)
    avg_spatial_error_4 = np.mean(spatial_errors_4)
    avg_spatial_error_5 = np.mean(spatial_errors_5)
    avg_spatial_error_6 = np.mean(spatial_errors_6)
    avg_spatial_error_7 = np.mean(spatial_errors_7)
    avg_spatial_error_8 = np.mean(spatial_errors_8)
    avg_spatial_error_9 = np.mean(spatial_errors_9)
    avg_spatial_error_10 = np.mean(spatial_errors_10)

    # Calculation of Variable Error
    variable_error_1 = np.std(spatial_errors_1)
    variable_error_2 = np.std(spatial_errors_2)
    variable_error_3 = np.std(spatial_errors_3)
    variable_error_4 = np.std(spatial_errors_4)
    variable_error_5 = np.std(spatial_errors_5)
    variable_error_6 = np.std(spatial_errors_6)
    variable_error_7 = np.std(spatial_errors_7)
    variable_error_8 = np.std(spatial_errors_8)
    variable_error_9 = np.std(spatial_errors_9)
    variable_error_10 = np.std(spatial_errors_10)

    # Append lists of Spatial Errors
    list_avg_spatial_error_1.append(avg_spatial_error_1)
    list_avg_spatial_error_2.append(avg_spatial_error_2)
    list_avg_spatial_error_3.append(avg_spatial_error_3)
    list_avg_spatial_error_4.append(avg_spatial_error_4)
    list_avg_spatial_error_5.append(avg_spatial_error_5)
    list_avg_spatial_error_6.append(avg_spatial_error_6)
    list_avg_spatial_error_7.append(avg_spatial_error_7)
    list_avg_spatial_error_8.append(avg_spatial_error_8)
    list_avg_spatial_error_9.append(avg_spatial_error_9)
    list_avg_spatial_error_10.append(avg_spatial_error_10)

    # Append Variable Error
    list_variable_error_1.append(variable_error_1)
    list_variable_error_2.append(variable_error_2)
    list_variable_error_3.append(variable_error_3)
    list_variable_error_4.append(variable_error_4)
    list_variable_error_5.append(variable_error_5)
    list_variable_error_6.append(variable_error_6)
    list_variable_error_7.append(variable_error_7)
    list_variable_error_8.append(variable_error_8)
    list_variable_error_9.append(variable_error_9)
    list_variable_error_10.append(variable_error_10)

# Create a dictionery
dist = {
       'ID'        : list_ID,
       'Group ID'  : list_ID_team,
       'Mean Spatial error trail 1': list_avg_spatial_error_1,
       'Mean Spatial error trail 2': list_avg_spatial_error_2,
       'Mean Spatial error trail 3': list_avg_spatial_error_3,
       'Mean Spatial error trail 4': list_avg_spatial_error_4,
       'Mean Spatial error trail 5': list_avg_spatial_error_5,
       'Mean Spatial error trail 6': list_avg_spatial_error_6,
       'Mean Spatial error trail 7': list_avg_spatial_error_7,
       'Mean Spatial error trail 8': list_avg_spatial_error_8,
       'Mean Spatial error trail 9': list_avg_spatial_error_9,
       'Mean Spatial error trail 10': list_avg_spatial_error_10,
       'Variable Error trial  1': list_variable_error_1,
       'Variable Error trial  2': list_variable_error_2,
       'Variable Error trial  3': list_variable_error_3,
       'Variable Error trial  4': list_variable_error_4,
       'Variable Error trial  5': list_variable_error_5,
       'Variable Error trial  6': list_variable_error_6,
       'Variable Error trial  7': list_variable_error_7,
       'Variable Error trial  8': list_variable_error_8,
       'Variable Error trial  9': list_variable_error_9,
       'Variable Error trial  10': list_variable_error_10
       }

# print(list_variable_error_1)
# print(len(list_variable_error_1))


# # Comparison of Performance and Targets over time
# plt.plot(synch_df1["Target"], color='g', label='Targets')
# plt.plot(synch_df1["Performance"], color='r', label='Performance')
# plt.legend()
# plt.show()
#
# # Comparison of Spatial Error and Variable Error over performance time
# plt.plot(list_variable_error_1, color='r', label='Variable Error')
# plt.plot(spatial_errors_1, color='b', label='Average Spatial Error')
# plt.legend()
# plt.show()

df_results = pd.DataFrame(dist)
directory_to_save = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Pilot study 4\Results'

os.chdir(directory_to_save)
df_results.to_excel('Mean_spatial_error_&_Variable_error_results_with_15_low_pass_filter.xlsx')