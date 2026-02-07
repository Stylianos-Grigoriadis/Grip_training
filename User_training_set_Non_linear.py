import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from Lib_grip import spatial_error
import glob
import lib

pd.set_option('display.max_rows', None)


Stylianos = True
if Stylianos == True:
    directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Valid data\Force data'
else:
    directory_path = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Data\Valid data\Force data'

files = glob.glob(os.path.join(directory_path, "*"))

list_SaEn_time_delay_training_1 = []
list_SaEn_time_delay_training_2 = []
list_SaEn_time_delay_training_3 = []
list_SaEn_time_delay_training_4 = []
list_SaEn_time_delay_training_5 = []
list_SaEn_time_delay_training_6 = []
list_SaEn_time_delay_training_7 = []
list_SaEn_time_delay_training_8 = []
list_SaEn_time_delay_training_9 = []
list_SaEn_time_delay_training_10 = []

list_DFA_training_1 = []
list_DFA_training_2 = []
list_DFA_training_3 = []
list_DFA_training_4 = []
list_DFA_training_5 = []
list_DFA_training_6 = []
list_DFA_training_7 = []
list_DFA_training_8 = []
list_DFA_training_9 = []
list_DFA_training_10 = []


list_ID = []
list_ID_team = []
list_signal = []
list_speed = []

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    ID_team = ID.split(".")
    signal = ID.split('_')[0]
    speed_code = ID.split('_')[1].split('.')[0]
    speed_map = {
        '65': 'Slow',
        '100': 'Fast'
    }
    speed = speed_map[speed_code]

    list_ID.append(ID)
    list_ID_team.append(ID_team[0])
    list_signal.append(signal)
    list_speed.append(speed)

    print(ID) # We keep this so that we know which participant is assessed during the run of the code
    # print(ID_team[0])
    # print(signal)
    # print(speed)

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

    # After visual inspection we decided that after 3 seconds the force output stabilizes
    time_threshold = 3
    df1_after_threshold = df1[df1['Time'] > time_threshold].reset_index(drop=True).copy()
    df2_after_threshold = df2[df2['Time'] > time_threshold].reset_index(drop=True).copy()
    df3_after_threshold = df3[df3['Time'] > time_threshold].reset_index(drop=True).copy()
    df4_after_threshold = df4[df4['Time'] > time_threshold].reset_index(drop=True).copy()
    df5_after_threshold = df5[df5['Time'] > time_threshold].reset_index(drop=True).copy()
    df6_after_threshold = df6[df6['Time'] > time_threshold].reset_index(drop=True).copy()
    df7_after_threshold = df7[df7['Time'] > time_threshold].reset_index(drop=True).copy()
    df8_after_threshold = df8[df8['Time'] > time_threshold].reset_index(drop=True).copy()
    df9_after_threshold = df9[df9['Time'] > time_threshold].reset_index(drop=True).copy()
    df10_after_threshold = df10[df10['Time'] > time_threshold].reset_index(drop=True).copy()


    # Calculation of DFA of each training trial
    scales_training_set_1 = np.arange(10, len(df1_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_2 = np.arange(10, len(df2_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_3 = np.arange(10, len(df3_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_4 = np.arange(10, len(df4_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_5 = np.arange(10, len(df5_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_6 = np.arange(10, len(df6_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_7 = np.arange(10, len(df7_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_8 = np.arange(10, len(df8_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_9 = np.arange(10, len(df9_after_threshold['Performance'].to_numpy()) // 10)
    scales_training_set_10 = np.arange(10, len(df10_after_threshold['Performance'].to_numpy()) // 10)


    _, _, DFA_training_set_1 = lb.DFA_NONAN(df1_after_threshold['Performance'].to_numpy(), scales_training_set_1, order=1, plot=False)
    _, _, DFA_training_set_2 = lb.DFA_NONAN(df2_after_threshold['Performance'].to_numpy(), scales_training_set_2, order=1, plot=False)
    _, _, DFA_training_set_3 = lb.DFA_NONAN(df3_after_threshold['Performance'].to_numpy(), scales_training_set_3, order=1, plot=False)
    _, _, DFA_training_set_4 = lb.DFA_NONAN(df4_after_threshold['Performance'].to_numpy(), scales_training_set_4, order=1, plot=False)
    _, _, DFA_training_set_5 = lb.DFA_NONAN(df5_after_threshold['Performance'].to_numpy(), scales_training_set_5, order=1, plot=False)
    _, _, DFA_training_set_6 = lb.DFA_NONAN(df6_after_threshold['Performance'].to_numpy(), scales_training_set_6, order=1, plot=False)
    _, _, DFA_training_set_7 = lb.DFA_NONAN(df7_after_threshold['Performance'].to_numpy(), scales_training_set_7, order=1, plot=False)
    _, _, DFA_training_set_8 = lb.DFA_NONAN(df8_after_threshold['Performance'].to_numpy(), scales_training_set_8, order=1, plot=False)
    _, _, DFA_training_set_9 = lb.DFA_NONAN(df9_after_threshold['Performance'].to_numpy(), scales_training_set_9, order=1, plot=False)
    _, _, DFA_training_set_10 = lb.DFA_NONAN(df10_after_threshold['Performance'].to_numpy(), scales_training_set_10, order=1, plot=False)

    # Create the DFA lists
    list_DFA_training_1.append(DFA_training_set_1)
    list_DFA_training_2.append(DFA_training_set_2)
    list_DFA_training_3.append(DFA_training_set_3)
    list_DFA_training_4.append(DFA_training_set_4)
    list_DFA_training_5.append(DFA_training_set_5)
    list_DFA_training_6.append(DFA_training_set_6)
    list_DFA_training_7.append(DFA_training_set_7)
    list_DFA_training_8.append(DFA_training_set_8)
    list_DFA_training_9.append(DFA_training_set_9)
    list_DFA_training_10.append(DFA_training_set_10)


    # Calculation of SaEn of each training trial
    # AMI_training_set_1, _ = lb.AMI_Stergiou(df1_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_2, _ = lb.AMI_Stergiou(df2_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_3, _ = lb.AMI_Stergiou(df3_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_4, _ = lb.AMI_Stergiou(df4_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_5, _ = lb.AMI_Stergiou(df5_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_6, _ = lb.AMI_Stergiou(df6_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_7, _ = lb.AMI_Stergiou(df7_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_8, _ = lb.AMI_Stergiou(df8_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_9, _ = lb.AMI_Stergiou(df9_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)
    # AMI_training_set_10, _ = lb.AMI_Stergiou(df10_after_threshold['Performance'].to_numpy(), 5, 75, plot=False)


    # SaEn_time_delay_training_1 = lb.SaEn_once_again(df1_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_2 = lb.SaEn_once_again(df2_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_3 = lb.SaEn_once_again(df3_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_4 = lb.SaEn_once_again(df4_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_5 = lb.SaEn_once_again(df5_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_6 = lb.SaEn_once_again(df6_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_7 = lb.SaEn_once_again(df7_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_8 = lb.SaEn_once_again(df8_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_9 = lb.SaEn_once_again(df9_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)
    # SaEn_time_delay_training_10 = lb.SaEn_once_again(df10_after_threshold['Performance'], 2, 0.2, tau=None, Theiler_Window=False)

    # Create the SaEn lists
    # list_SaEn_time_delay_training_1.append(SaEn_time_delay_training_1)
    # list_SaEn_time_delay_training_2.append(SaEn_time_delay_training_2)
    # list_SaEn_time_delay_training_3.append(SaEn_time_delay_training_3)
    # list_SaEn_time_delay_training_4.append(SaEn_time_delay_training_4)
    # list_SaEn_time_delay_training_5.append(SaEn_time_delay_training_5)
    # list_SaEn_time_delay_training_6.append(SaEn_time_delay_training_6)
    # list_SaEn_time_delay_training_7.append(SaEn_time_delay_training_7)
    # list_SaEn_time_delay_training_8.append(SaEn_time_delay_training_8)
    # list_SaEn_time_delay_training_9.append(SaEn_time_delay_training_9)
    # list_SaEn_time_delay_training_10.append(SaEn_time_delay_training_10)

# Create a dictionary SaEn
# dist_SaEn = {
#        'ID'        : list_ID,
#        'Group ID'  : list_ID_team,
#        'Signal'    : list_signal,
#        'Speed'     : list_speed,
#        'SaEn training set 1': list_SaEn_time_delay_training_1,
#        'SaEn training set 2': list_SaEn_time_delay_training_2,
#        'SaEn training set 3': list_SaEn_time_delay_training_3,
#        'SaEn training set 4': list_SaEn_time_delay_training_4,
#        'SaEn training set 5': list_SaEn_time_delay_training_5,
#        'SaEn training set 6': list_SaEn_time_delay_training_6,
#        'SaEn training set 7': list_SaEn_time_delay_training_7,
#        'SaEn training set 8': list_SaEn_time_delay_training_8,
#        'SaEn training set 9': list_SaEn_time_delay_training_9,
#        'SaEn training set 10': list_SaEn_time_delay_training_10,
#        }

# Create a dictionary DFA
dist_DFA = {
       'ID'        : list_ID,
       'Group ID'  : list_ID_team,
       'Signal'    : list_signal,
       'Speed'     : list_speed,
       'DFA training set 1': list_DFA_training_1,
       'DFA training set 2': list_DFA_training_2,
       'DFA training set 3': list_DFA_training_3,
       'DFA training set 4': list_DFA_training_4,
       'DFA training set 5': list_DFA_training_5,
       'DFA training set 6': list_DFA_training_6,
       'DFA training set 7': list_DFA_training_7,
       'DFA training set 8': list_DFA_training_8,
       'DFA training set 9': list_DFA_training_9,
       'DFA training set 10': list_DFA_training_10,
       }



df_results = pd.DataFrame(dist_DFA)
if Stylianos == True:
    directory_to_save = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results'
else:
    directory_to_save = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Results'

os.chdir(directory_to_save)
df_results.to_excel('Training_trials_DFA_10_N10_results.xlsx')






