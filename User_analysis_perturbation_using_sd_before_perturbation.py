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


pd.set_option('display.max_rows', None)


Stylianos = True
if Stylianos == True:
    directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Valid data\Force data'
else:
    directory_path = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Data\Valid data\Force data'
files = glob.glob(os.path.join(directory_path, "*"))
information_excel = pd.read_excel(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Participants.xlsx')

list_time_to_adapt_before_down_1_sd = []
list_time_to_adapt_before_down_2_sd = []
list_time_to_adapt_before_down_3_sd = []
list_time_to_adapt_before_up_1_sd = []
list_time_to_adapt_before_up_2_sd = []
list_time_to_adapt_before_up_3_sd = []
list_time_to_adapt_after_down_1_sd = []
list_time_to_adapt_after_down_2_sd = []
list_time_to_adapt_after_down_3_sd = []
list_time_to_adapt_after_up_1_sd = []
list_time_to_adapt_after_up_2_sd = []
list_time_to_adapt_after_up_3_sd = []
list_average_time_to_adapt_before_down_sd = []
list_average_time_to_adapt_before_up_sd = []
list_average_time_to_adapt_after_down_sd = []
list_average_time_to_adapt_after_up_sd = []
list_min_time_to_adapt_before_down_sd = []
list_min_time_to_adapt_before_up_sd = []
list_min_time_to_adapt_after_down_sd = []
list_min_time_to_adapt_after_up_sd = []
list_difference_time_to_adapt_before_after_down_1_sd = []
list_difference_time_to_adapt_before_after_down_2_sd = []
list_difference_time_to_adapt_before_after_down_3_sd = []
list_difference_time_to_adapt_before_after_up_1_sd = []
list_difference_time_to_adapt_before_after_up_2_sd = []
list_difference_time_to_adapt_before_after_up_3_sd = []
list_difference_min_time_to_adapt_before_after_up_sd = []
list_difference_min_time_to_adapt_before_after_down_sd = []
list_difference_average_time_to_adapt_before_after_up_sd = []
list_difference_average_time_to_adapt_before_after_down_sd = []

list_time_to_adapt_before_down_1_asymp = []
list_time_to_adapt_before_down_2_asymp = []
list_time_to_adapt_before_down_3_asymp = []
list_time_to_adapt_before_up_1_asymp = []
list_time_to_adapt_before_up_2_asymp = []
list_time_to_adapt_before_up_3_asymp = []
list_time_to_adapt_after_down_1_asymp = []
list_time_to_adapt_after_down_2_asymp = []
list_time_to_adapt_after_down_3_asymp = []
list_time_to_adapt_after_up_1_asymp = []
list_time_to_adapt_after_up_2_asymp = []
list_time_to_adapt_after_up_3_asymp = []
list_average_time_to_adapt_before_down_asymp = []
list_average_time_to_adapt_before_up_asymp = []
list_average_time_to_adapt_after_down_asymp = []
list_average_time_to_adapt_after_up_asymp = []
list_min_time_to_adapt_before_down_asymp = []
list_min_time_to_adapt_before_up_asymp = []
list_min_time_to_adapt_after_down_asymp = []
list_min_time_to_adapt_after_up_asymp = []
list_difference_time_to_adapt_before_after_down_1_asymp = []
list_difference_time_to_adapt_before_after_down_2_asymp = []
list_difference_time_to_adapt_before_after_down_3_asymp = []
list_difference_time_to_adapt_before_after_up_1_asymp = []
list_difference_time_to_adapt_before_after_up_2_asymp = []
list_difference_time_to_adapt_before_after_up_3_asymp = []
list_difference_min_time_to_adapt_before_after_up_asymp = []
list_difference_min_time_to_adapt_before_after_down_asymp = []
list_difference_average_time_to_adapt_before_after_up_asymp = []
list_difference_average_time_to_adapt_before_after_down_asymp = []


list_ID = []
list_ID_team = []
list_signal = []
list_speed = []

sd_factor = 3
time_window = 1
asymptote_fraction = 0.95
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

    print(ID)

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

    # Read isometric trials
    file_isometric = file + r'\Isometric_trials'
    os.chdir(file_isometric)
    isometric_10 = pd.read_csv(r'Isometric_trial_10.csv', skiprows=2)
    isometric_50 = pd.read_csv(r'Isometric_trial_50.csv', skiprows=2)

    # Filtering
    Performance_perturbation_before_down_1 = Perturbation_before_down_1['Performance'].to_numpy()
    Performance_perturbation_before_down_2 = Perturbation_before_down_2['Performance'].to_numpy()
    Performance_perturbation_before_down_3 = Perturbation_before_down_3['Performance'].to_numpy()
    Performance_perturbation_before_up_1 = Perturbation_before_up_1['Performance'].to_numpy()
    Performance_perturbation_before_up_2 = Perturbation_before_up_2['Performance'].to_numpy()
    Performance_perturbation_before_up_3 = Perturbation_before_up_3['Performance'].to_numpy()
    Performance_perturbation_after_down_1 = Perturbation_after_down_1['Performance'].to_numpy()
    Performance_perturbation_after_down_2 = Perturbation_after_down_2['Performance'].to_numpy()
    Performance_perturbation_after_down_3 = Perturbation_after_down_3['Performance'].to_numpy()
    Performance_perturbation_after_up_1 = Perturbation_after_up_1['Performance'].to_numpy()
    Performance_perturbation_after_up_2 = Perturbation_after_up_2['Performance'].to_numpy()
    Performance_perturbation_after_up_3 = Perturbation_after_up_3['Performance'].to_numpy()
    Performance_isometric_10 = isometric_10['Performance'].to_numpy()
    Performance_isometric_50 = isometric_50['Performance'].to_numpy()

    low_pass_filter_frequency = 15
    Performance_perturbation_before_down_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_before_down_1)
    Performance_perturbation_before_down_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_before_down_2)
    Performance_perturbation_before_down_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_before_down_3)
    Performance_perturbation_before_up_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_before_up_1)
    Performance_perturbation_before_up_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_before_up_2)
    Performance_perturbation_before_up_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_before_up_3)
    Performance_perturbation_after_down_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_after_down_1)
    Performance_perturbation_after_down_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_after_down_2)
    Performance_perturbation_after_down_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_after_down_3)
    Performance_perturbation_after_up_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_after_up_1)
    Performance_perturbation_after_up_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_after_up_2)
    Performance_perturbation_after_up_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_perturbation_after_up_3)
    Performance_isometric_10_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_isometric_10)
    Performance_isometric_50_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_isometric_50)

    Perturbation_before_down_1['Performance'] = Performance_perturbation_before_down_1_filtered
    Perturbation_before_down_2['Performance'] = Performance_perturbation_before_down_2_filtered
    Perturbation_before_down_3['Performance'] = Performance_perturbation_before_down_3_filtered
    Perturbation_before_up_1['Performance'] = Performance_perturbation_before_up_1_filtered
    Perturbation_before_up_2['Performance'] = Performance_perturbation_before_up_2_filtered
    Perturbation_before_up_3['Performance'] = Performance_perturbation_before_up_3_filtered
    Perturbation_after_down_1['Performance'] = Performance_perturbation_after_down_1_filtered
    Perturbation_after_down_2['Performance'] = Performance_perturbation_after_down_2_filtered
    Perturbation_after_down_3['Performance'] = Performance_perturbation_after_down_3_filtered
    Perturbation_after_up_1['Performance'] = Performance_perturbation_after_up_1_filtered
    Perturbation_after_up_2['Performance'] = Performance_perturbation_after_up_2_filtered
    Perturbation_after_up_3['Performance'] = Performance_perturbation_after_up_3_filtered
    isometric_10['Performance'] = Performance_isometric_10_filtered
    isometric_50['Performance'] = Performance_isometric_50_filtered



    # Calculate average and sd of isometric trials for adaptation calculation
    # After visual inspection we decided that after 3 seconds the force output stabilizes
    synch_isometric_10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(isometric_10)
    synch_isometric_50 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(isometric_50)


    time_threshold = 3
    isometric_10_after_threshold = synch_isometric_10[synch_isometric_10['Time'] > time_threshold].reset_index(drop=True).copy()
    isometric_50_after_threshold = synch_isometric_50[synch_isometric_50['Time'] > time_threshold].reset_index(drop=True).copy()
    spatial_errors_10 = lb.spatial_error(isometric_10_after_threshold)
    spatial_errors_50 = lb.spatial_error(isometric_50_after_threshold)
    mean_spatial_error_10 = np.mean(spatial_errors_10)
    mean_spatial_error_50 = np.mean(spatial_errors_50)
    sd_spatial_error_10 = np.std(spatial_errors_10)
    sd_spatial_error_50 = np.std(spatial_errors_50)
    print(isometric_10_after_threshold.columns)
    # plt.plot(isometric_10_after_threshold['Time'], isometric_10_after_threshold['Performance'], label="force output")
    # plt.plot(isometric_10_after_threshold['ClosestSampleTime'], isometric_10_after_threshold['Target'], label="target")
    # plt.plot(isometric_10_after_threshold['ClosestSampleTime'],spatial_errors_10, label='Spatial error')
    # plt.legend()


    # Calculation of adaptation using sd from isometric trials
    print("Perturbation down before")
    time_to_adapt_before_down_1_sd, time_to_adapt_before_down_1_asymp = lb.adaptation_time_using_sd_from_isometric_trials_and_asymptotes(Perturbation_before_down_1, sd_factor, time_window, ID, mean_spatial_error_10, sd_spatial_error_10, asymptote_fraction=asymptote_fraction, plot='combined')
    time_to_adapt_before_down_2_sd, time_to_adapt_before_down_2_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_before_down_2, sd_factor, time_window, ID, mean_spatial_error_10, sd_spatial_error_10, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_before_down_3_sd, time_to_adapt_before_down_3_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_before_down_3, sd_factor, time_window, ID, mean_spatial_error_10, sd_spatial_error_10, asymptote_fraction=asymptote_fraction, plot='None')
    print("Perturbation up before")
    time_to_adapt_before_up_1_sd, time_to_adapt_before_up_1_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_before_up_1, sd_factor, time_window, ID, mean_spatial_error_50, sd_spatial_error_50, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_before_up_2_sd, time_to_adapt_before_up_2_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_before_up_2, sd_factor, time_window, ID, mean_spatial_error_50, sd_spatial_error_50, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_before_up_3_sd, time_to_adapt_before_up_3_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_before_up_3, sd_factor, time_window, ID, mean_spatial_error_50, sd_spatial_error_50, asymptote_fraction=asymptote_fraction, plot='None')
    print("Perturbation down after")
    time_to_adapt_after_down_1_sd, time_to_adapt_after_down_1_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_after_down_1, sd_factor, time_window, ID, mean_spatial_error_10, sd_spatial_error_10, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_after_down_2_sd, time_to_adapt_after_down_2_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_after_down_2, sd_factor, time_window, ID, mean_spatial_error_10, sd_spatial_error_10, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_after_down_3_sd, time_to_adapt_after_down_3_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_after_down_3, sd_factor, time_window, ID, mean_spatial_error_10, sd_spatial_error_10, asymptote_fraction=asymptote_fraction, plot='None')
    print("Perturbation up after")
    time_to_adapt_after_up_1_sd, time_to_adapt_after_up_1_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_after_up_1, sd_factor, time_window, ID, mean_spatial_error_50, sd_spatial_error_50, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_after_up_2_sd, time_to_adapt_after_up_2_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_after_up_2, sd_factor, time_window, ID, mean_spatial_error_50, sd_spatial_error_50, asymptote_fraction=asymptote_fraction, plot='None')
    time_to_adapt_after_up_3_sd, time_to_adapt_after_up_3_asymp = lb.adaptation_time_using_sd_from_isometric_trials(Perturbation_after_up_3, sd_factor, time_window, ID, mean_spatial_error_50, sd_spatial_error_50, asymptote_fraction=asymptote_fraction, plot='None')


    def safe_mean(*args):
        valid = [x for x in args if x is not None]
        return np.mean(valid) if valid else None


    def safe_min(*args):
        valid = [x for x in args if x is not None]
        return np.min(valid) if valid else None


    def safe_difference(*args):
        if any(x is None for x in args):
            return None
        return args[0] - args[1]


    average_time_to_adapt_before_down_sd = safe_mean(time_to_adapt_before_down_1_sd, time_to_adapt_before_down_2_sd, time_to_adapt_before_down_3_sd)
    average_time_to_adapt_before_up_sd = safe_mean(time_to_adapt_before_up_1_sd, time_to_adapt_before_up_2_sd, time_to_adapt_before_up_3_sd)
    average_time_to_adapt_after_down_sd = safe_mean(time_to_adapt_after_down_1_sd, time_to_adapt_after_down_2_sd, time_to_adapt_after_down_3_sd)
    average_time_to_adapt_after_up_sd = safe_mean(time_to_adapt_after_up_1_sd, time_to_adapt_after_up_2_sd, time_to_adapt_after_up_3_sd)
    min_time_to_adapt_before_down_sd = safe_min(time_to_adapt_before_down_1_sd, time_to_adapt_before_down_2_sd, time_to_adapt_before_down_3_sd)
    min_time_to_adapt_before_up_sd = safe_min(time_to_adapt_before_up_1_sd, time_to_adapt_before_up_2_sd, time_to_adapt_before_up_3_sd)
    min_time_to_adapt_after_down_sd = safe_min(time_to_adapt_after_down_1_sd, time_to_adapt_after_down_2_sd, time_to_adapt_after_down_3_sd)
    min_time_to_adapt_after_up_sd = safe_min(time_to_adapt_after_up_1_sd, time_to_adapt_after_up_2_sd, time_to_adapt_after_up_3_sd)
    difference_time_to_adapt_before_after_down_1_sd = safe_difference(time_to_adapt_after_down_1_sd, time_to_adapt_before_down_1_sd)
    difference_time_to_adapt_before_after_down_2_sd = safe_difference(time_to_adapt_after_down_2_sd, time_to_adapt_before_down_2_sd)
    difference_time_to_adapt_before_after_down_3_sd = safe_difference(time_to_adapt_after_down_3_sd, time_to_adapt_before_down_3_sd)
    difference_time_to_adapt_before_after_up_1_sd = safe_difference(time_to_adapt_after_up_1_sd, time_to_adapt_before_up_1_sd)
    difference_time_to_adapt_before_after_up_2_sd = safe_difference(time_to_adapt_after_up_2_sd, time_to_adapt_before_up_2_sd)
    difference_time_to_adapt_before_after_up_3_sd = safe_difference(time_to_adapt_after_up_3_sd, time_to_adapt_before_up_3_sd)
    difference_min_time_to_adapt_before_after_up_sd = safe_difference(min_time_to_adapt_after_up_sd, min_time_to_adapt_before_up_sd)
    difference_min_time_to_adapt_before_after_down_sd = safe_difference(min_time_to_adapt_after_down_sd, min_time_to_adapt_before_down_sd)
    difference_average_time_to_adapt_before_after_up_sd = safe_difference(average_time_to_adapt_after_up_sd, average_time_to_adapt_before_up_sd)
    difference_average_time_to_adapt_before_after_down_sd = safe_difference(average_time_to_adapt_after_down_sd, average_time_to_adapt_before_down_sd)

    average_time_to_adapt_before_down_asymp = safe_mean(time_to_adapt_before_down_1_asymp, time_to_adapt_before_down_2_asymp, time_to_adapt_before_down_3_asymp)
    average_time_to_adapt_before_up_asymp = safe_mean(time_to_adapt_before_up_1_asymp, time_to_adapt_before_up_2_asymp, time_to_adapt_before_up_3_asymp)
    average_time_to_adapt_after_down_asymp = safe_mean(time_to_adapt_after_down_1_asymp, time_to_adapt_after_down_2_asymp, time_to_adapt_after_down_3_asymp)
    average_time_to_adapt_after_up_asymp = safe_mean(time_to_adapt_after_up_1_asymp, time_to_adapt_after_up_2_asymp, time_to_adapt_after_up_3_asymp)
    min_time_to_adapt_before_down_asymp = safe_min(time_to_adapt_before_down_1_asymp, time_to_adapt_before_down_2_asymp, time_to_adapt_before_down_3_asymp)
    min_time_to_adapt_before_up_asymp = safe_min(time_to_adapt_before_up_1_asymp, time_to_adapt_before_up_2_asymp, time_to_adapt_before_up_3_asymp)
    min_time_to_adapt_after_down_asymp = safe_min(time_to_adapt_after_down_1_asymp, time_to_adapt_after_down_2_asymp, time_to_adapt_after_down_3_asymp)
    min_time_to_adapt_after_up_asymp = safe_min(time_to_adapt_after_up_1_asymp, time_to_adapt_after_up_2_asymp, time_to_adapt_after_up_3_asymp)
    difference_time_to_adapt_before_after_down_1_asymp = safe_difference(time_to_adapt_after_down_1_asymp, time_to_adapt_before_down_1_asymp)
    difference_time_to_adapt_before_after_down_2_asymp = safe_difference(time_to_adapt_after_down_2_asymp, time_to_adapt_before_down_2_asymp)
    difference_time_to_adapt_before_after_down_3_asymp = safe_difference(time_to_adapt_after_down_3_asymp, time_to_adapt_before_down_3_asymp)
    difference_time_to_adapt_before_after_up_1_asymp = safe_difference(time_to_adapt_after_up_1_asymp, time_to_adapt_before_up_1_asymp)
    difference_time_to_adapt_before_after_up_2_asymp = safe_difference(time_to_adapt_after_up_2_asymp, time_to_adapt_before_up_2_asymp)
    difference_time_to_adapt_before_after_up_3_asymp = safe_difference(time_to_adapt_after_up_3_asymp, time_to_adapt_before_up_3_asymp)
    difference_min_time_to_adapt_before_after_up_asymp = safe_difference(min_time_to_adapt_after_up_asymp, min_time_to_adapt_before_up_asymp)
    difference_min_time_to_adapt_before_after_down_asymp = safe_difference(min_time_to_adapt_after_down_asymp, min_time_to_adapt_before_down_asymp)
    difference_average_time_to_adapt_before_after_up_asymp = safe_difference(average_time_to_adapt_after_up_asymp, average_time_to_adapt_before_up_asymp)
    difference_average_time_to_adapt_before_after_down_asymp = safe_difference(average_time_to_adapt_after_down_asymp, average_time_to_adapt_before_down_asymp)


    list_time_to_adapt_before_down_1_sd.append(time_to_adapt_before_down_1_sd)
    list_time_to_adapt_before_down_2_sd.append(time_to_adapt_before_down_2_sd)
    list_time_to_adapt_before_down_3_sd.append(time_to_adapt_before_down_3_sd)
    list_time_to_adapt_before_up_1_sd.append(time_to_adapt_before_up_1_sd)
    list_time_to_adapt_before_up_2_sd.append(time_to_adapt_before_up_2_sd)
    list_time_to_adapt_before_up_3_sd.append(time_to_adapt_before_up_3_sd)
    list_time_to_adapt_after_down_1_sd.append(time_to_adapt_after_down_1_sd)
    list_time_to_adapt_after_down_2_sd.append(time_to_adapt_after_down_2_sd)
    list_time_to_adapt_after_down_3_sd.append(time_to_adapt_after_down_3_sd)
    list_time_to_adapt_after_up_1_sd.append(time_to_adapt_after_up_1_sd)
    list_time_to_adapt_after_up_2_sd.append(time_to_adapt_after_up_2_sd)
    list_time_to_adapt_after_up_3_sd.append(time_to_adapt_after_up_3_sd)
    list_average_time_to_adapt_before_down_sd.append(average_time_to_adapt_before_down_sd)
    list_average_time_to_adapt_before_up_sd.append(average_time_to_adapt_before_up_sd)
    list_average_time_to_adapt_after_down_sd.append(average_time_to_adapt_after_down_sd)
    list_average_time_to_adapt_after_up_sd.append(average_time_to_adapt_after_up_sd)
    list_min_time_to_adapt_before_down_sd.append(min_time_to_adapt_before_down_sd)
    list_min_time_to_adapt_before_up_sd.append(min_time_to_adapt_before_up_sd)
    list_min_time_to_adapt_after_down_sd.append(min_time_to_adapt_after_down_sd)
    list_min_time_to_adapt_after_up_sd.append(min_time_to_adapt_after_up_sd)
    list_difference_time_to_adapt_before_after_down_1_sd.append(difference_time_to_adapt_before_after_down_1_sd)
    list_difference_time_to_adapt_before_after_down_2_sd.append(difference_time_to_adapt_before_after_down_2_sd)
    list_difference_time_to_adapt_before_after_down_3_sd.append(difference_time_to_adapt_before_after_down_3_sd)
    list_difference_time_to_adapt_before_after_up_1_sd.append(difference_time_to_adapt_before_after_up_1_sd)
    list_difference_time_to_adapt_before_after_up_2_sd.append(difference_time_to_adapt_before_after_up_2_sd)
    list_difference_time_to_adapt_before_after_up_3_sd.append(difference_time_to_adapt_before_after_up_3_sd)
    list_difference_min_time_to_adapt_before_after_up_sd.append(difference_min_time_to_adapt_before_after_up_sd)
    list_difference_min_time_to_adapt_before_after_down_sd.append(difference_min_time_to_adapt_before_after_down_sd)
    list_difference_average_time_to_adapt_before_after_up_sd.append(difference_average_time_to_adapt_before_after_up_sd)
    list_difference_average_time_to_adapt_before_after_down_sd.append(difference_average_time_to_adapt_before_after_down_sd)

    list_time_to_adapt_before_down_1_asymp.append(time_to_adapt_before_down_1_asymp)
    list_time_to_adapt_before_down_2_asymp.append(time_to_adapt_before_down_2_asymp)
    list_time_to_adapt_before_down_3_asymp.append(time_to_adapt_before_down_3_asymp)
    list_time_to_adapt_before_up_1_asymp.append(time_to_adapt_before_up_1_asymp)
    list_time_to_adapt_before_up_2_asymp.append(time_to_adapt_before_up_2_asymp)
    list_time_to_adapt_before_up_3_asymp.append(time_to_adapt_before_up_3_asymp)
    list_time_to_adapt_after_down_1_asymp.append(time_to_adapt_after_down_1_asymp)
    list_time_to_adapt_after_down_2_asymp.append(time_to_adapt_after_down_2_asymp)
    list_time_to_adapt_after_down_3_asymp.append(time_to_adapt_after_down_3_asymp)
    list_time_to_adapt_after_up_1_asymp.append(time_to_adapt_after_up_1_asymp)
    list_time_to_adapt_after_up_2_asymp.append(time_to_adapt_after_up_2_asymp)
    list_time_to_adapt_after_up_3_asymp.append(time_to_adapt_after_up_3_asymp)
    list_average_time_to_adapt_before_down_asymp.append(average_time_to_adapt_before_down_asymp)
    list_average_time_to_adapt_before_up_asymp.append(average_time_to_adapt_before_up_asymp)
    list_average_time_to_adapt_after_down_asymp.append(average_time_to_adapt_after_down_asymp)
    list_average_time_to_adapt_after_up_asymp.append(average_time_to_adapt_after_up_asymp)
    list_min_time_to_adapt_before_down_asymp.append(min_time_to_adapt_before_down_asymp)
    list_min_time_to_adapt_before_up_asymp.append(min_time_to_adapt_before_up_asymp)
    list_min_time_to_adapt_after_down_asymp.append(min_time_to_adapt_after_down_asymp)
    list_min_time_to_adapt_after_up_asymp.append(min_time_to_adapt_after_up_asymp)
    list_difference_time_to_adapt_before_after_down_1_asymp.append(difference_time_to_adapt_before_after_down_1_asymp)
    list_difference_time_to_adapt_before_after_down_2_asymp.append(difference_time_to_adapt_before_after_down_2_asymp)
    list_difference_time_to_adapt_before_after_down_3_asymp.append(difference_time_to_adapt_before_after_down_3_asymp)
    list_difference_time_to_adapt_before_after_up_1_asymp.append(difference_time_to_adapt_before_after_up_1_asymp)
    list_difference_time_to_adapt_before_after_up_2_asymp.append(difference_time_to_adapt_before_after_up_2_asymp)
    list_difference_time_to_adapt_before_after_up_3_asymp.append(difference_time_to_adapt_before_after_up_3_asymp)
    list_difference_min_time_to_adapt_before_after_up_asymp.append(difference_min_time_to_adapt_before_after_up_asymp)
    list_difference_min_time_to_adapt_before_after_down_asymp.append(difference_min_time_to_adapt_before_after_down_asymp)
    list_difference_average_time_to_adapt_before_after_up_asymp.append(difference_average_time_to_adapt_before_after_up_asymp)
    list_difference_average_time_to_adapt_before_after_down_asymp.append(difference_average_time_to_adapt_before_after_down_asymp)


dist_sd = {'ID'        : list_ID,
        'Group ID'  : list_ID_team,
        'Signal'    : list_signal,
        'Speed'     : list_speed,
        'Time_to_adapt_before_down_1'  : list_time_to_adapt_before_down_1_sd,
        'Time_to_adapt_before_down_2'  : list_time_to_adapt_before_down_2_sd,
        'Time_to_adapt_before_down_3'  : list_time_to_adapt_before_down_3_sd,
        'Average Time to adapt before down': list_average_time_to_adapt_before_down_sd,
        'Min Time to adapt before down': list_min_time_to_adapt_before_down_sd,
        'Time_to_adapt_before_up_1'  : list_time_to_adapt_before_up_1_sd,
        'Time_to_adapt_before_up_2'  : list_time_to_adapt_before_up_2_sd,
        'Time_to_adapt_before_up_3'  : list_time_to_adapt_before_up_3_sd,
        'Average Time to adapt before up': list_average_time_to_adapt_before_up_sd,
        'Min Time to adapt before up': list_min_time_to_adapt_before_up_sd,
        'Time_to_adapt_after_down_1'  : list_time_to_adapt_after_down_1_sd,
        'Time_to_adapt_after_down_2'  : list_time_to_adapt_after_down_2_sd,
        'Time_to_adapt_after_down_3'  : list_time_to_adapt_after_down_3_sd,
        'Average Time to adapt after down': list_average_time_to_adapt_after_down_sd,
        'Min Time to adapt after down': list_min_time_to_adapt_after_down_sd,
        'Time_to_adapt_after_up_1'  : list_time_to_adapt_after_up_1_sd,
        'Time_to_adapt_after_up_2'  : list_time_to_adapt_after_up_2_sd,
        'Time_to_adapt_after_up_3'  : list_time_to_adapt_after_up_3_sd,
        'Average Time to adapt after up': list_average_time_to_adapt_after_up_sd,
        'Min Time to adapt after up': list_min_time_to_adapt_after_up_sd,
        'Difference time to adapt before after down 1': list_difference_time_to_adapt_before_after_down_1_sd,
        'Difference time to adapt before after down 2': list_difference_time_to_adapt_before_after_down_2_sd,
        'Difference time to adapt before after down 3': list_difference_time_to_adapt_before_after_down_3_sd,
        'Difference time to adapt before after up 1': list_difference_time_to_adapt_before_after_up_1_sd,
        'Difference time to adapt before after up 2': list_difference_time_to_adapt_before_after_up_2_sd,
        'Difference time to adapt before after up 3': list_difference_time_to_adapt_before_after_up_3_sd,
        'Difference min time to adapt before after up': list_difference_min_time_to_adapt_before_after_up_sd,
        'Difference min time to adapt before after down': list_difference_min_time_to_adapt_before_after_down_sd,
        'Difference average time to adapt before after up': list_difference_average_time_to_adapt_before_after_up_sd,
        'Difference average time to adapt before after down': list_difference_average_time_to_adapt_before_after_down_sd,
        }

dist_asymp = {'ID'        : list_ID,
        'Group ID'  : list_ID_team,
        'Signal'    : list_signal,
        'Speed'     : list_speed,
        'Time_to_adapt_before_down_1'  : list_time_to_adapt_before_down_1_asymp,
        'Time_to_adapt_before_down_2'  : list_time_to_adapt_before_down_2_asymp,
        'Time_to_adapt_before_down_3'  : list_time_to_adapt_before_down_3_asymp,
        'Average Time to adapt before down': list_average_time_to_adapt_before_down_asymp,
        'Min Time to adapt before down': list_min_time_to_adapt_before_down_asymp,
        'Time_to_adapt_before_up_1'  : list_time_to_adapt_before_up_1_asymp,
        'Time_to_adapt_before_up_2'  : list_time_to_adapt_before_up_2_asymp,
        'Time_to_adapt_before_up_3'  : list_time_to_adapt_before_up_3_asymp,
        'Average Time to adapt before up': list_average_time_to_adapt_before_up_asymp,
        'Min Time to adapt before up': list_min_time_to_adapt_before_up_asymp,
        'Time_to_adapt_after_down_1'  : list_time_to_adapt_after_down_1_asymp,
        'Time_to_adapt_after_down_2'  : list_time_to_adapt_after_down_2_asymp,
        'Time_to_adapt_after_down_3'  : list_time_to_adapt_after_down_3_asymp,
        'Average Time to adapt after down': list_average_time_to_adapt_after_down_asymp,
        'Min Time to adapt after down': list_min_time_to_adapt_after_down_asymp,
        'Time_to_adapt_after_up_1'  : list_time_to_adapt_after_up_1_asymp,
        'Time_to_adapt_after_up_2'  : list_time_to_adapt_after_up_2_asymp,
        'Time_to_adapt_after_up_3'  : list_time_to_adapt_after_up_3_asymp,
        'Average Time to adapt after up': list_average_time_to_adapt_after_up_asymp,
        'Min Time to adapt after up': list_min_time_to_adapt_after_up_asymp,
        'Difference time to adapt before after down 1': list_difference_time_to_adapt_before_after_down_1_asymp,
        'Difference time to adapt before after down 2': list_difference_time_to_adapt_before_after_down_2_asymp,
        'Difference time to adapt before after down 3': list_difference_time_to_adapt_before_after_down_3_asymp,
        'Difference time to adapt before after up 1': list_difference_time_to_adapt_before_after_up_1_asymp,
        'Difference time to adapt before after up 2': list_difference_time_to_adapt_before_after_up_2_asymp,
        'Difference time to adapt before after up 3': list_difference_time_to_adapt_before_after_up_3_asymp,
        'Difference min time to adapt before after up': list_difference_min_time_to_adapt_before_after_up_asymp,
        'Difference min time to adapt before after down': list_difference_min_time_to_adapt_before_after_down_asymp,
        'Difference average time to adapt before after up': list_difference_average_time_to_adapt_before_after_up_asymp,
        'Difference average time to adapt before after down': list_difference_average_time_to_adapt_before_after_down_asymp,
        }

results_sd = pd.DataFrame(dist_sd)
results_asymp = pd.DataFrame(dist_asymp)


if Stylianos == True:
    directory_to_save = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results\Perturbation results'
else:
    directory_to_save = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Results\Perturbation results'

os.chdir(directory_to_save)
results_sd.to_excel(f'Sd Method Perturbation_results_{sd_factor}_sd_after_max_threshold.xlsx')
# results_asymp.to_excel(f'Asymptote Method Perturbation results {asymptote_fraction}.xlsx')
