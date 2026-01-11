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

def RMS(original, filtered):
    residual = original - filtered
    rms = np.sqrt(np.mean(residual ** 2))
    return rms


stylianos = True
if stylianos == True:
    directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Valid data\Force data'
else:
    directory_path = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Data\Valid data\Force data'

files = glob.glob(os.path.join(directory_path, "*"))



list_ID = []
list_ID_team = []

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    list_ID.append(ID)
    ID_team = ID.split(".")
    print(ID_team)
    list_ID_team.append(ID_team[0])
    print(ID) # We keep this so that we know which participant is assessed during the run of the code

    # Creation of lists of rms for each participant
    list_rms_1 = []
    list_rms_2 = []
    list_rms_3 = []
    list_rms_4 = []
    list_rms_5 = []
    list_rms_6 = []
    list_rms_7 = []
    list_rms_8 = []
    list_rms_9 = []
    list_rms_10 = []

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

    print(df1.columns)

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

    list_of_frequencies = range(2,30)
    for low_pass_filter_frequency in  list_of_frequencies:

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

        rms_1 = RMS(Performance_1, Performance_1_filtered)
        rms_2 = RMS(Performance_2, Performance_2_filtered)
        rms_3 = RMS(Performance_3, Performance_3_filtered)
        rms_4 = RMS(Performance_4, Performance_4_filtered)
        rms_5 = RMS(Performance_5, Performance_5_filtered)
        rms_6 = RMS(Performance_6, Performance_6_filtered)
        rms_7 = RMS(Performance_7, Performance_7_filtered)
        rms_8 = RMS(Performance_8, Performance_8_filtered)
        rms_9 = RMS(Performance_9, Performance_9_filtered)
        rms_10 = RMS(Performance_10, Performance_10_filtered)

        list_rms_1.append(rms_1)
        list_rms_2.append(rms_2)
        list_rms_3.append(rms_3)
        list_rms_4.append(rms_4)
        list_rms_5.append(rms_5)
        list_rms_6.append(rms_6)
        list_rms_7.append(rms_7)
        list_rms_8.append(rms_8)
        list_rms_9.append(rms_9)
        list_rms_10.append(rms_10)

    plt.plot(list_of_frequencies, list_rms_1, label='Set 1')
    plt.plot(list_of_frequencies, list_rms_2, label='Set 2')
    plt.plot(list_of_frequencies, list_rms_3, label='Set 3')
    plt.plot(list_of_frequencies, list_rms_4, label='Set 4')
    plt.plot(list_of_frequencies, list_rms_5, label='Set 5')
    plt.plot(list_of_frequencies, list_rms_6, label='Set 6')
    plt.plot(list_of_frequencies, list_rms_7, label='Set 7')
    plt.plot(list_of_frequencies, list_rms_8, label='Set 8')
    plt.plot(list_of_frequencies, list_rms_9, label='Set 9')
    plt.plot(list_of_frequencies, list_rms_10, label='Set 10')
    plt.legend()
    plt.title(f'{ID}')
    plt.xlabel("Cutoff Frequency (Hz)")
    plt.ylabel("RMS Residual Error")
    plt.grid(True)
    plt.ylim(-0.05, 0.85)

    save_dir = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results\Residual analysis\Isometric trials"
    save_path = os.path.join(save_dir, f"{ID}_rms_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()








