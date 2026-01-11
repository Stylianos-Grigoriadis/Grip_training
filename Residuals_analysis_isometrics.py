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
    file_training_trials = file + r'\Isometric_trials'
    os.chdir(file_training_trials)

    df1 = pd.read_csv(r'Isometric_trial_10.csv', skiprows=2)
    df2 = pd.read_csv(r'Isometric_trial_50.csv', skiprows=2)


    print(df1.columns)

    # Filtering process
    Performance_1 = df1['Performance']
    Performance_2 = df2['Performance']

    list_of_frequencies = range(2,30)
    for low_pass_filter_frequency in  list_of_frequencies:

        Performance_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_1)
        Performance_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, Performance_2)


        rms_1 = RMS(Performance_1, Performance_1_filtered)
        rms_2 = RMS(Performance_2, Performance_2_filtered)


        list_rms_1.append(rms_1)
        list_rms_2.append(rms_2)


    plt.plot(list_of_frequencies, list_rms_1, label='Isometric trial 10')
    plt.plot(list_of_frequencies, list_rms_2, label='Isometric trial 50')

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








