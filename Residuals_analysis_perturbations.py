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
    list_rms_perturbation_before_down_1 = []
    list_rms_perturbation_before_down_2 = []
    list_rms_perturbation_before_down_3 = []
    list_rms_perturbation_before_up_1 = []
    list_rms_perturbation_before_up_2 = []
    list_rms_perturbation_before_up_3 = []
    list_rms_perturbation_after_down_1 = []
    list_rms_perturbation_after_down_2 = []
    list_rms_perturbation_after_down_3 = []
    list_rms_perturbation_after_up_1 = []
    list_rms_perturbation_after_up_2 = []
    list_rms_perturbation_after_up_3 = []


    # Trial analysis for calculation of mean and sd of spatial error
    file_Perturbation_trials= file + r'\Perturbation_trials\before'
    os.chdir(file_Perturbation_trials)
    perturbation_before_down_1 = pd.read_csv(r'pertr_down_1.csv', skiprows=2)
    perturbation_before_down_2 = pd.read_csv(r'pertr_down_2.csv', skiprows=2)
    perturbation_before_down_3 = pd.read_csv(r'pertr_down_3.csv', skiprows=2)
    perturbation_before_up_1 = pd.read_csv(r'pertr_up_1.csv', skiprows=2)
    perturbation_before_up_2 = pd.read_csv(r'pertr_up_2.csv', skiprows=2)
    perturbation_before_up_3 = pd.read_csv(r'pertr_up_3.csv', skiprows=2)

    file_Perturbation_trials= file + r'\Perturbation_trials\after'
    os.chdir(file_Perturbation_trials)
    perturbation_after_down_1 = pd.read_csv(r'pertr_down_1.csv', skiprows=2)
    perturbation_after_down_2 = pd.read_csv(r'pertr_down_2.csv', skiprows=2)
    perturbation_after_down_3 = pd.read_csv(r'pertr_down_3.csv', skiprows=2)
    perturbation_after_up_1 = pd.read_csv(r'pertr_up_1.csv', skiprows=2)
    perturbation_after_up_2 = pd.read_csv(r'pertr_up_2.csv', skiprows=2)
    perturbation_after_up_3 = pd.read_csv(r'pertr_up_3.csv', skiprows=2)

    # Filtering process
    perturbation_before_down_1 = perturbation_before_down_1['Performance']
    perturbation_before_down_2 = perturbation_before_down_2['Performance']
    perturbation_before_down_3 = perturbation_before_down_3['Performance']
    perturbation_before_up_1 = perturbation_before_up_1['Performance']
    perturbation_before_up_2 = perturbation_before_up_2['Performance']
    perturbation_before_up_3 = perturbation_before_up_3['Performance']
    perturbation_after_down_1 = perturbation_after_down_1['Performance']
    perturbation_after_down_2 = perturbation_after_down_2['Performance']
    perturbation_after_down_3 = perturbation_after_down_3['Performance']
    perturbation_after_up_1 = perturbation_after_up_1['Performance']
    perturbation_after_up_2 = perturbation_after_up_2['Performance']
    perturbation_after_up_3 = perturbation_after_up_3['Performance']


    list_of_frequencies = range(2,30)
    for low_pass_filter_frequency in list_of_frequencies:
        perturbation_before_down_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_before_down_1)
        perturbation_before_down_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_before_down_2)
        perturbation_before_down_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_before_down_3)
        perturbation_before_up_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_before_up_1)
        perturbation_before_up_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_before_up_2)
        perturbation_before_up_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_before_up_3)
        perturbation_after_down_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_after_down_1)
        perturbation_after_down_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_after_down_2)
        perturbation_after_down_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_after_down_3)
        perturbation_after_up_1_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_after_up_1)
        perturbation_after_up_2_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_after_up_2)
        perturbation_after_up_3_filtered = lib.Butterworth(75, low_pass_filter_frequency, perturbation_after_up_3)

        rms_perturbation_before_down_1 = RMS(perturbation_before_down_1, perturbation_before_down_1_filtered)
        rms_perturbation_before_down_2 = RMS(perturbation_before_down_2, perturbation_before_down_2_filtered)
        rms_perturbation_before_down_3 = RMS(perturbation_before_down_3, perturbation_before_down_3_filtered)
        rms_perturbation_before_up_1 = RMS(perturbation_before_up_1, perturbation_before_up_1_filtered)
        rms_perturbation_before_up_2 = RMS(perturbation_before_up_2, perturbation_before_up_2_filtered)
        rms_perturbation_before_up_3 = RMS(perturbation_before_up_3, perturbation_before_up_3_filtered)
        rms_perturbation_after_down_1 = RMS(perturbation_after_down_1, perturbation_after_down_1_filtered)
        rms_perturbation_after_down_2 = RMS(perturbation_after_down_2, perturbation_after_down_2_filtered)
        rms_perturbation_after_down_3 = RMS(perturbation_after_down_3, perturbation_after_down_3_filtered)
        rms_perturbation_after_up_1 = RMS(perturbation_after_up_1, perturbation_after_up_1_filtered)
        rms_perturbation_after_up_2 = RMS(perturbation_after_up_2, perturbation_after_up_2_filtered)
        rms_perturbation_after_up_3 = RMS(perturbation_after_up_3, perturbation_after_up_3_filtered)

        list_rms_perturbation_before_down_1.append(rms_perturbation_before_down_1)
        list_rms_perturbation_before_down_2.append(rms_perturbation_before_down_2)
        list_rms_perturbation_before_down_3.append(rms_perturbation_before_down_3)
        list_rms_perturbation_before_up_1.append(rms_perturbation_before_up_1)
        list_rms_perturbation_before_up_2.append(rms_perturbation_before_up_2)
        list_rms_perturbation_before_up_3.append(rms_perturbation_before_up_3)
        list_rms_perturbation_after_down_1.append(rms_perturbation_after_down_1)
        list_rms_perturbation_after_down_2.append(rms_perturbation_after_down_2)
        list_rms_perturbation_after_down_3.append(rms_perturbation_after_down_3)
        list_rms_perturbation_after_up_1.append(rms_perturbation_after_up_1)
        list_rms_perturbation_after_up_2.append(rms_perturbation_after_up_2)
        list_rms_perturbation_after_up_3.append(rms_perturbation_after_up_3)

    plt.plot(list_of_frequencies, list_rms_perturbation_before_down_1, label='before_down_1')
    plt.plot(list_of_frequencies, list_rms_perturbation_before_down_2, label='before_down_2')
    plt.plot(list_of_frequencies, list_rms_perturbation_before_down_3, label='before_down_3')
    plt.plot(list_of_frequencies, list_rms_perturbation_before_up_1, label='before_up_1')
    plt.plot(list_of_frequencies, list_rms_perturbation_before_up_2, label='before_up_2')
    plt.plot(list_of_frequencies, list_rms_perturbation_before_up_3, label='before_up_3')
    plt.plot(list_of_frequencies, list_rms_perturbation_after_down_1, label='after_down_1', ls='--')
    plt.plot(list_of_frequencies, list_rms_perturbation_after_down_2, label='after_down_2', ls='--')
    plt.plot(list_of_frequencies, list_rms_perturbation_after_down_3, label='after_down_3', ls='--')
    plt.plot(list_of_frequencies, list_rms_perturbation_after_up_1, label='after_up_1', ls='--')
    plt.plot(list_of_frequencies, list_rms_perturbation_after_up_2, label='after_up_2', ls='--')
    plt.plot(list_of_frequencies, list_rms_perturbation_after_up_3, label='after_up_3', ls='--')

    plt.legend()
    plt.title(f'{ID}')
    plt.xlabel("Cutoff Frequency (Hz)")
    plt.ylabel("RMS Residual Error")
    plt.grid(True)
    plt.ylim(-0.05, 0.85)
    save_dir = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results\Residual analysis\Perturbation"
    save_path = os.path.join(save_dir, f"{ID}_rms_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()








