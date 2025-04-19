import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
import glob


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Data'
files = glob.glob(os.path.join(directory_path, "*"))

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    ID_team = ID[:-2]
    print(ID)

    # Isometric trials analysis
    file_isometrics = file+r'\Isometric_trials'
    os.chdir(file_isometrics)

    Isometric_trial_10 = pd.read_csv(r'Isometric_trial_10.csv', skiprows=2)
    Isometric_trial_50 = pd.read_csv(r'Isometric_trial_50.csv', skiprows=2)
    # print(Isometric_trial_10)
    print(Isometric_trial_10.columns)
    plt.plot(Isometric_trial_10['Time'], Isometric_trial_10['Performance'], label='Player')
    plt.plot(Isometric_trial_10['ClosestSampleTime'], Isometric_trial_10['Target'], label='Target')
    plt.legend()
    plt.show()