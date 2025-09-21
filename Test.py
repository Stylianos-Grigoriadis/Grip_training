import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
import glob

# directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Data\White_100.6\Isometric_trials'
# os.chdir(directory)
# files = glob.glob(os.path.join(directory, "*"))
# for file in files:
#     data = pd.read_csv(file, skiprows=2)
#     ID = os.path.basename(file)
#     print(ID)
#     print(data)
#     plt.plot(data['Time'], data['Performance'], label='Player')
#     plt.plot(data['ClosestSampleTime'], data['Target'], label='Target')
#     plt.legend()
#     plt.show()

# Isometric_trial_10
# Isometric_trial_50


directory_1 = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Signals\Sine_65.6_signals\Training_sine_65.6_T5.txt'
directory_2 = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Signals\Sine_100.3_signals\Training_sine_100.3_T3.txt'

Sine_65 = lb.read_my_txt_file(directory_1)
Sine_100 = lb.read_my_txt_file(directory_2)
plt.plot(Sine_65, label='Sine_65')
plt.plot(Sine_100, label='Sine_100')
plt.legend()
plt.show()

