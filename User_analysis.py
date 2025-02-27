from pandas.core.array_algos.datetimelike_accumulations import cumsum

import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np

from Lib_grip import spatial_error

directory = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Pilot study 4\Data\White_65.1\Training_trials'
os.chdir(directory)

# data = pd.read_csv(, skiprows=2)
# print(data.columns)
# plt.plot(data['Performance'])
# plt.show()


df1 = pd.read_csv(r"White_65.1_trial_1.csv", skiprows=2)
df2 = pd.read_csv(r"White_65.1_trial_2.csv", skiprows=2)
df3 = pd.read_csv(r"White_65.1_trial_3.csv", skiprows=2)
df4 = pd.read_csv(r"White_65.1_trial_4.csv", skiprows=2)
df5 = pd.read_csv(r"White_65.1_trial_5.csv", skiprows=2)
df6 = pd.read_csv(r"White_65.1_trial_6.csv", skiprows=2)
df7 = pd.read_csv(r"White_65.1_trial_7.csv", skiprows=2)
df8 = pd.read_csv(r"White_65.1_trial_8.csv", skiprows=2)
df9 = pd.read_csv(r"White_65.1_trial_9.csv", skiprows=2)
df10 = pd.read_csv(r"White_65.1_trial_10.csv", skiprows=2)

pd.set_option('display.max_rows', None)

new_df1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df1)
new_df2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df2)
new_df3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df3)
new_df4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df4)
new_df5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df5)
new_df6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df6)
new_df7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df7)
new_df8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df8)
new_df9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df9)
new_df10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df10)

spatial_errors1 = spatial_error(new_df1)
spatial_errors2 = spatial_error(new_df2)
spatial_errors3 = spatial_error(new_df3)
spatial_errors4 = spatial_error(new_df4)
spatial_errors5 = spatial_error(new_df5)
spatial_errors6 = spatial_error(new_df6)
spatial_errors7 = spatial_error(new_df7)
spatial_errors8 = spatial_error(new_df8)
spatial_errors9 = spatial_error(new_df9)
spatial_errors10 = spatial_error(new_df10)


combined_spatial_errors = list(itertools.chain(spatial_errors1, spatial_errors2, spatial_errors3, spatial_errors4, spatial_errors5, spatial_errors6, spatial_errors7, spatial_errors8, spatial_errors9, spatial_errors10))

# Finding the length of each list of spatial errors
spatial_error_lengths = [len(spatial_errors1), len(spatial_errors2), len(spatial_errors3), len(spatial_errors4), len(spatial_errors5),len(spatial_errors6), len(spatial_errors7), len(spatial_errors8), len(spatial_errors9), len(spatial_errors10)]
print(spatial_error_lengths)

# Exclude last cumulative sum as it's the total length
spatial_error_indices = np.cumsum(spatial_error_lengths)[:-1]

# Plot the combined spatial errors
plt.plot(combined_spatial_errors, label='Spatial Errors', linewidth=1)
# Add red vertical lines at the transition points
for idx in spatial_error_indices:
    plt.axvline(x=idx, color='black', linestyle='--', linewidth=1, label="Trial Boundary" if idx == spatial_error_indices[0] else "")

plt.xlabel("Targets")
plt.ylabel("Spatial Error")
plt.title("Spatial Errors Across Trials")
plt.legend()
plt.show()