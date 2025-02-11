from pandas.core.array_algos.datetimelike_accumulations import cumsum

import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np

from Lib_grip import spatial_error

directory = r'C:\Users\USER\Desktop\Biomechanincs\projects\Grip Strength\pilot trials\Pink_65.1\trials'
os.chdir(directory)

# data = pd.read_csv(, skiprows=2)
# print(data.columns)
# plt.plot(data['Performance'])
# plt.show()


df1 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_13_33.csv", skiprows=2)
df2 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_16_00.csv", skiprows=2)
df3 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_18_44.csv", skiprows=2)
df4 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_21_08.csv", skiprows=2)
df5 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_23_17.csv", skiprows=2)
df6 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_25_53.csv", skiprows=2)
df7 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_28_12.csv", skiprows=2)
df8 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_30_30.csv", skiprows=2)
df9 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_32_48.csv", skiprows=2)
df10 = pd.read_csv(r"grip_Makri_Stephani__11Φεβ25_12_35_23.csv", skiprows=2)

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




#new_df1_anestis = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df1)
#new_df1_Stylianos = lb.synchronization_of_Time_and_ClosestSampleTime_Stylianos(df1, 100)
#
#new_df1_anestis.to_excel(r'Anestis_pert.xlsx')
#new_df1_Stylianos.to_excel(r'Stylianos_pert.xlsx')
#sd = 2
#consecutive_values = 37
#first_values = 100
#total_targets = 500
#
#perturbation = lb.adaptation_time_using_sd(df1, sd, first_values, consecutive_values, 'Akis', plot=True)


combined_spatial_errors = list(itertools.chain(spatial_errors1, spatial_errors2, spatial_errors3, spatial_errors4, spatial_errors5, spatial_errors6, spatial_errors7, spatial_errors8, spatial_errors9, spatial_errors10))
cumsum_combined_spatial_errors = np.cumsum(combined_spatial_errors)
plt.plot(cumsum_combined_spatial_errors)
plt.show()