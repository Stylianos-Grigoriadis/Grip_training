import Lib_grip as lb
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 2\Data\AKIS'
os.chdir(directory)

# data = pd.read_csv(r'grip_Karagiorgos_Akis__06Φεβ25_10_49_38.csv', skiprows=2)
# print(data.columns)
# plt.plot(data['Performance'])
# plt.show()








df1 = pd.read_csv(r"grip_Karagiorgos_Akis__06Φεβ25_10_37_03.csv", skiprows=2)
pd.set_option('display.max_rows', None)



new_df1_anestis = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df1)
new_df1_Stylianos = lb.synchronization_of_Time_and_ClosestSampleTime_Stylianos(df1, 100)

new_df1_anestis.to_excel(r'Anestis_pert.xlsx')
new_df1_Stylianos.to_excel(r'Stylianos_pert.xlsx')
sd = 2
consecutive_values = 37
first_values = 100
total_targets = 500

perturbation = lb.adaptation_time_using_sd(df1, sd, first_values, consecutive_values, 'Akis', plot=True)




# combined_spatial_errors = list(itertools.chain(spatial_errors1, spatial_errors2, spatial_errors3, spatial_errors4, spatial_errors5, spatial_errors6, spatial_errors7, spatial_errors8, spatial_errors9, spatial_errors10))
#plt.plot(integrate_signal(combined_spatial_errors))
#plt.show()