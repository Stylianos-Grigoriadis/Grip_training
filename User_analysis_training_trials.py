import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
from Lib_grip import spatial_error
import glob


stylianos = True
if stylianos == True:
    directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\data to fix code'
else:
    directory_path = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Pilot study 4\Data'

files = glob.glob(os.path.join(directory_path, "*"))



list_avg_spatial_error_1 = []
list_avg_spatial_error_2 = []
list_avg_spatial_error_3 = []
list_avg_spatial_error_4 = []
list_avg_spatial_error_5 = []
list_avg_spatial_error_6 = []
list_avg_spatial_error_7 = []
list_avg_spatial_error_8 = []
list_avg_spatial_error_9 = []
list_avg_spatial_error_10 = []


list_ID = []
list_ID_team = []

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    list_ID.append(ID)
    ID_team = ID.split(".")
    list_ID_team.append(ID_team[0])
    print(ID) # We keep this so that we know which participant is assessed during the run of the code


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

    # Synchronise training trials
    synch_df1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df1)
    synch_df2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df2)
    synch_df3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df3)
    synch_df4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df4)
    synch_df5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df5)
    synch_df6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df6)
    synch_df7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df7)
    synch_df8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df8)
    synch_df9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df9)
    synch_df10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df10)

    # Calculation of spatial error for each trial
    spatial_errors1 = spatial_error(synch_df1)
    spatial_errors2 = spatial_error(synch_df2)
    spatial_errors3 = spatial_error(synch_df3)
    spatial_errors4 = spatial_error(synch_df4)
    spatial_errors5 = spatial_error(synch_df5)
    spatial_errors6 = spatial_error(synch_df6)
    spatial_errors7 = spatial_error(synch_df7)
    spatial_errors8 = spatial_error(synch_df8)
    spatial_errors9 = spatial_error(synch_df9)
    spatial_errors10 = spatial_error(synch_df10)

    # Calculation of mean spatial error for each trial
    avg_spatial_error_1 = np.mean(spatial_errors1)
    print(avg_spatial_error_1)
    avg_spatial_error_2 = np.mean(spatial_errors2)
    avg_spatial_error_3 = np.mean(spatial_errors3)
    avg_spatial_error_4 = np.mean(spatial_errors4)
    avg_spatial_error_5 = np.mean(spatial_errors5)
    avg_spatial_error_6 = np.mean(spatial_errors6)
    avg_spatial_error_7 = np.mean(spatial_errors7)
    avg_spatial_error_8 = np.mean(spatial_errors8)
    avg_spatial_error_9 = np.mean(spatial_errors9)
    avg_spatial_error_10 = np.mean(spatial_errors10)

    # Append lists of spatial errors
    list_avg_spatial_error_1.append(avg_spatial_error_1)
    print(type(list_avg_spatial_error_1[-1]))
    print(list_avg_spatial_error_1[-1])

    list_avg_spatial_error_2.append(avg_spatial_error_2)
    list_avg_spatial_error_3.append(avg_spatial_error_3)
    list_avg_spatial_error_4.append(avg_spatial_error_4)
    list_avg_spatial_error_5.append(avg_spatial_error_5)
    list_avg_spatial_error_6.append(avg_spatial_error_6)
    list_avg_spatial_error_7.append(avg_spatial_error_7)
    list_avg_spatial_error_8.append(avg_spatial_error_8)
    list_avg_spatial_error_9.append(avg_spatial_error_9)
    list_avg_spatial_error_10.append(avg_spatial_error_10)

    # Create a dictionery
dist = {
        'ID'        : list_ID,
        'Group ID'  : list_ID_team,
        'Mean Spatial error trail 1': list_avg_spatial_error_1,
        'Mean Spatial error trail 2': list_avg_spatial_error_2,
        'Mean Spatial error trail 3': list_avg_spatial_error_3,
        'Mean Spatial error trail 4': list_avg_spatial_error_4,
        'Mean Spatial error trail 5': list_avg_spatial_error_5,
        'Mean Spatial error trail 6': list_avg_spatial_error_6,
        'Mean Spatial error trail 7': list_avg_spatial_error_7,
        'Mean Spatial error trail 8': list_avg_spatial_error_8,
        'Mean Spatial error trail 9': list_avg_spatial_error_9,
        'Mean Spatial error trail 10': list_avg_spatial_error_10
        }
print(type(dist))

df_results = pd.DataFrame(dist)
#print(dist)

if stylianos == True:
    directory_to_save = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Results'
else:
    directory_to_save = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Pilot study 4\Results'

os.chdir(directory_to_save)
df_results.to_excel('Mean_spatial_error_results stylianos.xlsx')
