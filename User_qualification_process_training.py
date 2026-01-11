import pandas as pd
import numpy as np
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os

Stylianos = True
if Stylianos == True:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Valid data\Force data'
else:
    directory = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Data\Valid data\Force data'

files = glob.glob(os.path.join(directory, "*"))

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    print(ID)

    file_training_trials = file + r'\Training_trials'
    os.chdir(file_training_trials)


    training_set_1 = pd.read_csv(r'Trial_1.csv', skiprows=2)
    training_set_2 = pd.read_csv(r'Trial_2.csv', skiprows=2)
    training_set_3 = pd.read_csv(r'Trial_3.csv', skiprows=2)
    training_set_4 = pd.read_csv(r'Trial_4.csv', skiprows=2)
    training_set_5 = pd.read_csv(r'Trial_5.csv', skiprows=2)
    training_set_6 = pd.read_csv(r'Trial_6.csv', skiprows=2)
    training_set_7 = pd.read_csv(r'Trial_7.csv', skiprows=2)
    training_set_8 = pd.read_csv(r'Trial_8.csv', skiprows=2)
    training_set_9 = pd.read_csv(r'Trial_9.csv', skiprows=2)
    training_set_10 = pd.read_csv(r'Trial_10.csv', skiprows=2)

    training_set_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_1)
    training_set_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_2)
    training_set_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_3)
    training_set_4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_4)
    training_set_5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_5)
    training_set_6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_6)
    training_set_7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_7)
    training_set_8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_8)
    training_set_9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_9)
    training_set_10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_set_10)



    list_set = [training_set_1,
            training_set_2,
            training_set_3,
            training_set_4,
            training_set_5,
            training_set_6,
            training_set_7,
            training_set_8,
            training_set_9,
            training_set_10
            ]
    list_average_difference_time_to_ClosestSampleTime = []
    list_std_difference_time_to_ClosestSampleTime = []
    very_high_time_difference = False
    set_number_with_high_time_lag = []
    for index, set in enumerate(list_set):
        force_time = set['Time'].to_numpy()
        target_time = set['ClosestSampleTime'].to_numpy()
        difference = force_time-target_time
        average = np.average(np.abs(difference))
        std = np.std(np.abs(difference))
        if average > 0.01:
            very_high_time_difference = True
            set_number_with_high_time_lag.append(index+1)
        # print(f'Average is {average}')
        # print(f'Std is {std}')
        list_average_difference_time_to_ClosestSampleTime.append(average)
        list_std_difference_time_to_ClosestSampleTime.append(std)
    if very_high_time_difference == True:

        list_average_difference_time_to_ClosestSampleTime = np.array(list_average_difference_time_to_ClosestSampleTime)
        list_std_difference_time_to_ClosestSampleTime = np.array(list_std_difference_time_to_ClosestSampleTime)
        upper = list_average_difference_time_to_ClosestSampleTime + list_std_difference_time_to_ClosestSampleTime
        lower = list_average_difference_time_to_ClosestSampleTime - list_std_difference_time_to_ClosestSampleTime

        x = np.arange(len(list_average_difference_time_to_ClosestSampleTime))
        plt.plot(list_average_difference_time_to_ClosestSampleTime, label='Average Difference', linewidth=2)
        plt.fill_between(x, lower, upper, alpha=0.3, label='±SD')
        labels = [f"Set {i+1}" for i in range(len(list_average_difference_time_to_ClosestSampleTime))]
        plt.xlabel('Time')
        plt.ylabel('Difference')
        plt.title('Difference to ClosestSampleTime with ±SD')
        plt.legend(loc='upper left')
        plt.xticks(x, labels, rotation=45)  # Set custom labels

        plt.show()
        raise ValueError(f"The asynchrony between the Performance time and the target time is higher than 0.1 on the Set(s): {set_number_with_high_time_lag}")

    fig, axes = plt.subplots(5, 2, figsize=(12, 6), constrained_layout=True)
    axes = axes.flatten()  # Make indexing easier

    for i, ax in enumerate(axes):
        set = list_set[i]

        ax.plot(set['Time'], set['Performance'], label='Force Output')
        ax.plot(set['ClosestSampleTime'], set['Target'], label='Target')

        ax.set_title(f"Set {i + 1}")
        ax.legend(fontsize=8)

    plt.show()