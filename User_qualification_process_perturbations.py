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

    file_perturbation_before_trials = file + r'\Perturbation_trials\before'
    os.chdir(file_perturbation_before_trials)
    pertr_before_up_1 = pd.read_csv(r'pertr_up_1.csv', skiprows=2)
    pertr_before_up_2 = pd.read_csv(r'pertr_up_2.csv', skiprows=2)
    pertr_before_up_3 = pd.read_csv(r'pertr_up_3.csv', skiprows=2)
    pertr_before_down_1 = pd.read_csv(r'pertr_down_1.csv', skiprows=2)
    pertr_before_down_2 = pd.read_csv(r'pertr_down_2.csv', skiprows=2)
    pertr_before_down_3 = pd.read_csv(r'pertr_down_3.csv', skiprows=2)

    file_perturbation_after_trials = file + r'\Perturbation_trials\after'
    os.chdir(file_perturbation_after_trials)
    pertr_after_up_1 = pd.read_csv(r'pertr_up_1.csv', skiprows=2)
    pertr_after_up_2 = pd.read_csv(r'pertr_up_2.csv', skiprows=2)
    pertr_after_up_3 = pd.read_csv(r'pertr_up_3.csv', skiprows=2)
    pertr_after_down_1 = pd.read_csv(r'pertr_down_1.csv', skiprows=2)
    pertr_after_down_2 = pd.read_csv(r'pertr_down_2.csv', skiprows=2)
    pertr_after_down_3 = pd.read_csv(r'pertr_down_3.csv', skiprows=2)

    pertr_before_up_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_before_up_1)
    pertr_before_up_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_before_up_2)
    pertr_before_up_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_before_up_3)
    pertr_before_down_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_before_down_1)
    pertr_before_down_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_before_down_2)
    pertr_before_down_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_before_down_3)
    pertr_after_up_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_after_up_1)
    pertr_after_up_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_after_up_2)
    pertr_after_up_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_after_up_3)
    pertr_after_down_1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_after_down_1)
    pertr_after_down_2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_after_down_2)
    pertr_after_down_3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(pertr_after_down_3)



    list_set = [pertr_before_up_1, pertr_after_up_1,
                pertr_before_up_2, pertr_after_up_2,
                pertr_before_up_3, pertr_after_up_3,
                pertr_before_down_1, pertr_after_down_1,
                pertr_before_down_2, pertr_after_down_2,
                pertr_before_down_3, pertr_after_down_3
                        ]
    list_names = ['pertr_before_up_1', 'pertr_after_up_1',
                'pertr_before_up_2', 'pertr_after_up_2',
                'pertr_before_up_3', 'pertr_after_up_3',
                'pertr_before_down_1', 'pertr_after_down_1',
                'pertr_before_down_2', 'pertr_after_down_2',
                'pertr_before_down_3', 'pertr_after_down_3'
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
        plt.xlabel('Time')
        plt.ylabel('Difference')
        plt.title('Difference to ClosestSampleTime with ±SD')
        plt.legend(loc='upper left')
        plt.xticks(x, list_names, rotation=45)  # Set custom labels

        plt.show()
        raise ValueError(f"The asynchrony between the Performance time and the target time is higher than 0.1 on the Set(s): {set_number_with_high_time_lag}")

    fig, axes = plt.subplots(6, 2, figsize=(12, 6), constrained_layout=True)
    axes = axes.flatten()  # Make indexing easier

    for i, ax in enumerate(axes):
        set = list_set[i]

        ax.plot(set['Time'], set['Performance'], label='Force Output')
        ax.plot(set['ClosestSampleTime'], set['Target'], label='Target')

        ax.set_title(list_names[i])
        ax.legend(fontsize=8)

    plt.show()