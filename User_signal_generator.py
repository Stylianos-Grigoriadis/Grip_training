import matplotlib.pyplot as plt
import Lib_grip as lb
import numpy as np



# Define parameters for signals creation
num_points_65 = 65
num_points_100 = 100
num_periods_100 = 22
num_periods_65 = num_periods_100 * 0.65
desired_sd = 15
desired_average = 50

# Create the signals
# sine_65 = lb.sine_wave_signal_creation(200, desired_sd, desired_average, num_periods_65)
# white_65 = lb.white_noise_signal_creation(num_points_65, desired_sd, desired_average)
# pink_65 = lb.pink_noise_signal_creation_using_cn(num_points_65, desired_sd, desired_average)
# sine_100 = lb.sine_wave_signal_creation(200, desired_sd, desired_average, num_periods_100)
# white_100 = lb.white_noise_signal_creation(num_points_100, desired_sd, desired_average)
# pink_100 = lb.pink_noise_signal_creation_using_cn(num_points_100, desired_sd, desired_average)

# # Print the outputs of the signals
# print('Signals of 65')
# lb.outputs(white_65, pink_65, sine_65)
# print('Signals of 100')
# lb.outputs(white_100, pink_100, sine_100)

for i in range(1, 11):
    pink_65 = lb.pink_noise_signal_creation_using_cn(num_points_65, desired_sd, desired_average)
    # lb.create_txt_file(pink_65, f'Training_pink_65.1_T{i}', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')

isometric_10_perc_MVC = lb.isometric_generator_single_rep(300, 16.6)
isometric_50_perc_MVC = lb.isometric_generator_single_rep(300, 83.3)
Pert_up_T1 = lb.perturbation_single_trial_with_random_change(500, 50, 83.3)
Pert_up_T2 = lb.perturbation_single_trial_with_random_change(500, 50, 83.3)
Pert_up_T3 = lb.perturbation_single_trial_with_random_change(500, 50, 83.3)
Pert_down_T1 = lb.perturbation_single_trial_with_random_change(500, 50, 16.6)
Pert_down_T2 = lb.perturbation_single_trial_with_random_change(500, 50, 16.6)
Pert_down_T3 = lb.perturbation_single_trial_with_random_change(500, 50, 16.6)


# lb.create_txt_file(isometric_10_perc_MVC, f'isometric_10_perc_MVC', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(isometric_50_perc_MVC, f'isometric_50_perc_MVC', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(Pert_up_T1, f'Pert_up_T1', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(Pert_up_T2, f'Pert_up_T2', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(Pert_up_T3, f'Pert_up_T3', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(Pert_down_T1, f'Pert_down_T1', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(Pert_down_T2, f'Pert_down_T2', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')
# lb.create_txt_file(Pert_down_T3, f'Pert_down_T3', r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 3\Signals\Pink_65.1')

# Hello Stylianos







