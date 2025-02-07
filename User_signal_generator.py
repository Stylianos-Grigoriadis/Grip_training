import matplotlib.pyplot as plt
import Lib_grip as lb


# Define parameters for signals creation
num_points_65 = 65
num_points_100 = 100
num_periods_100 = 22
num_periods_65 = num_periods_100 * 0.65
desired_sd = 15
desired_average = 50

# Create the signals
sine_65 = lb.sine_wave_signal_creation(200, desired_sd, desired_average, num_periods_65)
white_65 = lb.white_noise_signal_creation(num_points_65, desired_sd, desired_average)
pink_65 = lb.pink_noise_signal_creation_using_cn(num_points_65, desired_sd, desired_average)
sine_100 = lb.sine_wave_signal_creation(200, desired_sd, desired_average, num_periods_100)
white_100 = lb.white_noise_signal_creation(num_points_100, desired_sd, desired_average)
pink_100 = lb.pink_noise_signal_creation_using_cn(num_points_100, desired_sd, desired_average)

# Print the outputs of the signals
print('Signals of 65')
lb.outputs(white_65, pink_65, sine_65)
print('Signals of 100')
lb.outputs(white_100, pink_100, sine_100)
