import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorednoise as cn
import Lib_grip as lb
from scipy.signal import welch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

def outputs(white, pink, sine):
    white_average = np.mean(white)
    pink_average = np.mean(pink)
    sine_average = np.mean(sine)
    list_average = [white_average, pink_average, sine_average]

    white_std = np.std(white)
    pink_std = np.std(pink)
    sine_std = np.std(sine)
    list_std = [white_std, pink_std, sine_std]

    white_total_load = np.cumsum(white)[-1]
    pink_total_load = np.cumsum(pink)[-1]
    sine_total_load = np.cumsum(sine)[-1]
    list_total_load = [white_total_load, pink_total_load, sine_total_load]

    dist = {'Signals': ['White', 'Pink', 'Sine'],
            'Average' : list_average,
            'std' : list_std,
            'Total_load' : list_total_load}
    df = pd.DataFrame(dist)
    print(df)

def change_sd_and_average(signal, desired_sd, desired_average):
    sd_signal = np.std(signal)
    signal = signal * desired_sd / sd_signal
    average_signal = np.mean(signal)
    signal = signal + desired_average - average_signal

    return signal

def z_transform(signal, desired_sd, desired_average):
    average = np.mean(signal)
    sd = np.std(signal)
    standarized_signal = (signal - average) / sd
    transformed_signal = standarized_signal * desired_sd + desired_average

    return transformed_signal

def sine_wave_creation(N, Number_of_periods):
    frequency = Number_of_periods * 2
    sampling_rate = N
    t = np.linspace(0, N / sampling_rate, N)
    sine_wave = np.sin(np.pi * frequency * t)

    return sine_wave

def seperate_signal(signal, sets):
    pass
# desired_sd = 20
# desired_average = 50
#
# pink_max = 101
# pink_min = -1
# times_of_iteration_pink = 0
# while pink_max > 100 or pink_min < 0:
#     pink_noise = cn.powerlaw_psd_gaussian(1, num_points)
#     pink_noise_final = change_sd_and_average(pink_noise, desired_sd, desired_average)
#     pink_max = np.max(pink_noise_final)
#     pink_min = np.min(pink_noise_final)
#     times_of_iteration_pink += 1
#     print(times_of_iteration_pink)
#
# white_max = 101
# white_min = -1
# times_of_iteration_white = 0
# while white_max > 100 or white_min < 0:
#     white_noise = np.random.normal(0, 1, num_points)
#     white_noise_final = change_sd_and_average(white_noise, desired_sd, desired_average)
#     white_max = np.max(white_noise_final)
#     white_min = np.min(white_noise_final)
#     times_of_iteration_white += 1
#     print(times_of_iteration_white)
# sine_wave_final = change_sd_and_average(sine_wave, desired_sd, desired_average)
#
#
# print('***Before conversion')
# outputs(white_noise, pink_noise, sine_wave)
# print('***After conversion')
# outputs(white_noise_final, pink_noise_final, sine_wave_final)
#
# plt.plot(white_noise_final, label='white_noise', c='grey')
# plt.plot(pink_noise_final, label='pink_noise', c='pink')
# plt.plot(sine_wave_final, label='sine_wave', c='red')
# plt.ylim(-5,105)
# plt.legend()
# plt.show()

num_points = 65

white_signal = white_noise = np.random.normal(0, 1, num_points)
pink_noise = cn.powerlaw_psd_gaussian(1, num_points)
sine_wave = sine_wave_creation(num_points, 5)


desired_sd = 15
desired_average = 50

white_signal_z_ = z_transform(white_signal, desired_sd, desired_average)
pink_noise_z_ = z_transform(pink_noise, desired_sd, desired_average)
sine_wave_z_ = z_transform(sine_wave, desired_sd, desired_average)

print('***Before conversion')
outputs(white_noise, pink_noise, sine_wave)
print('***After z transformation')
outputs(white_signal_z_, pink_noise_z_, sine_wave_z_)


plt.plot(white_signal_z_, label='white_signal_z_', c='gray', lw=3)
plt.plot(pink_noise_z_, label='pink_noise_z_', c='pink', lw=3)
plt.plot(sine_wave_z_, label='sine_wave_z_', c='red', lw=3)
plt.legend()
plt.show()

frequencies, psd = welch(pink_noise_z_, 1, nperseg=1024)
# Convert to log-log scale for fitting
log_freq = np.log10(frequencies[1:])  # Skip zero frequency
log_psd = np.log10(psd[1:])

# Fit a linear model to the log-log data
slope, intercept = np.polyfit(log_freq, log_psd, 1)
best_fit_line = slope * log_freq + intercept

print(f"Estimated slope of PSD: {slope:.2f}")
if -1.2 < slope < -0.8:
    print("The signal is likely pink noise.")
elif slope < -1.8:
    print("The signal might be brown/red noise.")
elif slope > -0.5:
    print("The signal might be white noise.")
else:
    print("The signal doesn't match typical noise patterns.")

# Plot the PSD and the best-fit line
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, psd, label="PSD", linewidth=2)
plt.loglog(10**log_freq, 10**best_fit_line, '--', label=f"Best-fit line (slope={slope:.2f})", color='red')
plt.title("Power Spectral Density (PSD) with Best-Fit Line")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
