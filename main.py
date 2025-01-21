import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorednoise as cn
import Lib_grip as lb
from scipy.signal import welch
from scipy import stats
from itertools import chain
from scipy.stats import pearsonr


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

def z_transform(signal, desired_sd, desired_average):
    average = np.mean(signal)
    sd = np.std(signal)
    standarized_signal = (signal - average) / sd
    transformed_signal = standarized_signal * desired_sd + desired_average

    return transformed_signal

def sine_wave_signal_creation(N, desired_sd, desired_average, Number_of_periods):
    frequency = Number_of_periods * 2
    t = np.linspace(0, 1, N)
    sine_wave = np.sin(np.pi * frequency * t)

    sine_wave_z = z_transform(sine_wave, desired_sd, desired_average)

    return sine_wave_z

def pink_noise_signal_creation_using_cn(N, desired_sd, desired_average):
    pink = False
    iterations = 0
    while pink == False:

        pink_noise = cn.powerlaw_psd_gaussian(1, N)

        pink_noise_z = z_transform(pink_noise, desired_sd, desired_average)

        slope, positive_freqs_log, positive_magnitude_log, intercept, name, r, p, positive_freqs, positive_magnitude = quality_assessment_of_temporal_structure_FFT_method(pink_noise_z, 'pink_noise_z')

        if round(slope, 2) == -0.5 and (r <= -0.7) and (p < 0.05):

            #  Figure of Frequincies vs Magnitude
            # plt.figure(figsize=(10,6))
            # plt.plot(positive_freqs, positive_magnitude)
            # plt.title(f'{name}\nFFT of Sine Wave')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Magnitude')
            # plt.grid()
            # plt.show()

            # plt.figure(figsize=(10, 6))
            # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
            # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept,
            #          label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
            # plt.title(f'{name}\nLog-Log Plot of FFT (Frequency vs Magnitude)')
            # plt.xlabel('Log(Frequency) (Hz)')
            # plt.ylabel('Log(Magnitude)')
            # plt.legend()
            # plt.grid()
            # plt.show()
            pink = True
        else:
            print('Not valid pink noise signal')
            iterations +=1
            print(iterations)

    return pink_noise_z

def white_noise_signal_creation(N, desired_sd, desired_average):

    white_noise = np.random.normal(0, 1, N)
    white_signal_z = z_transform(white_noise, desired_sd, desired_average)

    return white_signal_z

def quality_assessment_of_temporal_structure_FFT_method(signal, name):
    # Apply FFT
    fft_output = np.fft.fft(signal)  # FFT of the signal
    fft_magnitude = np.abs(fft_output)  # Magnitude of the FFT

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1/0.01)  # Frequency bins

    # Keep only the positive frequencies
    positive_freqs = frequencies[1:len(frequencies) // 2]  # Skip the zero frequency
    positive_magnitude = fft_magnitude[1:len(frequencies) // 2]  # Skip the zero frequency

    #  Figure of Frequincies vs Magnitude
    # plt.figure(figsize=(10,6))
    # plt.plot(positive_freqs, positive_magnitude)
    # plt.title(f'{name}\nFFT of Sine Wave')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid()
    # plt.show()

    positive_freqs_log = np.log10(positive_freqs[positive_freqs > 0])
    positive_magnitude_log = np.log10(positive_magnitude[positive_freqs > 0])

    r, p = pearsonr(positive_freqs_log, positive_magnitude_log)

    # Perform linear regression (best fit) to assess the slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(positive_freqs_log, positive_magnitude_log)
    print(f'r_value = {r_value}')
    print(f'p_value = {p_value}')

    # Plot the log-log results
    # plt.figure(figsize=(10,6))
    # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
    # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept, label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
    # plt.title(f'{name}\nLog-Log Plot of FFT (Frequency vs Magnitude)')
    # plt.xlabel('Log(Frequency) (Hz)')
    # plt.ylabel('Log(Magnitude)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return slope, positive_freqs_log, positive_magnitude_log, intercept, name, r, p, positive_freqs, positive_magnitude

def one_pink_signal_from_several(num_signals, num_points, desired_sd, desired_average):
    one_pink_signal = []
    for i in range(num_signals):
        one_pink_signal.append(pink_noise_signal_creation_using_cn(num_points, desired_sd, desired_average))

    flattened_list = list(chain.from_iterable(one_pink_signal))

    return flattened_list

def one_white_signal_from_several(num_signals, num_points, desired_sd, desired_average):
    one_white_signal = []
    for i in range(num_signals):
        one_white_signal.append(white_noise_signal_creation(num_points, desired_sd, desired_average))

    flattened_list = list(chain.from_iterable(one_white_signal))

    return flattened_list

def one_sine_signal_from_several(num_signals, num_points, desired_sd, desired_average, num_periods):
    one_sine_signal = []
    for i in range(num_signals):
        one_sine_signal.append(sine_wave_signal_creation(num_points, desired_sd, desired_average, num_periods))

    flattened_list = list(chain.from_iterable(one_sine_signal))

    return flattened_list
num_points = 65
desired_sd = 15
desired_average = 50
num_signals = 10

# for i in range(1, 11):
#
#     pink_noise = pink_noise_signal_creation(num_points, desired_sd, desired_average)
#     df = pd.DataFrame(pink_noise)
#     directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 2\Signals\pink noise signals\100 data points'
#     lb.create_txt_file(pink_noise, rf'pink noise signal {num_points} datapoints {i}', directory)
H_list = []
for i in range(100):

    white_noise = white_noise_signal_creation(num_points, desired_sd, desired_average)
    pink_noise = pink_noise_signal_creation_using_cn(num_points, desired_sd, desired_average)
    sine_wave = sine_wave_signal_creation(num_points, desired_sd, desired_average, 5)
    H_list.append(lb.DFA(pink_noise))
plt.plot(H_list)
plt.title(f'Average = {np.mean(H_list)}\nSD = {np.std(H_list)}')
plt.show()

#
# white_noise = one_white_signal_from_several(num_signals, num_points, desired_sd, desired_average)
# pink_noise = one_pink_signal_from_several(num_signals, num_points, desired_sd, desired_average)
# sine_wave = one_sine_signal_from_several(num_signals, num_points, desired_sd, desired_average, 5)
#
# plt.plot(pink_noise)
# plt.show()
#
# print('***After z transformation')
# outputs(white_noise, pink_noise, sine_wave)
#
#
# print('Pink DFA')
# lb.DFA(pink_noise)
# print('white DFA')
# lb.DFA(white_noise)
# print('sine DFA')
# lb.DFA(sine_wave)
#
#
plt.plot(white_noise, label='white_signal', c='gray', lw=3)
plt.plot(pink_noise, label='pink_noise', c='pink', lw=3)
plt.plot(sine_wave, label='sine_wave', c='red', lw=3)
plt.legend()
plt.show()