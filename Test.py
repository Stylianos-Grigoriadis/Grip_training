import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
import glob
from scipy.stats import pearsonr
from scipy import stats

# directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Data\White_100.6\Isometric_trials'
# os.chdir(directory)
# files = glob.glob(os.path.join(directory, "*"))
# for file in files:
#     data = pd.read_csv(file, skiprows=2)
#     ID = os.path.basename(file)
#     print(ID)
#     print(data)
#     plt.plot(data['Time'], data['Performance'], label='Player')
#     plt.plot(data['ClosestSampleTime'], data['Target'], label='Target')
#     plt.legend()
#     plt.show()

# Isometric_trial_10
# Isometric_trial_50


# directory_1 = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Signals\Sine_65.6_signals\Training_sine_65.6_T5.txt'
# directory_2 = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Pilot study 4\Signals\Sine_100.3_signals\Training_sine_100.3_T3.txt'
#
# Sine_65 = lb.read_my_txt_file(directory_1)
# Sine_100 = lb.read_my_txt_file(directory_2)
# plt.plot(Sine_65, label='Sine_65')
# plt.plot(Sine_100, label='Sine_100')
# plt.legend()
# plt.show()

def quality_assessment_of_temporal_structure_FFT_method(signal, name):
    # Apply FFT
    fft_output = np.fft.fft(signal)

    # Magnitude of FFT
    fft_magnitude = np.abs(fft_output)

    # Power spectrum
    fft_power = fft_magnitude ** 2

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1/0.01)

    # Keep only the positive frequencies, skipping zero frequency
    positive_freqs = frequencies[1:len(frequencies) // 2]
    positive_power = fft_power[1:len(frequencies) // 2]

    # Avoid log10 problems from zero or negative values
    valid_indices = (positive_freqs > 0) & (positive_power > 0)

    positive_freqs_log = np.log10(positive_freqs[valid_indices])
    positive_power_log = np.log10(positive_power[valid_indices])

    # Pearson correlation
    r, p = pearsonr(positive_freqs_log, positive_power_log)

    # Linear regression on log-frequency vs log-power
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        positive_freqs_log,
        positive_power_log
    )

    # Plot the log-log results
    plt.figure(figsize=(10, 6))
    plt.scatter(
        positive_freqs_log,
        positive_power_log,
        label='Log-Log Data',
        color='blue'
    )
    plt.plot(
        positive_freqs_log,
        slope * positive_freqs_log + intercept,
        label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}',
        color='red'
    )
    plt.title(f'{name}\nLog-Log Plot of FFT Power Spectrum')
    plt.xlabel('Log(Frequency) (Hz)')
    plt.ylabel('Log(Power)')
    plt.legend()
    plt.grid()
    plt.show()

    return (
        slope,
        positive_freqs_log,
        positive_power_log,
        intercept,
        name,
        r,
        p,
        positive_freqs,
        positive_power
    )
N=100
average = 10
sd = 5

pink_signal = lb.pink_noise_signal_creation_using_cn(N,sd, average)
plt.plot(pink_signal)
plt.show()

quality_assessment_of_temporal_structure_FFT_method(pink_signal, "name")