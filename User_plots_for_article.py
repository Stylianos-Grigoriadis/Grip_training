import Lib_grip as lb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob
import pwlf
from scipy.stats import pearsonr
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from matplotlib import font_manager
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

def quality_assessment_of_temporal_structure_FFT_method(signal):
    # Apply FFT
    fft_output = np.fft.fft(signal)  # FFT of the signal
    fft_power = np.abs(fft_output) ** 2  # Power spectrum

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(len(signal), d=1 / 0.01)  # Frequency bins

    # Keep only the positive frequencies
    positive_freqs = frequencies[1:len(frequencies) // 2]  # Skip the zero frequency
    positive_power = fft_power[1:len(frequencies) // 2]

    positive_freqs_log = np.log10(positive_freqs[positive_freqs > 0])
    positive_power_log = np.log10(positive_power[positive_freqs > 0])

    r, p = pearsonr(positive_freqs_log, positive_power_log)

    # Perform linear regression (best fit) to assess the slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(positive_freqs_log, positive_power_log)

    # Plot the log-log results
    # plt.figure(figsize=(10,6))
    # plt.scatter(positive_freqs_log, positive_magnitude_log, label='Log-Log Data', color='blue')
    # plt.plot(positive_freqs_log, slope * positive_freqs_log + intercept, label=f'Fit: \nSlope = {slope:.2f}\nr = {r}\np = {p}', color='red')
    # plt.xlabel('Log(Frequency) (Hz)')
    # plt.ylabel('Log(Magnitude)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return slope, positive_freqs_log, positive_power_log, intercept, r, p

sine = lb.read_my_txt_file(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Signals\Training\Sine_65.1_signals\Training_sine_65.1_T1.txt')
time_sine = np.linspace(0, 30, len(sine))
pink = lb.pink_noise_signal_creation_using_cn(65, 15, 50)
time_pink = np.linspace(0, 30, len(pink))
white = lb.white_noise_signal_creation(65, 15, 50)
time_white = np.linspace(0, 30, len(white))

# plt.plot(time_sine, sine, label='Sine')
# plt.plot(time_pink, pink, label='Pink')
# plt.plot(time_white, white, label='White')
# plt.legend()
# plt.show()


sine_slope, sine_positive_freqs_log, sine_positive_magnitude_log, sine_intercept, sine_r, sine_p = quality_assessment_of_temporal_structure_FFT_method(sine)
pink_slope, pink_positive_freqs_log, pink_positive_magnitude_log, pink_intercept, pink_r, pink_p = quality_assessment_of_temporal_structure_FFT_method(pink)
white_slope, white_positive_freqs_log, white_positive_magnitude_log, white_intercept, white_r, white_p = quality_assessment_of_temporal_structure_FFT_method(white)


fig, axes = plt.subplots(2, 3, figsize=(15, 5))  # Adjust figsize as needed

font = font_manager.FontProperties(family='serif', size=12, weight='bold')


axes[0,0].scatter(sine_positive_freqs_log, sine_positive_magnitude_log, label='X axis', c='#2F2F2F')
axes[0,0].plot(sine_positive_freqs_log, sine_slope * sine_positive_freqs_log + sine_intercept, label=f'Slope = {sine_slope:.1f}', c='#2F2F2F', lw=3)
axes[0,0].set_title("Non-variable group")
axes[0,0].set_xlabel("Log(Frequency)")
axes[0,0].set_ylabel("Log(Power)\n")
axes[0,0].legend(frameon=False, prop=font)
axes[0,0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


axes[0,1].scatter(pink_positive_freqs_log, pink_positive_magnitude_log, label='X axis', c='#FF8FA3')
axes[0,1].plot(pink_positive_freqs_log, pink_slope * pink_positive_freqs_log + pink_intercept, label=f'Slope = {pink_slope:.1f}', c='#FF8FA3', lw=3)
axes[0,1].set_title("Structured group")
axes[0,1].set_xlabel("Log(Frequency)")
axes[0,1].legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=False, prop=font)
axes[0,1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


axes[0,2].scatter(white_positive_freqs_log, white_positive_magnitude_log, label='X axis', c='#B0B0B0')
axes[0,2].plot(white_positive_freqs_log, white_slope * white_positive_freqs_log + white_intercept, label=f'Slope = {np.abs(white_slope):.1f}', c='#B0B0B0', lw=3)
axes[0,2].set_title("Non-structured group")
axes[0,2].set_xlabel("Log(Frequency)")
axes[0,2].legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=False, prop=font)
axes[0,2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axes[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

axes[1,0].plot(time_sine, sine, label=f'Sinusoidal signal', c='#2F2F2F', lw=3)
axes[1,0].set_xlabel("Time (seconds)")
axes[1,0].set_ylabel("Screen Height %")

axes[1,1].plot(time_pink, pink, label=f'Pink noise signal', c='#FF8FA3', lw=3)
axes[1,1].set_xlabel("Time (seconds)")

axes[1,2].plot(time_white, white, label=f'White noise signal', c='#B0B0B0', lw=3)
axes[1,2].set_xlabel("Time (seconds)")



# Optional: Adjust layout
plt.subplots_adjust(
    left=0.08,     # space from left edge of figure
    bottom=0.08,    # space from bottom edge
    right=0.98,    # space from right edge
    top=0.95,       # space from top edge
    wspace=0.17,    # width (horizontal) space between subplots
    hspace=0.3     # height (vertical) space between subplots
)
plt.show()

# Colors
# "Static Lighter"  : #6F6F6F
# "Static"          : #4F4F4F
# "Static Darker"   : #2F2F2F

# "Pink Lighter"  : #FFE4EC
# "Pink"          : #FFC0CB
# "Pink Darker"   : #FF8FA3

# "White Lighter"  : #E8E8E8
# "White"          : #D3D3D3
# "White Darker"   : #B0B0B0