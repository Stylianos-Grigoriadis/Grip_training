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

def indexing(df):
    """ This function creates a new dataframe by synchronizing the Time column to the ClosestSampleTime column and then returns a new dataframe with the correct values"""

    time_index = []
    for i in range(len(df)):
        # Calculate the difference of the element i of the column ClosestSampleTime with every value of the column Time
        closest_index = (df['Time'] - df['ClosestSampleTime'].iloc[i]).abs()
        # Drop the None values of the closest_index so that in the next step the .empty attribute if it has only None values it would show False
        closest_index = closest_index.dropna()

        if not closest_index.empty:
            # Find the index of the minimum difference
            closest_index = closest_index.idxmin()
            # Keep only the index of minimum difference
            time_index.append(closest_index)
    # Create all other columns
    time = df.loc[time_index, 'Time'].to_numpy()
    performance = df.loc[time_index, 'Performance'].to_numpy()
    targets = df['Target'].dropna().to_numpy()
    time_close_to_target = df['ClosestSampleTime'].dropna().to_numpy()

    # Create the dataframe which will be returned afterward.
    dist = {'Indices': time_index,
            'Time': time,
            'Performance': performance,
            'ClosestSampleTime': time_close_to_target,
            'Target': targets}
    new_df = pd.DataFrame(dist)

    return new_df

def spatial_error(df):
    pass

def integrate_signal(signal):
    """
    Integrates the input signal to compute the cumulative sum
    after subtracting the mean of the signal.
    Parameters:
    signal (numpy array): Input time series signal
    Returns:
    numpy array: Integrated (cumulative sum) signal
    """
    # Compute the mean of the signal
    mean_signal = np.mean(signal)
    # Subtract the mean from the signal
    detrended_signal = signal - mean_signal
    # Compute the cumulative sum (integrated series)
    integrated_signal = np.cumsum(detrended_signal)
    return integrated_signal

def moving_average(data):
    series = pd.Series(data)
    moving_avg = series.rolling(window=5).mean()
    return moving_avg


df1 = pd.read_csv(r"grip_Karagiorgos_Akis__06Φεβ25_10_37_03.csv", skiprows=2)
pd.set_option('display.max_rows', None)



new_df1_anestis = indexing(df1)
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