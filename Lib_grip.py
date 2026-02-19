import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import colorednoise as cn
import random
from scipy.optimize import curve_fit
from scipy.signal import decimate
from scipy.signal import welch
from scipy import stats
import itertools
from itertools import chain
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
import statsmodels.api as sm
import scipy.sparse as sp





def Ent_Ap(data, dim, r):
    """
    Ent_Ap20120321
      data : time-series data
      dim : embedded dimension
      r : tolerance (typically 0.2)

      Changes in version 1
          Ver 0 had a minor error in the final step of calculating ApEn
          because it took logarithm after summation of phi's.
          In Ver 1, I restored the definition according to original paper's
          definition, to be consistent with most of the work in the
          literature. Note that this definition won't work for Sample
          Entropy which doesn't count self-matching case, because the count
          can be zero and logarithm can fail.

      *NOTE: This code is faster and gives the same result as ApEn =
             ApEnt(data,m,R) created by John McCamley in June of 2015.
             -Will Denton

    ---------------------------------------------------------------------
    coded by Kijoon Lee,  kjlee@ntu.edu.sg
    Ver 0 : Aug 4th, 2011
    Ver 1 : Mar 21st, 2012
    ---------------------------------------------------------------------
    """

    r = r * np.std(data)
    N = len(data)
    phim = np.zeros(2)
    for j in range(2):
        m = dim + j
        phi = np.zeros(N - m + 1)
        data_mat = np.zeros((N - m + 1, m))
        for i in range(m):
            data_mat[:, i] = data[i:N - m + i + 1]
        for i in range(N - m + 1):
            temp_mat = np.abs(data_mat - data_mat[i, :])
            AorB = np.unique(np.where(temp_mat > r)[0])
            AorB = len(temp_mat) - len(AorB)
            phi[i] = AorB / (N - m + 1)
        phim[j] = np.sum(np.log(phi)) / (N - m + 1)
    AE = phim[0] - phim[1]
    return AE

def Ent_Samp(data, m, r):
    """
    function SE = Ent_Samp20200723(data,m,r)
    SE = Ent_Samp20200723(data,m,R) Returns the sample entropy value.
    inputs - data, single column time seres
            - m, length of vectors to be compared
            - r, radius for accepting matches (as a proportion of the
              standard deviation)

    output - SE, sample entropy
    Remarks
    - This code finds the sample entropy of a data series using the method
      described by - Richman, J.S., Moorman, J.R., 2000. "Physiological
      time-series analysis using approximate entropy and sample entropy."
      Am. J. Physiol. Heart Circ. Physiol. 278, H2039–H2049.
    - m is generally recommendation as 2
    - R is generally recommendation as 0.2
    May 2016 - Modified by John McCamley, unonbcf@unomaha.edu
             - This is a faster version of the previous code.
    May 2019 - Modified by Will Denton
             - Added code to check version number in relation to a server
               and to automatically update the code.
    Jul 2020 - Modified by Ben Senderling, bmchnonan@unomaha.edu
             - Removed the code that automatically checks for updates and
               keeps a version history.
    Define r as R times the standard deviation
    """
    R = r * np.std(data)
    N = len(data)

    data = np.array(data)

    dij = np.zeros((N - m, m + 1))
    dj = np.zeros((N - m, 1))
    dj1 = np.zeros((N - m, 1))
    Bm = np.zeros((N - m, 1))
    Am = np.zeros((N - m, 1))

    for i in range(N - m):
        for k in range(m + 1):
            dij[:, k] = np.abs(data[k:N - m + k] - data[i + k])
        dj = np.max(dij[:, 0:m], axis=1)
        dj1 = np.max(dij, axis=1)
        d = np.where(dj <= R)
        d1 = np.where(dj1 <= R)
        nm = d[0].shape[0] - 1  # subtract the self match
        Bm[i] = nm / (N - m)
        nm1 = d1[0].shape[0] - 1  # subtract the self match
        Am[i] = nm1 / (N - m)

    Bmr = np.sum(Bm) / (N - m)
    Amr = np.sum(Am) / (N - m)

    return -np.log(Amr / Bmr)

def Perc(signal , upper_lim, lower_lim):
    """This function takes a signal as a np.array and turns it as values from upper_lim to lower_lim"""
    if np.min(signal) < 0:
        signal = signal - np.min(signal)
    signal = 100 * signal / np.max(signal)
    min_val = signal.min()
    max_val = signal.max()
    signal = (signal - min_val) / (max_val - min_val)
    new_range = upper_lim - lower_lim
    signal = signal * new_range + lower_lim
    return signal

def read_kinvent(path):
    """This funcion reads the Kinvent csv file for the grip"""
    df = pd.read_csv(path, header=None, delimiter=';')
    index = []
    for i, string in enumerate(df[0]):
        if 'Repetition: ' in string:
            index.append(i)
    print(index)
    df_set_1 = pd.read_csv(path, skiprows=2, nrows=index[1] - 3)
    df_set_2 = pd.read_csv(path, skiprows=index[1] + 2, nrows=index[2] - index[1] - 3)
    df_set_3 = pd.read_csv(path, skiprows=index[2] + 2, nrows=index[3] - index[2] - 3)
    df_set_4 = pd.read_csv(path, skiprows=index[3] + 2, nrows=index[4] - index[3] - 3)
    df_set_5 = pd.read_csv(path, skiprows=index[4] + 2, nrows=index[5] - index[4] - 3)
    df_set_6 = pd.read_csv(path, skiprows=index[5] + 2, nrows=index[6] - index[5] - 3)
    df_set_7 = pd.read_csv(path, skiprows=index[6] + 2, nrows=index[7] - index[6] - 3)
    df_set_8 = pd.read_csv(path, skiprows=index[7] + 2, nrows=index[8] - index[7] - 3)
    df_set_9 = pd.read_csv(path, skiprows=index[8] + 2, nrows=index[9] - index[8] - 3)
    df_set_10 = pd.read_csv(path, skiprows=index[9] + 2)

    return (df_set_1,
            df_set_2,
            df_set_3,
            df_set_4,
            df_set_5,
            df_set_6,
            df_set_7,
            df_set_8,
            df_set_9,
            df_set_10)

def sine_signal_generator(Number_of_data_points, frequency, upper_lim, lower_lim):

    x = np.arange(0, Number_of_data_points)
    signal = np.sin(x*frequency)
    signal = Perc(signal, upper_lim, lower_lim)

    # time = np.arange(0, Total_Time, Total_Time / Number_of_data_points)
    # return signal, time
    return signal

def isometric_generator_with_reps(Number_of_data_points,value):
    reps_in_set = 20
    total_reps = Number_of_data_points/reps_in_set
    targets_in_each_rep = Number_of_data_points/total_reps
    array_force = np.full(int(targets_in_each_rep/2), value)
    array_zero = np.zeros(int(targets_in_each_rep/2))
    array_single_rep = np.concatenate((array_zero, array_force))
    signal = np.tile(array_single_rep, reps_in_set)
    return signal

def isometric_generator_single_rep(Number_of_data_points,value):
    signal = np.full(Number_of_data_points, value)
    return signal

def create_txt_file(signal, name, path):
    "This Function takes a np.array and turns it into a txt file so that it can be used in the KINVENT grip game"
    element = ''
    for i in signal:
        element = element + str(i) + ','
    element = element[:-1]
    list_to_save = [element]
    df = pd.DataFrame(list_to_save)
    df.to_csv(rf'{path}\{name}.txt',header=False, index=False, sep=' ')

def make_it_random(up_1, up_2, up_3, down_1, down_2, down_3):
    list1 = [up_1, up_2, up_3, down_1, down_2, down_3]
    random.shuffle(list1)

    return list1

def perturbation_both_force(up_1, up_2, up_3, down_1, down_2, down_3, step_1, step_2, step_3, data_num):
    dat_for_each_pert = int(data_num/12)

    baseline = np.zeros(dat_for_each_pert)
    pert_up_1 = np.full(dat_for_each_pert, up_1)
    pert_up_2 = np.full(dat_for_each_pert, up_2)
    pert_up_3 = np.full(dat_for_each_pert, up_3)
    pert_down_1 = np.full(dat_for_each_pert, down_1)
    pert_down_2 = np.full(dat_for_each_pert, down_2)
    pert_down_3 = np.full(dat_for_each_pert, down_3)
    pert_step_1 = np.full(int(dat_for_each_pert/3), step_1)
    pert_step_2 = np.full(int(dat_for_each_pert/3), step_2)
    pert_step_3 = np.full(int(dat_for_each_pert/3), step_3)
    pert_down_whole_1 = np.concatenate((pert_step_1, pert_down_1))
    pert_down_whole_2 = np.concatenate((pert_step_2, pert_down_2))
    pert_down_whole_3 = np.concatenate((pert_step_3, pert_down_3))

    overall_list = make_it_random(pert_up_1, pert_up_2, pert_up_3, pert_down_whole_1, pert_down_whole_2, pert_down_whole_3)

    final_pert = np.concatenate((baseline, overall_list[0],
                                 baseline, overall_list[1],
                                 baseline, overall_list[2],
                                 baseline, overall_list[3],
                                 baseline, overall_list[4],
                                 baseline, overall_list[5]))
    return final_pert

def total_force(signal):
    total = np.sum(signal)
    return total

def synchronization_of_Time_and_ClosestSampleTime_Stylianos(df, Targets_N):
    """ This function takes a dataframe and converts it so that the Time column and the ClosestSampleTime column
        are matched. This is a method to synchronize the two time stamps
    Parameters
    Input
            df          :   the Dataframe
            Target_N    :   the total number of targets
    Output
            new_df      :   the new Dataframe

    """
    # Find the index of the first value where the ClosestSampleTime is equal to Time
    df['Time'] = df['Time'].round(3)
    df['ClosestSampleTime'] = df['ClosestSampleTime'].round(3)

    # Find the value of the column Time with the smallest absolute difference with the first value of ClosestSampleTime
    closest_value = df['Time'].iloc[(df['Time'] - df['ClosestSampleTime'][0]).abs().idxmin()]

    # Find the index of the column Time with the smallest absolute difference with the first value of ClosestSampleTime
    index = df.loc[df['Time'] == closest_value].index[0]

    # Create a list of ClosestSampleTime with the initial value being the value of Time with the smallest difference
    # with the first value of ClosestSampleTime
    initial_value_time = df['Time'][index]
    list_ClosestSampleTime = [initial_value_time]
    for i in range(Targets_N - 1):
        list_ClosestSampleTime.append(round(list_ClosestSampleTime[-1] + 0.3, 3))

    # Create a list of indices of column Time, where the values of list_ClosestSampleTime are equal with the values of
    # column Time
    matching_indices = df.index[df['Time'].isin(list_ClosestSampleTime)].tolist()

    # Create the Performance, Time, and Target lists with the values at the right indices of the initial df
    Performance = [df['Performance'].iloc[i] for i in matching_indices]
    Time = [df['Time'].iloc[i] for i in matching_indices]
    Target = df['Target'].head(len(matching_indices)).tolist()

    # Delete any values from the end of the list_ClosestSampleTime so that all lists Performance, Time, Target and list_ClosestSampleTime
    # have the same length
    list_ClosestSampleTime = list_ClosestSampleTime[:len(Time)]

    # create the new_df
    dist = {'Time': Time, 'Performance': Performance, 'ClosestSampleTime': list_ClosestSampleTime, 'Target': Target}
    new_df = pd.DataFrame(dist)

    return new_df

def synchronization_of_Time_and_ClosestSampleTime_Anestis(df):
    """ This function creates a new dataframe by synchronizing the Time column to the ClosestSampleTime column and then returns a new dataframe with the correct values"""

    # The following lines where added because sometimes the ClosestSampleTime column starts with a negative value.
    # A temporal fix is to make each negative value None and erase it after
    for i in range(len(df['ClosestSampleTime'])):
        if df['ClosestSampleTime'][i] < 0:
            df.loc[i, "ClosestSampleTime"] = None
            df.loc[i, "Target"] = None
    # The duplicated values of ClosestSampleTime are replaced as None so that we exclude them later
    dup_mask = df['ClosestSampleTime'].duplicated(keep='first')

    df.loc[dup_mask, ['ClosestSampleTime', 'Target']] = None

    time_index = []
    for i in range(len(df)):
        # Calculate the difference of the element i of the column ClosestSampleTime with every value of the column Time
        # Here we added a non NaN so that we dont look over the NaN rows
        if not pd.isna(df['ClosestSampleTime'].iloc[i]):

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

def isolate_Target(df):

    new_Time = []
    ClosestSampleTime = df['ClosestSampleTime'].dropna().to_list()
    Performance = []
    Target = []
    index_list = []

    Time = df['Time'].to_list()
    # Time = [round(value, 3) for value in Time]
    df['Time'] = df['Time'].round(3)
    print(df['Time'])

    for i in ClosestSampleTime:
        if i in df['Time'].values:
            index = df.index[df['Time'] == i].tolist()[0]
            print(index)
            Performance.append(df['Performance'][index])
            Target.append(df['Target'][index])
            index_list.append(index)
            new_Time.append(i)
    df_targets = pd.DataFrame({'Time': new_Time, 'Performance': Performance, 'ClosestSampleTime': ClosestSampleTime, 'Target': Target})

    return df_targets

def spatial_error(df):
    """ Calculate the spatial error of the Performance and Target
    Parameters
    Input
            df              :   the Dataframe
    Output
            spatial_error   :   the spatial_error between the Performance and Target
    """

    spatial_error = []
    for i in range(len(df['Time'])):
        # Spatial error with absolute values
        spatial_error.append((abs(df['Performance'][i]-df['Target'][i])))
    spatial_error = np.array(spatial_error)
    return spatial_error

def read_my_txt_file(path):
    df = pd.read_csv(path, delimiter=',', decimal='.',header=None)

    signal_list = []
    for i in range(df.shape[1]):
        signal_list.append(df[i][0])
    signal = np.array(signal_list)

    return signal

def asymptotes(df, asymptote_fraction=0.99, plot=True):
    df = synchronization_of_Time_and_ClosestSampleTime_Anestis(df)

    perturbation_index = df[df['Target'] != df['Target'].shift(1)].index[1]
    # print(df.columns)

    # Calculate the spatial error

    spatial_er = spatial_error(df)
    x = df['Time'].to_numpy()
    start_idx = perturbation_index
    x_fit = (df['Time'].iloc[start_idx:] - df['Time'].iloc[start_idx]).to_numpy()
    y_fit = spatial_er[start_idx:]

    def exp_decay(t, A, k, C):
        return A * np.exp(-k * t) + C

    C0 = np.mean(y_fit[-max(5, len(y_fit) // 10):])
    A0 = y_fit[0] - C0
    k0 = 1 / (x_fit[-1] + 1e-6)

    # Fit
    popt, _ = curve_fit(exp_decay, x_fit, y_fit, p0=[A0, k0, C0])
    A, k, C = popt

    t_asymptote = -np.log(1 - asymptote_fraction) / k
    results = {
        "A": A,
        "k": k,
        "asymptote_value": C,
        "t_asymptote": t_asymptote,
        "total_time": x[start_idx] + t_asymptote
    }
    if plot:
        t_plot = np.linspace(0, x_fit.max(), 500)
        y_plot = exp_decay(t_plot, A, k, C)

        plt.figure(figsize=(7, 4))
        plt.scatter(x, spatial_er, s=15, alpha=0.6, label="Data")
        plt.plot(t_plot + x[start_idx], y_plot, 'r', lw=2, label="Exp fit")
        plt.axvline(x[start_idx], color='r', lw=2)
        plt.axhline(C, color='k', ls='--', label="Asymptote")
        plt.axvline(
            x[start_idx] + t_asymptote,
            color='k',
            ls=':',
            label=f"Asymptote reached ({t_asymptote:.2f} s)"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results


def adaptation_time_using_sd_right_before_perturbation(df, perturbation_index, sd_factor, first_values, consecutive_values, values_for_sd, name, plot=False):
    """
    This function returns the time after the perturbation which was needed to adapt to the perturbation
    Parameters
    Input
            df                  :   The Dataframe
            perturbation_index  :   The index where the perturbation occurred
            sd_factor           :   This will be multiplied with the sd of the error before the perturbation
                                    and if the error after the is less than the mean + sd*sd_factor and more than
                                    the mean - sd*sd_factor, the algorithm will consider that the adaptation of the
                                    perturbation occurred
            first_values        :   At first the error will be too much so to calculate the mean and sd before the perturbation
                                    right, we erase some values from the beginning
            consecutive_values  :   This is how many values the algorithm needs to consider so that it decides that the adaptation occurred.
            total targets       :   The total number of targets
            Plot                :   Plot the spatial error and the time of adaptation (default value False)

    Output
            time_of_adaptation  :   The time it took the df['Performance'] to steadily reach df['Target']. This
                                    number corresponds to the first value of time at which for the next X consecutive_values
                                    the spatial error was lower than the average +- (sd * sd_factor)
    """
    # First synchronize the Time and ClosestSampleTime columns and create a new df with
    # only the synchronized values
    df = synchronization_of_Time_and_ClosestSampleTime_Anestis(df)

    # Calculate the spatial error and the average and sd of the spatial error
    # after the first_values
    spatial_er = spatial_error(df)


    # The following line calculate the lowest sd for 'values_for_sd' in overlapping window and
    # the lowest sd is the sd used for further analysis
    list_for_mean_and_sd = spatial_er[perturbation_index:]
    list_of_sd = []
    list_of_means = []
    for i in range(len(list_for_mean_and_sd) - values_for_sd):
        average = np.mean(list_for_mean_and_sd[i:i+values_for_sd])
        sd = np.std(list_for_mean_and_sd[i:i+values_for_sd])
        list_of_means.append(average)
        list_of_sd.append(sd)
    min_sd = min(list_of_sd)
    min_sd_index = list_of_sd.index(min_sd)
    average_at_min_sd = list_of_means[min_sd_index]

    if plot == True:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
        fig.suptitle(f"{name}", fontsize=14)

        axs[0].plot(spatial_er)
        axs[0].set_title(f'Spatial Error')
        axs[1].plot(list_of_sd, label='SD')
        axs[1].plot(list_of_means, label='Average')
        axs[1].axvline(x=min_sd_index, color="red", linestyle="--", label="Lowest SD")
        axs[1].set_title(f'Standard Deviation')
        axs[1].legend(loc="best")

        plt.tight_layout()
        plt.show()


    # Create an array with consecutive_values equal number
    consecutive_values_list = np.arange(0,consecutive_values,1)

    # Iterate the spatial error after the perturbation_index to calculate the time of adaptation
    for i in range(len(spatial_er) - consecutive_values+1):
        if i >= perturbation_index:

            if (all(spatial_er[i + j] < average_at_min_sd + min_sd * sd_factor for j in consecutive_values_list) #and
                # all(spatial_er[i + j] > average_at_min_sd - min_sd * sd_factor for j in consecutive_values_list)
            ):
                time_of_adaptation = df['Time'][i] - df['Time'][perturbation_index]
                break

    if plot == True:
        try:
            time_of_adaptation
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.axhline(y=average_at_min_sd, c='k', label = 'Average')
            plt.axhline(y=average_at_min_sd + min_sd*sd_factor, c='k', ls=":", label=f'{sd_factor}*std')
            # plt.axhline(y=average_at_min_sd - min_sd*sd_factor, c='k', ls=":")
            plt.axvline(x=df['Time'][perturbation_index] + time_of_adaptation, lw=3, c='red', label='Adaptation instance')
            plt.axvline(x=df['Time'][perturbation_index], linestyle='--', c='gray', label='Perturbation instance')

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name}\ntime for adaptation: {round(time_of_adaptation,3)} sec')
            plt.show()
        except NameError:
            print(f"No adaptation was evident for {name}")
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.axhline(y=average_at_min_sd, c='k', label='Average')
            plt.axhline(y=average_at_min_sd + min_sd * sd_factor, c='k', ls=":", label=f'{sd_factor}*std')
            # plt.axhline(y=average_at_min_sd - min_sd * sd_factor, c='k', ls=":")
            plt.axvline(x=df['Time'][perturbation_index], linestyle='--', c='gray', label='Perturbation instance')

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name}\nNo adaptation')
            plt.show()

    try:
        time_of_adaptation
        return time_of_adaptation
    except:
        time_of_adaptation = None
        return time_of_adaptation

def adaptation_time_using_sd_from_isometric_trials(df, sd_factor, time_window, name, mean_spatial_error_isometric_trials, sd_spatial_error_isometric_trials, asymptote_fraction=0.95, plot=False):
    """
    This function returns the time after the perturbation which was needed to adapt to the perturbation
    Parameters
    Input
            df                                      :   The Dataframe
            sd_factor                               :   This will be multiplied with the sd of the error before the perturbation
                                                        and if the error after the is less than the mean + sd*sd_factor and more than
                                                        the mean - sd*sd_factor, the algorithm will consider that the adaptation of the
                                                        perturbation occurred
            time_window                             :   This is how much time the algorithm needs to consider so that it decides that the adaptation occurred.
            name                                    :   The name of the participant and/or the trial
            mean_spatial_error_isometric_trials     :   The average of spatial error which has been previously calculated by isometric trials
            sd_spatial_error_isometric_trials       :   The sd of spatial error which has been previously calculated by isometric trials
            Plot                                    :   Plot the spatial error and the time of adaptation (default value False)

    Output
            time_of_adaptation                      :   The time it took the df['Performance'] to steadily reach df['Target']. This
                                                        number corresponds to the first value of time at which for the next X consecutive_values
                                                        the spatial error was lower than the average +- (sd * sd_factor)
    """
    # First synchronize the Time and ClosestSampleTime columns and create a new df with
    # only the synchronized values
    df = synchronization_of_Time_and_ClosestSampleTime_Anestis(df)

    perturbation_index = df[df['Target'] != df['Target'].shift(1)].index[1]
    # print(df.columns)

    # Calculate the spatial error
    spatial_er = spatial_error(df)
    # time = np.linspace(0, len(spatial_er), len(spatial_er))
    # plt.scatter(time, spatial_er)
    # plt.show()
    ############################################################################
    # Calculation of time_of_adaptation using the sd method
    mean_isometric_trial = mean_spatial_error_isometric_trials
    sd_isometric_trial = sd_spatial_error_isometric_trials

    # Create an array with consecutive_values equal number
    target_time = df['Time'].iloc[-1] - time_window

    idx = (df['Time'] - target_time).abs().idxmin()
    pos_idx = df.index.get_loc(idx)
    last_pos = len(df) - 1
    last_index_of_iteration = last_pos - pos_idx

    # Iterate the spatial error after the perturbation_index to calculate the time of adaptation
    for i in range(len(spatial_er) - last_index_of_iteration):
        if i >= perturbation_index:
            start_time_window_position = i
            end_time_window = df['Time'].iloc[start_time_window_position] + time_window

            end_idx_label = (df['Time'] - end_time_window).abs().idxmin()

            end_time_window_position = df.index.get_loc(end_idx_label)

            consecutive_values = end_time_window_position - start_time_window_position

            consecutive_values_list = np.arange(0, consecutive_values, 1)
            if all(spatial_er[i + j] < mean_isometric_trial + (sd_isometric_trial * sd_factor) for j in consecutive_values_list):
                time_of_adaptation_sd = df['Time'].iloc[i] - df['Time'].iloc[perturbation_index]
                time_until_spatial_error_is_lower_than_theshold = df['Time'].iloc[end_time_window_position]
                break

    ############################################################################
    # Calculation of time_of_adaptation using the asymptote method

    x = df['Time'].to_numpy()
    start_idx = perturbation_index
    x_fit = (df['Time'].iloc[start_idx:] - df['Time'].iloc[start_idx]).to_numpy()
    y_fit = spatial_er[start_idx:]

    def exp_decay(t, A, k, C):
        return A * np.exp(-k * t) + C

    C0 = np.mean(y_fit[-max(5, len(y_fit) // 10):])
    A0 = y_fit[0] - C0
    k0 = 1 / (x_fit[-1] + 1e-6)

    # Fit
    try:
        popt, _ = curve_fit(exp_decay, x_fit, y_fit, p0=[A0, k0, C0])
        A, k, C = popt
        t_asymptote = -np.log(1 - asymptote_fraction) / k
        time_to_adapt_asymptote = t_asymptote
    except RuntimeError:
        time_to_adapt_asymptote = None
    ############################################################################
    # Plot both methods
    if plot == 'both':

        plt.figure(figsize=(8, 5))
        plt.plot(df['Time'], spatial_er, label='Spatial Error')
        plt.scatter(df['Time'], spatial_er, lw=0.5)

        # ----- COMMON -----
        plt.axhline(y=mean_isometric_trial, c='k', label='Average')
        plt.axhline(
            y=mean_isometric_trial + sd_isometric_trial * sd_factor,
            c='k',
            ls=":",
            label=f'{sd_factor}*std'
        )

        plt.axvline(
            x=df['Time'].iloc[perturbation_index],
            linestyle='--',
            c='gray',
            label='Perturbation instance'
        )

        # ----- ASYMPTOTE METHOD (only if exists) -----
        if time_to_adapt_asymptote is not None:
            t_plot = np.linspace(0, x_fit.max(), 500)
            y_plot = exp_decay(t_plot, A, k, C)

            plt.plot(
                t_plot + df['Time'].iloc[perturbation_index],
                y_plot,
                'b',
                lw=2,
                label='Exponential fit'
            )

            plt.axhline(
                C,
                color='b',
                ls='--',
                label='Asymptote'
            )

            plt.axvline(
                df['Time'].iloc[perturbation_index] + time_to_adapt_asymptote,
                color='b',
                ls=':',
                lw=2,
                label=f'Asymptote reached ({time_to_adapt_asymptote:.2f} s)'
            )

        # ----- SD METHOD (only if exists) -----
        try:
            time_of_adaptation_sd

            plt.axvline(
                x=df['Time'].iloc[perturbation_index] + time_of_adaptation_sd,
                lw=3,
                c='red',
                label='Adaptation instance (SD)'
            )

            plt.axvspan(
                df['Time'].iloc[perturbation_index] + time_of_adaptation_sd,
                time_until_spatial_error_is_lower_than_theshold,
                color='gray',
                alpha=0.3,
                label='SD check window'
            )

            title_sd = f'SD adaptation: {round(time_of_adaptation_sd, 3)} s'
        except NameError:
            title_sd = 'SD adaptation: not detected'

        title_asymp = (
            f'Asymptote adaptation: {round(time_to_adapt_asymptote, 3)} s'
            if time_to_adapt_asymptote is not None
            else 'Asymptote adaptation: not detected'
        )

        plt.legend()
        plt.ylabel('Force difference (kg)')
        plt.xlabel('Time (sec)')
        plt.title(f'{name}\n{title_sd} | {title_asymp}')
        plt.tight_layout()
        plt.show()

    if plot == 'sd method':
        try:
            time_of_adaptation_sd
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.scatter(df['Time'], spatial_er, lw=0.5)

            plt.axhline(y=mean_isometric_trial, c='k', label='Average')
            plt.axhline(
                y=mean_isometric_trial + sd_isometric_trial * sd_factor,
                c='k',
                ls=":",
                label=f'{sd_factor}*std'
            )

            plt.axvline(
                x=df['Time'][perturbation_index] + time_of_adaptation_sd,
                lw=3,
                c='red',
                label='Adaptation instance'
            )

            plt.axvspan(
                df['Time'][perturbation_index] + time_of_adaptation_sd,
                time_until_spatial_error_is_lower_than_theshold,
                color='gray',
                alpha=0.3,
                label='Check Window'
            )

            plt.axvline(
                x=df['Time'][perturbation_index],
                linestyle='--',
                c='gray',
                label='Perturbation instance'
            )

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name}\ntime for adaptation: {round(time_of_adaptation_sd, 3)} sec')
            plt.show()

        except NameError:
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.axhline(y=mean_isometric_trial, c='k', label='Average')
            plt.axhline(
                y=mean_isometric_trial + sd_isometric_trial * sd_factor,
                c='k',
                ls=":",
                label=f'{sd_factor}*std'
            )
            plt.axvline(
                x=df['Time'][perturbation_index],
                linestyle='--',
                c='gray',
                label='Perturbation instance'
            )

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name}\nNo adaptation')
            plt.show()

    if plot == 'asymptote method' and time_to_adapt_asymptote is not None:
        t_plot = np.linspace(0, x_fit.max(), 500)
        y_plot = exp_decay(t_plot, A, k, C)

        plt.figure(figsize=(7, 4))
        plt.scatter(x, spatial_er, s=15, alpha=0.6, label="Data")
        plt.plot(t_plot + x[start_idx], y_plot, 'r', lw=2, label="Exp fit")
        plt.axvline(x[start_idx], color='r', lw=2)
        plt.axhline(C, color='k', ls='--', label="Asymptote")
        plt.axvline(
            x[start_idx] + time_to_adapt_asymptote,
            color='k',
            ls=':',
            label=f"Asymptote reached ({time_to_adapt_asymptote:.2f} s)"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================
    # RETURN VALUES
    # =========================

    try:
        sd_val = round(time_of_adaptation_sd, 2)
    except NameError:
        sd_val = None

    asymp_val = (
        round(time_to_adapt_asymptote, 2)
        if time_to_adapt_asymptote is not None
        else None
    )

    return sd_val, asymp_val

def adaptation_time_using_sd_from_isometric_trials_and_asymptotes(df, sd_factor, time_window, name, mean_spatial_error_isometric_trials, sd_spatial_error_isometric_trials, asymptote_fraction=0.95, plot=False):
    """
    This function returns the time after the perturbation which was needed to adapt to the perturbation
    Parameters
    Input
            df                                      :   The Dataframe
            sd_factor                               :   This will be multiplied with the sd of the error before the perturbation
                                                        and if the error after the is less than the mean + sd*sd_factor and more than
                                                        the mean - sd*sd_factor, the algorithm will consider that the adaptation of the
                                                        perturbation occurred
            time_window                             :   This is how much time the algorithm needs to consider so that it decides that the adaptation occurred.
            name                                    :   The name of the participant and/or the trial
            mean_spatial_error_isometric_trials     :   The average of spatial error which has been previously calculated by isometric trials
            sd_spatial_error_isometric_trials       :   The sd of spatial error which has been previously calculated by isometric trials
            Plot                                    :   Plot the spatial error and the time of adaptation (default value False)

    Output
            time_of_adaptation                      :   The time it took the df['Performance'] to steadily reach df['Target']. This
                                                        number corresponds to the first value of time at which for the next X consecutive_values
                                                        the spatial error was lower than the average +- (sd * sd_factor)
    """
    # First synchronize the Time and ClosestSampleTime columns and create a new df with
    # only the synchronized values
    df = synchronization_of_Time_and_ClosestSampleTime_Anestis(df)

    perturbation_index = df[df['Target'] != df['Target'].shift(1)].index[1]
    # print(df.columns)

    # Calculate the spatial error
    spatial_er = spatial_error(df)
    # time = np.linspace(0, len(spatial_er), len(spatial_er))
    # plt.scatter(time, spatial_er)
    # plt.show()
    ############################################################################
    # Calculation of time_of_adaptation using the sd method
    mean_isometric_trial = mean_spatial_error_isometric_trials
    sd_isometric_trial = sd_spatial_error_isometric_trials

    # Create an array with consecutive_values equal number
    target_time = df['Time'].iloc[-1] - time_window

    idx = (df['Time'] - target_time).abs().idxmin()
    pos_idx = df.index.get_loc(idx)
    last_pos = len(df) - 1
    last_index_of_iteration = last_pos - pos_idx

    # Iterate the spatial error after the perturbation_index to calculate the time of adaptation
    for i in range(len(spatial_er) - last_index_of_iteration):
        if i >= perturbation_index:
            start_time_window_position = i
            end_time_window = df['Time'].iloc[start_time_window_position] + time_window

            end_idx_label = (df['Time'] - end_time_window).abs().idxmin()

            end_time_window_position = df.index.get_loc(end_idx_label)

            consecutive_values = end_time_window_position - start_time_window_position

            consecutive_values_list = np.arange(0, consecutive_values, 1)
            if all(spatial_er[i + j] < mean_isometric_trial + (sd_isometric_trial * sd_factor) for j in consecutive_values_list):
                time_of_adaptation_sd = df['Time'].iloc[i] - df['Time'].iloc[perturbation_index]
                time_until_spatial_error_is_lower_than_theshold = df['Time'].iloc[end_time_window_position]
                break

    ############################################################################
    # Calculation of time_of_adaptation using the asymptote method

    x = df['Time'].to_numpy()
    start_idx = perturbation_index
    x_fit = (df['Time'].iloc[start_idx:] - df['Time'].iloc[start_idx]).to_numpy()
    y_fit = spatial_er[start_idx:]

    def exp_decay(t, A, k, C):
        return A * np.exp(-k * t) + C

    C0 = np.mean(y_fit[-max(5, len(y_fit) // 10):])
    A0 = y_fit[0] - C0
    k0 = 1 / (x_fit[-1] + 1e-6)

    # Fit
    try:
        popt, _ = curve_fit(exp_decay, x_fit, y_fit, p0=[A0, k0, C0])
        A, k, C = popt
        t_asymptote = -np.log(1 - asymptote_fraction) / k
        time_to_adapt_asymptote = t_asymptote
    except RuntimeError:
        time_to_adapt_asymptote = None

    ############################################################################
    # Calculation of time_of_adaptation using the combination of sd and asymptote methods

    ############################################################################
    # Calculation of time_of_adaptation using COMBINED method
    # (Asymptote must be reached first AND threshold must be maintained)

    time_of_adaptation_combined = None

    # Proceed only if asymptote fitting succeeded
    if time_to_adapt_asymptote is not None:

        # Absolute time (in df['Time']) when asymptote criterion is reached
        asymptote_absolute_time = df['Time'].iloc[perturbation_index] + time_to_adapt_asymptote

        # Find the index closest to that asymptote time
        asymptote_idx = (df['Time'] - asymptote_absolute_time).abs().idxmin()
        asymptote_pos = df.index.get_loc(asymptote_idx)

        # Iterate ONLY after the asymptote index
        for i in range(asymptote_pos, len(spatial_er)):

            start_time_window_position = i
            end_time_window = df['Time'].iloc[start_time_window_position] + time_window

            # Stop if window exceeds trial duration
            if end_time_window > df['Time'].iloc[-1]:
                break

            end_idx_label = (df['Time'] - end_time_window).abs().idxmin()
            end_time_window_position = df.index.get_loc(end_idx_label)

            consecutive_values = end_time_window_position - start_time_window_position

            consecutive_values_list = np.arange(0, consecutive_values, 1)

            # Check threshold condition for the full window
            if all(
                    spatial_er[i + j] < mean_isometric_trial + (sd_isometric_trial * sd_factor)
                    for j in consecutive_values_list
            ):
                time_of_adaptation_combined = (
                        df['Time'].iloc[i] - df['Time'].iloc[perturbation_index]
                )
                break

    ############################################################################
    # Plot both methods
    if plot == 'both':

        plt.figure(figsize=(8, 5))
        plt.plot(df['Time'], spatial_er, label='Spatial Error')
        plt.scatter(df['Time'], spatial_er, lw=0.5)

        # ----- COMMON -----
        plt.axhline(y=mean_isometric_trial, c='k', label='Average')
        plt.axhline(
            y=mean_isometric_trial + sd_isometric_trial * sd_factor,
            c='k',
            ls=":",
            label=f'{sd_factor}*std'
        )

        plt.axvline(
            x=df['Time'].iloc[perturbation_index],
            linestyle='--',
            c='gray',
            label='Perturbation instance'
        )

        # ----- ASYMPTOTE METHOD (only if exists) -----
        if time_to_adapt_asymptote is not None:
            t_plot = np.linspace(0, x_fit.max(), 500)
            y_plot = exp_decay(t_plot, A, k, C)

            plt.plot(
                t_plot + df['Time'].iloc[perturbation_index],
                y_plot,
                'b',
                lw=2,
                label='Exponential fit'
            )

            plt.axhline(
                C,
                color='b',
                ls='--',
                label='Asymptote'
            )

            plt.axvline(
                df['Time'].iloc[perturbation_index] + time_to_adapt_asymptote,
                color='b',
                ls=':',
                lw=2,
                label=f'Asymptote reached ({time_to_adapt_asymptote:.2f} s)'
            )

        # ----- SD METHOD (only if exists) -----
        try:
            time_of_adaptation_sd

            plt.axvline(
                x=df['Time'].iloc[perturbation_index] + time_of_adaptation_sd,
                lw=3,
                c='red',
                label='Adaptation instance (SD)'
            )

            plt.axvspan(
                df['Time'].iloc[perturbation_index] + time_of_adaptation_sd,
                time_until_spatial_error_is_lower_than_theshold,
                color='gray',
                alpha=0.3,
                label='SD check window'
            )

            title_sd = f'SD adaptation: {round(time_of_adaptation_sd, 3)} s'
        except NameError:
            title_sd = 'SD adaptation: not detected'

        title_asymp = (
            f'Asymptote adaptation: {round(time_to_adapt_asymptote, 3)} s'
            if time_to_adapt_asymptote is not None
            else 'Asymptote adaptation: not detected'
        )

        plt.legend()
        plt.ylabel('Force difference (kg)')
        plt.xlabel('Time (sec)')
        plt.title(f'{name}\n{title_sd} | {title_asymp}')
        plt.tight_layout()
        plt.show()

    if plot == 'sd method':
        try:
            time_of_adaptation_sd
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.scatter(df['Time'], spatial_er, lw=0.5)

            # plt.axhline(y=mean_isometric_trial, c='k', label='Average')
            plt.axhline(
                y=mean_isometric_trial + sd_isometric_trial * sd_factor,
                c='k',
                ls=":",
                # label=f'{sd_factor}*std'
                label=f'Threshold'

            )

            plt.axvline(
                x=df['Time'][perturbation_index] + time_of_adaptation_sd,
                lw=3,
                c='red',
                label='Adaptation instance'
            )

            plt.axvspan(
                df['Time'][perturbation_index] + time_of_adaptation_sd,
                time_until_spatial_error_is_lower_than_theshold,
                color='gray',
                alpha=0.3,
                label='Check Window'
            )

            plt.axvline(
                x=df['Time'][perturbation_index],
                linestyle='--',
                c='gray',
                label='Perturbation instance'
            )

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name}\ntime for adaptation: {round(time_of_adaptation_sd, 3)} sec')
            plt.show()

        except NameError:
            plt.plot(df['Time'], spatial_er, label='Spatial Error')
            plt.axhline(y=mean_isometric_trial, c='k', label='Average')
            plt.axhline(
                y=mean_isometric_trial + sd_isometric_trial * sd_factor,
                c='k',
                ls=":",
                label=f'{sd_factor}*std'
            )
            plt.axvline(
                x=df['Time'][perturbation_index],
                linestyle='--',
                c='gray',
                label='Perturbation instance'
            )

            plt.legend()
            plt.ylabel('Force difference (kg)')
            plt.xlabel('Time (sec)')
            plt.title(f'{name}\nNo adaptation')
            plt.show()

    if plot == 'asymptote method' and time_to_adapt_asymptote is not None:
        t_plot = np.linspace(0, x_fit.max(), 500)
        y_plot = exp_decay(t_plot, A, k, C)

        plt.figure(figsize=(7, 4))
        plt.scatter(x, spatial_er, s=15, alpha=0.6, label="Data")
        plt.plot(t_plot + x[start_idx], y_plot, 'r', lw=2, label="Exp fit")
        plt.axvline(x[start_idx], color='r', lw=2)
        plt.axhline(C, color='k', ls='--', label="Asymptote")
        plt.axvline(
            x[start_idx] + time_to_adapt_asymptote,
            color='k',
            ls=':',
            label=f"Asymptote reached ({time_to_adapt_asymptote:.2f} s)"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    ############################################################################
    ############################################################################
    # Plot COMBINED method (SD + Asymptote) + SD reference
    if plot == 'combined':

        plt.figure(figsize=(8, 5))
        plt.plot(df['Time'], spatial_er, label='Spatial Error')
        plt.scatter(df['Time'], spatial_er, lw=0.5)

        # ----- COMMON -----
        plt.axhline(y=mean_isometric_trial, c='k', label='Average')
        plt.axhline(
            y=mean_isometric_trial + sd_isometric_trial * sd_factor,
            c='k',
            ls=":",
            label=f'{sd_factor}*std'
        )

        plt.axvline(
            x=df['Time'].iloc[perturbation_index],
            linestyle='--',
            c='gray',
            label='Perturbation instance'
        )

        # ----- ASYMPTOTE METHOD (only if exists) -----
        if time_to_adapt_asymptote is not None:
            t_plot = np.linspace(0, x_fit.max(), 500)
            y_plot = exp_decay(t_plot, A, k, C)

            plt.plot(
                t_plot + df['Time'].iloc[perturbation_index],
                y_plot,
                'b',
                lw=2,
                label='Exponential fit'
            )

            plt.axhline(
                C,
                color='b',
                ls='--',
                label='Asymptote'
            )

            plt.axvline(
                df['Time'].iloc[perturbation_index] + time_to_adapt_asymptote,
                color='b',
                ls=':',
                lw=2,
                label=f'Asymptote reached ({time_to_adapt_asymptote:.2f} s)'
            )

        # ----- SD METHOD (reference, red) -----
        try:
            time_of_adaptation_sd

            plt.axvline(
                x=df['Time'].iloc[perturbation_index] + time_of_adaptation_sd,
                lw=2,
                c='red',
                ls='--',
                label='Adaptation instance (SD)'
            )
        except NameError:
            pass

        # ----- COMBINED METHOD (orange) -----
        if time_of_adaptation_combined is not None:

            combined_absolute_time = (
                    df['Time'].iloc[perturbation_index] + time_of_adaptation_combined
            )

            plt.axvline(
                x=combined_absolute_time,
                lw=3,
                c='orange',
                label='Adaptation instance (Combined)'
            )

            plt.axvspan(
                combined_absolute_time,
                combined_absolute_time + time_window,
                color='orange',
                alpha=0.25,
                label='Combined check window'
            )

            title_combined = (
                f'Combined adaptation: {round(time_of_adaptation_combined, 3)} s'
            )
        else:
            title_combined = 'Combined adaptation: not detected'

        title_asymp = (
            f'Asymptote adaptation: {round(time_to_adapt_asymptote, 3)} s'
            if time_to_adapt_asymptote is not None
            else 'Asymptote adaptation: not detected'
        )

        plt.legend()
        plt.ylabel('Force difference (kg)')
        plt.xlabel('Time (sec)')
        plt.title(f'{name}\n{title_combined} | {title_asymp}')
        plt.tight_layout()
        plt.show()

    # =========================
    # RETURN VALUES
    # =========================

    try:
        sd_val = round(time_of_adaptation_sd, 2)
    except NameError:
        sd_val = None

    asymp_val = (
        round(time_to_adapt_asymptote, 2)
        if time_to_adapt_asymptote is not None
        else None
    )

    return sd_val, asymp_val
def single_perturbation_generator(baseline, perturbation, data_num):
    baseline_array = np.full(int(data_num/2), baseline)
    perturbation_array = np.full(int(data_num/2), perturbation)
    final_pert = np.concatenate((baseline_array, perturbation_array))

    return final_pert

def isometric_min_max(MVC):
    sd = 10
    iso_90 = 90
    iso_80 = 80
    iso_70 = 70
    iso_60 = 60
    iso_50 = 50
    iso_40 = 40
    iso_30 = 30
    iso_20 = 20
    iso_15 = 15
    iso_10 = 10
    iso_5 = 5
    iso_2_half = 2.5

    iso_90_perc = MVC * iso_90 / 100
    iso_80_perc = MVC * iso_80 / 100
    iso_70_perc = MVC * iso_70 / 100
    iso_60_perc = MVC * iso_60 / 100
    iso_50_perc = MVC * iso_50 / 100
    iso_40_perc = MVC * iso_40 / 100
    iso_30_perc = MVC * iso_30 / 100
    iso_20_perc = MVC * iso_20 / 100
    iso_15_perc = MVC * iso_15 / 100
    iso_10_perc = MVC * iso_10 / 100
    iso_5_perc = MVC * iso_5 / 100
    iso_2_half_perc = MVC * iso_2_half / 100

    iso_90_min = (iso_90 - sd) * MVC / 100
    iso_80_min = (iso_80 - sd) * MVC / 100
    iso_70_min = (iso_70 - sd) * MVC / 100
    iso_60_min = (iso_60 - sd) * MVC / 100
    iso_50_min = (iso_50 - sd) * MVC / 100
    iso_40_min = (iso_40 - sd) * MVC / 100
    iso_30_min = (iso_30 - sd) * MVC / 100
    iso_20_min = (iso_20 - sd) * MVC / 100
    iso_15_min = (iso_15 - sd) * MVC / 100
    iso_10_min = (iso_10 - sd) * MVC / 100
    iso_5_min = (iso_5 - iso_5) * MVC / 100
    iso_2_half_min = (iso_2_half - iso_2_half) * MVC / 100

    iso_90_max = (iso_90 + sd) * MVC / 100
    iso_80_max = (iso_80 + sd) * MVC / 100
    iso_70_max = (iso_70 + sd) * MVC / 100
    iso_60_max = (iso_60 + sd) * MVC / 100
    iso_50_max = (iso_50 + sd) * MVC / 100
    iso_40_max = (iso_40 + sd) * MVC / 100
    iso_30_max = (iso_30 + sd) * MVC / 100
    iso_20_max = (iso_20 + sd) * MVC / 100
    iso_15_max = (iso_15 + sd) * MVC / 100
    iso_10_max = (iso_10 + sd) * MVC / 100
    iso_5_max = (iso_5 + iso_5) * MVC / 100
    iso_2_half_max = (iso_2_half + iso_2_half) * MVC / 100


    print(f"For 90% of MVC ({iso_90_perc}) the min values is {iso_90_min} and the max values is {iso_90_max}")
    print(f"For 80% of MVC ({iso_80_perc}) the min values is {iso_80_min} and the max values is {iso_80_max}")
    print(f"For 70% of MVC ({iso_70_perc}) the min values is {iso_70_min} and the max values is {iso_70_max}")
    print(f"For 60% of MVC ({iso_60_perc}) the min values is {iso_60_min} and the max values is {iso_60_max}")
    print(f"For 50% of MVC ({iso_50_perc}) the min values is {iso_50_min} and the max values is {iso_50_max}")
    print(f"For 40% of MVC ({iso_40_perc}) the min values is {iso_40_min} and the max values is {iso_40_max}")
    print(f"For 30% of MVC ({iso_30_perc}) the min values is {iso_30_min} and the max values is {iso_30_max}")
    print(f"For 90% of MVC ({iso_20_perc}) the min values is {iso_20_min} and the max values is {iso_20_max}")
    print(f"For 15% of MVC ({iso_15_perc}) the min values is {iso_15_min} and the max values is {iso_15_max}")
    print(f"For 10% of MVC ({iso_10_perc}) the min values is {iso_10_min} and the max values is {iso_10_max}")
    print(f"For 5% of MVC ({iso_5_perc}) the min values is {iso_5_min} and the max values is {iso_5_max}")
    print(f"For 2.5% of MVC ({iso_2_half_perc}) the min values is {iso_2_half_min} and the max values is {iso_2_half_max}")

def signal_from_min_to_max(signal,max):
    ''' Where:
                signal: is the signal I want to change
                max:    is the max force I inserted into Kinvent app
                '''
    signal = np.array(signal)
    signal = signal * max / 100
    return signal

def add_generated_signal(kinvent_path, generated_signal_path, max_force):
    df_kinvent = pd.read_csv(kinvent_path, skiprows=2)
    df_kinvent_no_zeros = isolate_Target(df_kinvent)
    length_kinvent = len(df_kinvent_no_zeros['Target'])

    generated_signal = read_my_txt_file(generated_signal_path)
    print(generated_signal)
    generated_signal = signal_from_min_to_max(generated_signal, max_force)

    length_generated_signal = len(generated_signal)
    length_erase = length_generated_signal - length_kinvent

    length_generated_signal_erase = length_erase//2 +1
    length_generated_signal_start = length_generated_signal_erase
    print(f'reminder {length_erase % 2}')
    length_generated_signal_end = length_generated_signal - length_generated_signal_erase + length_erase % 2




    print(rf"length_generated_signal_start: {length_generated_signal_start}")
    print(rf"length_generated_signal_end: {length_generated_signal_end}")

    print(rf"length_kinvent: {length_kinvent}")
    print(rf"length_generated_signal_before: {length_generated_signal}")
    print(rf"length_erase: {length_erase}")

    generated_signal = generated_signal[length_generated_signal_start:length_generated_signal_end]
    print(rf"generated_signal: {len(generated_signal)}")

    df_kinvent_no_zeros['Generated_Signal'] = generated_signal

    return df_kinvent_no_zeros

def down_sampling(df, f_out, f_in):
    """
    Parameters
    In
                df:     the dataframe to be downsampled
                f_out:  the frequency I want
                f_in:   the frequency the df has
    Out
                df_downsampled:     the df with downsampled the 'Time' and 'Performance' columns
                                    while 'ClosestSampleTime' and 'Target' were left intact

    """
    factor = int(f_in/f_out)

    df_downsampled_first_two_cols = df[['Time', 'Performance']].iloc[::factor].reset_index(drop=True)
    df_remaining_cols = df[['ClosestSampleTime', 'Target']]
    df_downsampled = pd.concat([df_downsampled_first_two_cols, df_remaining_cols], axis=1)

    return df_downsampled

def outputs(white, pink, sine):
    white_average = np.mean(white)
    pink_average = np.mean(pink)
    sine_average = np.mean(sine)
    list_average = [white_average, pink_average, sine_average]

    white_std = np.std(white)
    pink_std = np.std(pink)
    sine_std = np.std(sine)
    list_std = [white_std, pink_std, sine_std]

    x_axis_white = np.linspace(0, 30, len(white))
    x_axis_pink = np.linspace(0, 30, len(pink))
    x_axis_sine = np.linspace(0, 30, len(sine))

    white_total_load = np.trapz(white, x_axis_white)
    pink_total_load = np.trapz(pink, x_axis_pink)
    sine_total_load = np.trapz(sine, x_axis_sine)
    list_total_load = [white_total_load, pink_total_load, sine_total_load]

    dist = {'Signals': ['White', 'Pink', 'Sine'],
            'Average': list_average,
            'std': list_std,
            'Total_load': list_total_load}
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
            # print('Not valid pink noise signal')
            iterations +=1
            # print(iterations)

    return pink_noise_z

def white_noise_signal_creation(N, desired_sd, desired_average):
    white = False
    iterations = 0
    while white == False:

        white_noise = np.random.normal(0, 1, N)

        white_noise_z = z_transform(white_noise, desired_sd, desired_average)

        slope, positive_freqs_log, positive_magnitude_log, intercept, name, r, p, positive_freqs, positive_magnitude = quality_assessment_of_temporal_structure_FFT_method(white_noise_z, 'white_noise_z')
        print(f"slope: {round(slope, 2)}")
        print(f"r: {r}")
        print(f"p: {p}")

        if round(slope, 2) == 0.0:
            white = True
        else:
            # print('Not valid pink noise signal')
            iterations +=1
            # print(iterations)

    return white_noise_z

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
    # print(f'r_value = {r_value}')
    # print(f'p_value = {p_value}')

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

def perturbation_single_trial_with_random_change(Number_of_data_points, starting_point, ending_point):
    """This function creates a perturbation from the starting point to the ending point, this happens at a random moment
    from the 40% to the 60% of the duration of the perturbation"""

    shift_amount = float(ending_point - starting_point)
    signal = np.full(Number_of_data_points, starting_point)
    low_limit = Number_of_data_points * 0.4
    high_limit = Number_of_data_points * 0.6 + 1
    random_index = np.random.randint(low_limit, high_limit)
    signal[random_index:] = signal[random_index:] + shift_amount

    return signal

def AMI_Stergiou(data, L_seconds, fs, n_bins=0, plot=False):
    """
    inputs    - data, column oriented time series
              - L, maximal lag to which AMI will be calculated
              - bins, number of bins to use in the calculation, if empty an
                adaptive formula will be used
              - to_matlab, an option for MATLAB users of the code, if MATLAB
                datatypes are needed for output, use this to have them
                returned with proper types. Default is false.

                Only use if you have 'matlab.engine' installed in your current
                Python env.

                Note: this cannot be installed through the usual conda or pip
                commands, search online to view resources to help in installing
                'matlab.engine' for Python.

    outputs   - tau, first minimum in the AMI vs lag plot
              - v_AMI, vector of AMI values and associated lags

    inputs    - x, single column array with the same length as y.
              - y, single column array with the same length as x.
    outputs   - ami, the average mutual information between the two arrays

    Remarks
    - This code uses average mutual information to find an appropriate lag
      with which to perform phase space reconstruction. It is based on a
      histogram method of calculating AMI.
    - In the case a value of tau could not be found before L the code will
      automatically re-execute with a higher value of L, and will continue to
      re-execute up to a ceiling value of L.

    Future Work
    - None currently.

    Mar 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Modified code to output a plot and notify the user if a value
                of tau could not be found.
    Sep 2015 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Previously the number of bins was hard coded at 128. This
                created a large amount of error in calculated AMI value and
                vastly decreased the sensitivity of the calculation to changes
                in lag. The number of bins was replaced with an adaptive
                formula well known in statistics. (Scott 1979
              - The previous plot output was removed.
    Oct 2017 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - Added print commands to display progress.
    May 2019 - Modified by Ben Senderling, email unonbcf@unomaha.edu
              - In cases where L was not high enough to find a minimun the
                code would reexecute with a higher L, and the binned data.
                This second part is incorrect and was corrected by using
                data2.
              - The reexecution part did not have the correct input
                parameters.
    Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
    Variability, University of Nebraska at Omaha

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    eps = np.finfo(float).eps  # smallest floating point value

    L = int(L_seconds*fs)
    if isinstance(L, int):
        N = len(data)

        data = np.array(data)

        if n_bins == 0:
            bins = np.ceil((np.max(data) - np.min(data)) / (3.49 * np.nanstd(data * N ** (-1 / 3), axis=0)))
        else:
            bins = n_bins

        bins = int(bins)

        data = data - min(data)  # make all data points positive
        y = np.floor(data / (np.max(data) / (bins - eps)))
        y = np.array(y,
                     dtype=int)  # converts the vector of double vals from data2 into a list of integers from 0 to overlap (where overlap is N-L).

        v = np.zeros((L, 1))  # preallocate the vector
        overlap = N - L
        increment = 1 / overlap

        pA = sp.csr_matrix((np.full(overlap, increment), (y[0:overlap], np.ones(overlap, dtype=int)))).toarray()[:, 1]

        v = np.zeros((2, L))

        for lag in range(L):  # used to be from 0:L-1 (BS)
            v[0, lag] = lag

            pB = sp.csr_matrix(
                (np.full(overlap, increment), (y[lag:overlap + lag], np.ones(overlap, dtype=int)))).toarray()[:, 1]
            # find joint probability p(A,B)=p(x(t),x(t+time_lag))
            pAB = sp.csr_matrix((np.full(overlap, increment), (y[0:overlap], y[lag:overlap + lag])))

            (A, B) = np.nonzero(pAB)
            AB = pAB.data

            v[1, lag] = np.sum(
                np.multiply(AB, np.log2(np.divide(AB, np.multiply(pA[A], pB[B])))))  # Average Mutual Information

        tau = np.array(np.full((L, 2), -1, dtype=float))

        j = 0
        for i in range(v.shape[1] - 1):  # Finds first minimum
            if v[1, i - 1] >= v[1, i] and v[1, i] <= v[1, i + 1]:
                ami = v[1, i]
                tau[j, :] = np.array([i, ami])
                j += 1

        tau = tau[:j]  # only include filled in data.

        # --- Fallback rule if no local minimum was found ---
        initial_AMI = v[1, 0]

        if tau.shape[0] == 0:
            # Try the "20% of initial AMI" rule
            found = False
            for i in range(v.shape[1]):
                if v[1, i] < (0.2 * initial_AMI):
                    # store: [lag_index, AMI_value_at_that_lag]
                    tau = np.array([[i, v[1, i]]], dtype=float)
                    found = True
                    break

            # If still nothing found, return NaNs instead of crashing
            if not found:
                tau = np.array([[np.nan, np.nan]], dtype=float)

        v_AMI = v

        if plot:
            fig, ax = plt.subplots(figsize=(8, 5))

            # AMI curve
            ax.plot(
                v_AMI[0, :],  # lags
                v_AMI[1, :],  # AMI
                marker='o',
                markersize=4,
                linewidth=2.0,
                alpha=0.9,
                label='AMI'
            )

            # Zero lag reference
            ax.axvline(
                x=0,
                color='gray',
                linestyle='--',
                linewidth=1.2,
                alpha=0.7,
                label='Lag = 0'
            )

            # First local minimum (tau)
            if tau.shape[0] > 0 and not np.isnan(tau[0, 0]):
                ax.axvline(
                    x=tau[0][0],
                    color='red',
                    linestyle='--',
                    linewidth=2.5,
                    label=f'τ = {int(tau[0, 0])}\ntime = {int(tau[0, 0])/fs}'
                )

            # Labels & title
            ax.set_xlabel('Time lag (samples)', fontsize=14)
            ax.set_ylabel('Average Mutual Information', fontsize=14)
            ax.set_title('Average Mutual Information vs Time Lag', fontsize=16, pad=12)

            # Ticks
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Grid (subtle)
            ax.grid(True, which='major', axis='y', alpha=0.25)
            ax.grid(False, axis='x')

            # Legend
            ax.legend(
                fontsize=12,
                frameon=False,
                loc='upper right'
            )

            plt.tight_layout()
            plt.show()

        return (tau, v_AMI)
    else:
        raise ValueError('Invalid input, read documentation for input options.')

def SaEn_once_again(data, m, r, tau=None, Theiler_Window=False):
    """
    Computes the Sample Entropy (SampEn) of a one-dimensional time series, with
    optional time-delay embedding and an optional Theiler window.

    When the time delay (tau) is not provided, the function computes the standard
    Sample Entropy using consecutive data points (tau = 1). When tau is specified,
    the function computes time-delay Sample Entropy using delayed embedding
    vectors.

    Sample Entropy estimates the negative logarithm of the conditional probability
    that two sequences similar for m points remain similar when extended to m+1
    points, using the Chebyshev (maximum) norm as the distance metric.

    Parameters
    ----------
    data : array-like
        One-dimensional time series (e.g., center-of-pressure, force, or
        physiological signal).

    m : int
        Embedding dimension (length of the template vectors).
        Typical value is m = 2.

    r : float
        Tolerance parameter, expressed as a proportion of the standard deviation
        of the data (R = r * std(data)).
        Typical value is r = 0.2.

    tau : int, optional
        Time delay (in samples) used to construct delayed embedding vectors.
        If None, tau is set to 1 and the function computes standard Sample Entropy.

    Theiler_Window : bool, optional
        If True, applies a Theiler window to exclude temporally adjacent vectors
        from comparisons. The window size is defined as:

            W = m * tau

        ensuring that overlapping or strongly correlated vectors are not compared.
        The Theiler window requires tau to be explicitly defined.
        If False, all vectors are compared except for self-matches.

    Returns
    -------
    SaEn : float
        Sample Entropy value.

        - Returns a finite positive value when sufficient matches exist.
        - Returns np.inf if no matches are found for vectors of length m+1.
        - Returns np.nan if the time series is too short for the given m and tau.

    Notes
    -----
    - Uses the Chebyshev (maximum) norm to define vector similarity.
    - If tau is not specified, the algorithm reduces to standard Sample Entropy.
    - When the Theiler window is enabled, self-matches and temporally nearby
      vectors are automatically excluded.
    """

    R = r * np.std(data)
    N = len(data)
    data = np.array(data)
    if Theiler_Window and tau is None:
        raise ValueError("Theiler window requires tau to be defined.")

        # --- resolve tau ---
    if tau is None:
        tau = 1
    tau = int(tau)

    # --- Theiler window width ---
    if Theiler_Window:
        W = m * tau

    max_i = N - m*tau
    if max_i <= 1:
        return np.nan

    dij = np.zeros((max_i, m + 1))
    Bm = np.zeros((max_i, 1))
    Am = np.zeros((max_i, 1))

    for i in range(max_i):
        for k in range(m + 1):

            dij[:, k] = np.abs(data[k*tau : k*tau + (N - m*tau)] - data[i + k*tau])
        dj = np.max(dij[:, 0:m], axis=1)
        dj1 = np.max(dij, axis=1)

        if Theiler_Window:
            valid = np.abs(np.arange(max_i) - i) > W

            nm = np.sum((dj <= R) & valid)
            nm1 = np.sum((dj1 <= R) & valid)
        else:
            # original behavior: include all, then remove self-match
            nm  = np.sum(dj  <= R) - 1
            nm1 = np.sum(dj1 <= R) - 1


        Bm[i] = nm / max_i
        Am[i] = nm1 / max_i

    Bmr = np.sum(Bm) / max_i
    Amr = np.sum(Am) / max_i

    if Amr == 0 or Bmr == 0:
        return np.inf
    return -np.log(Amr / Bmr)

def DFA_NONAN(data, scales, order=1, plot=True):
    """Perform Detrended Fluctuation Analysis on data

    Inputs:
        data: 1D numpy array of time series to be analyzed.
        scales: List or array of scales to calculate fluctuations
        order: Integer of polynomial fit (default=1 for linear)
        plot: Return loglog plot (default=True to return plot)

    Outputs:
        scales: The scales that were entered as input
        fluctuations: Variability measured at each scale with RMS
        alpha value: Value quantifying the relationship between the scales
                     and fluctuations

....References:
........Damouras, S., Chang, M. D., Sejdi, E., & Chau, T. (2010). An empirical
..........examination of detrended fluctuation analysis for gait data. Gait &
..........posture, 31(3), 336-340.
........Mirzayof, D., & Ashkenazy, Y. (2010). Preservation of long range
..........temporal correlations under extreme random dilution. Physica A:
..........Statistical Mechanics and its Applications, 389(24), 5573-5580.
........Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995).
..........Quantification of scaling exponents and crossover phenomena in
..........nonstationary heartbeat time series. Chaos: An Interdisciplinary
..........Journal of Nonlinear Science, 5(1), 82-87.
# =============================================================================
                            ------ EXAMPLE ------

      - Generate random data
      data = np.random.randn(5000)

      - Create a vector of the scales you want to use
      scales = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]

      - Set a detrending order. Use 1 for a linear detrend.
      order = 1

      - run dfa function
      s, f, a = dfa(data, scales, order, plot=True)
# =============================================================================
"""

    # Check if data is a column vector (2D array with one column)
    if data.shape[0] == 1:
        # Reshape the data to be a column vector
        data = data.reshape(-1, 1)
    else:
        # Data is already a column vector
        data = data

    # =============================================================================
    ##########################   START DFA CALCULATION   ##########################
    # =============================================================================

    # Step 1: Integrate the data
    integrated_data = np.cumsum(data - np.mean(data))

    fluctuation = []

    for scale in scales:
        # Step 2: Divide data into non-overlapping window of size 'scale'
        chunks = len(data) // scale
        ms = 0.0

        for i in range(chunks):
            this_chunk = integrated_data[i * scale:(i + 1) * scale]
            x = np.arange(len(this_chunk))

            # Step 3: Fit polynomial (default is linear, i.e., order=1)
            coeffs = np.polyfit(x, this_chunk, order)
            fit = np.polyval(coeffs, x)

            # Detrend and calculate RMS for the current window
            ms += np.mean((this_chunk - fit) ** 2)

            # Calculate average RMS for this scale
        fluctuation.append(np.sqrt(ms / chunks))

        # Perform linear regression
    alpha, intercept = np.polyfit(np.log(scales), np.log(fluctuation), 1)

    # Create a log-log plot to visualize the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(scales, fluctuation, marker='o', markerfacecolor='red', markersize=8,
                   linestyle='-', color='black', linewidth=1.7, label=f'Alpha = {alpha:.3f}')
        plt.xlabel('Scale (log)')
        plt.ylabel('Fluctuation (log)')
        plt.legend()
        plt.title('Detrended Fluctuation Analysis')
        plt.grid(True)
        plt.show()

    # Return the scales used, fluctuation functions and the alpha value
    return scales, fluctuation, alpha
