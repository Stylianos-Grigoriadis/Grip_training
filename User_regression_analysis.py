import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
import lib
from Lib_grip import spatial_error


stylianos = True
if stylianos == True:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Results'
else:
    directory = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Results'
os.chdir(directory)

spatial_error_resutls = pd.read_excel(r'Training_trials_results.xlsx')
perturbation_results = pd.read_excel(r'Perturbation results\Sd Method Perturbation_results_3_sd_after_max_threshold.xlsx')

spatial_error_resutls = spatial_error_resutls[spatial_error_resutls['Exclude'] == 0].reset_index(drop=True).copy()
perturbation_results = perturbation_results[perturbation_results['Exclude'] == 0].reset_index(drop=True).copy()
print(perturbation_results.columns)

trial_cols = [
    'Mean Spatial error trail 1',
    'Mean Spatial error trail 2',
    'Mean Spatial error trail 3',
    'Mean Spatial error trail 4',
    'Mean Spatial error trail 5',
    'Mean Spatial error trail 6',
    'Mean Spatial error trail 7',
    'Mean Spatial error trail 8',
    'Mean Spatial error trail 9',
    'Mean Spatial error trail 10'
]

avg_spatial_error = spatial_error_resutls[['ID']].copy()
avg_spatial_error['Mean Spatial Error'] = spatial_error_resutls[trial_cols].mean(axis=1)


trial_cols = [
    'Mean Spatial error trail 1',
    'Mean Spatial error trail 2',
    'Mean Spatial error trail 3',
    'Mean Spatial error trail 4',
    'Mean Spatial error trail 5',
    'Mean Spatial error trail 6',
    'Mean Spatial error trail 7',
    'Mean Spatial error trail 8',
    'Mean Spatial error trail 9',
    'Mean Spatial error trail 10'
]


x = np.arange(1, 11)

slopes = []

for i in range(len(spatial_error_resutls)):
    y = spatial_error_resutls.iloc[i][trial_cols].values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    slopes.append(slope)
    # plt.figure(figsize=(5, 4))
    # plt.scatter(x, y, color='k', zorder=3, label='Data')
    # plt.plot(x, y_fit, color='r', lw=2, label='Slope fit')
    #
    # plt.xlabel('Trial')
    # plt.ylabel('Mean Spatial Error')
    # plt.title(f'Participant {spatial_error_resutls["ID"].iloc[i]}')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

spatial_error_resutls['Slope'] = slopes

correlation_results_spatial_slop_vs_before_down = lib.correlation_analysis(avg_spatial_error['Mean Spatial Error'], perturbation_results['Min Time to adapt before down'], plot=True)
correlation_results_spatial_slop_vs_before_up = lib.correlation_analysis(avg_spatial_error['Mean Spatial Error'], perturbation_results['Min Time to adapt before up'], plot=True)
correlation_results_spatial_slop_vs_after_down = lib.correlation_analysis(avg_spatial_error['Mean Spatial Error'], perturbation_results['Min Time to adapt after down'], plot=True)
correlation_results_spatial_slop_vs_after_up = lib.correlation_analysis(avg_spatial_error['Mean Spatial Error'], perturbation_results['Min Time to adapt after up'], plot=True)
print(correlation_results_spatial_slop_vs_before_down)
print(correlation_results_spatial_slop_vs_before_up)
print(correlation_results_spatial_slop_vs_after_down)
print(correlation_results_spatial_slop_vs_after_up)


