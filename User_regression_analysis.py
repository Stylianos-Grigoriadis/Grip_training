import Lib_grip as lb
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
from Lib_grip import spatial_error


stylianos = True
if stylianos == True:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training\Data\Valid data\Force data\Pink_65.2\Training_trials'
else:
    directory = r'C:\Users\USER\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\Grip training\Data\Valid data\Force data\Pink_65.2\Training_trials'
os.chdir(directory)

# data = pd.read_csv(, skiprows=2)
# print(data.columns)
# plt.plot(data['Performance'])
# plt.show()


df1 = pd.read_csv(r"Trial_1.csv", skiprows=2)
df2 = pd.read_csv(r"Trial_2.csv", skiprows=2)
df3 = pd.read_csv(r"Trial_3.csv", skiprows=2)
df4 = pd.read_csv(r"Trial_4.csv", skiprows=2)
df5 = pd.read_csv(r"Trial_5.csv", skiprows=2)
df6 = pd.read_csv(r"Trial_6.csv", skiprows=2)
df7 = pd.read_csv(r"Trial_7.csv", skiprows=2)
df8 = pd.read_csv(r"Trial_8.csv", skiprows=2)
df9 = pd.read_csv(r"Trial_9.csv", skiprows=2)
df10 = pd.read_csv(r"Trial_10.csv", skiprows=2)

pd.set_option('display.max_rows', None)

new_df1 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df1)
new_df2 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df2)
new_df3 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df3)
new_df4 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df4)
new_df5 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df5)
new_df6 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df6)
new_df7 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df7)
new_df8 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df8)
new_df9 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df9)
new_df10 = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(df10)

spatial_errors1 = spatial_error(new_df1)
spatial_errors2 = spatial_error(new_df2)
spatial_errors3 = spatial_error(new_df3)
spatial_errors4 = spatial_error(new_df4)
spatial_errors5 = spatial_error(new_df5)
spatial_errors6 = spatial_error(new_df6)
spatial_errors7 = spatial_error(new_df7)
spatial_errors8 = spatial_error(new_df8)
spatial_errors9 = spatial_error(new_df9)
spatial_errors10 = spatial_error(new_df10)


combined_spatial_errors = list(itertools.chain(spatial_errors1, spatial_errors2, spatial_errors3, spatial_errors4, spatial_errors5, spatial_errors6, spatial_errors7, spatial_errors8, spatial_errors9, spatial_errors10))

# Finding the length of each list of spatial errors
spatial_error_lengths = [len(spatial_errors1), len(spatial_errors2), len(spatial_errors3), len(spatial_errors4), len(spatial_errors5),len(spatial_errors6), len(spatial_errors7), len(spatial_errors8), len(spatial_errors9), len(spatial_errors10)]
print(spatial_error_lengths)

# Exclude last cumulative sum as it's the total length
spatial_error_indices = np.cumsum(spatial_error_lengths)[:-1]

# Plot the combined spatial errors
plt.plot(combined_spatial_errors, label='Spatial Errors', linewidth=1)
# Add red vertical lines at the transition points
for idx in spatial_error_indices:
    plt.axvline(x=idx, color='black', linestyle='--', linewidth=1, label="Trial Boundary" if idx == spatial_error_indices[0] else "")

plt.xlabel("Targets")
plt.ylabel("Spatial Error")
plt.title("Spatial Errors Across Trials")
plt.legend()
plt.show()

spatial_errors_list = [spatial_errors1,
spatial_errors2,
spatial_errors3,
spatial_errors4,
spatial_errors5,
spatial_errors6,
spatial_errors7,
spatial_errors8,
spatial_errors9,
spatial_errors10
]


## Perform separate linear regressions
#for i, spatial_errors in enumerate(spatial_errors_list, start=1):
#    x = np.arange(len(spatial_errors))  # X values (index)
#    y = np.array(spatial_errors)  # Y values (spatial errors)
#
#    slope, intercept, r_value, p_value, std_err = linregress(x, y)
#    y_pred = slope * x + intercept
#
#    # Plot results for each trial
#    plt.figure()
#    plt.scatter(x, y, label=f'Trial {i} Spatial Errors', color='blue', alpha=0.5)
#    plt.plot(x, y_pred, color='red', label='Regression Line')
#    plt.xlabel('Index')
#    plt.ylabel('Spatial Error')
#    plt.title(f'Linear Regression - Trial {i}')
#    plt.legend()
#    plt.show()
#
#    # Print regression results
#    print(f'Trial {i}:')
#    print(f'  Coefficient (slope): {slope}')
#    print(f'  Intercept: {intercept}')
#    print(f'  R-squared: {r_value ** 2}')
#    print('-' * 40)


# Combined Spatial Errors Regression
combined_spatial_errors = list(itertools.chain(*spatial_errors_list))
x_combined = np.arange(len(combined_spatial_errors))
y_combined = np.array(combined_spatial_errors)

slope_combined, intercept_combined, r_value_combined, p_value_combined, std_err_combined = linregress(x_combined, y_combined)
y_pred_combined = slope_combined * x_combined + intercept_combined

# Plot combined regression
plt.figure()
plt.scatter(x_combined, y_combined, label='Combined Spatial Errors', color='blue', alpha=0.5)
plt.plot(x_combined, y_pred_combined, color='red', label='Regression Line')
plt.xlabel('Index')
plt.ylabel('Spatial Error')
plt.title('Linear Regression on Combined Spatial Errors')
plt.legend()
plt.show()

# Print combined regression results
print('Combined Spatial Errors:')
print(f'  Coefficient (slope): {slope_combined}')
print(f'  Intercept: {intercept_combined}')
print(f'  R-squared: {r_value_combined**2}')

