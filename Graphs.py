#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:28:24 2024

@author: JuliusSiebenaller
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.utils import resample
from scipy.interpolate import interp1d  # For smooth interpolation


# Load data (replace with actual data source)
data = pd.read_csv('/Users/JuliusSiebenaller/Documents/GitHub/ShockExposure/cleaned_data.csv')  # Uncomment and adjust as needed

# Function for local polynomial regression
def local_poly_regression(x, y, degree=3, frac=0.3, x_eval=None):
    if x_eval is None:
        x_eval = np.linspace(min(x), max(x), 500)  # High-resolution x-values
    fit = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    fit_interp = interp1d(fit[:, 0], fit[:, 1], fill_value="extrapolate")
    return x_eval, fit_interp(x_eval)

# Function to compute bootstrap confidence intervals
def bootstrap_ci(x, y, degree=3, frac=0.3, n_bootstrap=1000, ci=95, x_eval=None):
    if x_eval is None:
        x_eval = np.linspace(min(x), max(x), 500)  # High-resolution x-values

    boot_coefs = np.zeros((n_bootstrap, len(x_eval)))
    for i in range(n_bootstrap):
        x_resampled, y_resampled = resample(x, y)
        _, boot_fit = local_poly_regression(x_resampled, y_resampled, degree=degree, frac=frac, x_eval=x_eval)
        boot_coefs[i, :] = boot_fit

    # Calculate lower and upper percentiles for the confidence intervals
    lower_bound = np.percentile(boot_coefs, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(boot_coefs, 100 - (100 - ci) / 2, axis=0)
    
    return lower_bound, upper_bound

# Perform local polynomial regression for 'hta' and 'wta'
# Data
childage = data['childage']
hta = data['hta']
wta = data['wta']

# Perform local polynomial regression and get the smoothed points
x_eval = np.linspace(min(childage), max(childage), 500)  # High-resolution x-values
hta_fit_x, hta_fit_y = local_poly_regression(childage, hta, degree=3, frac=0.3, x_eval=x_eval)
wta_fit_x, wta_fit_y = local_poly_regression(childage, wta, degree=3, frac=0.3, x_eval=x_eval)

# Compute confidence intervals using bootstrap
hta_lower, hta_upper = bootstrap_ci(childage, hta, degree=3, frac=0.3, x_eval=x_eval)
wta_lower, wta_upper = bootstrap_ci(childage, wta, degree=3, frac=0.3, x_eval=x_eval)


# Plotting
plt.figure(figsize=(10, 6))

# Plot HAZ with confidence intervals
plt.plot(hta_fit_x, hta_fit_y, label='HAZ', color='blue')
plt.fill_between(hta_fit_x, hta_lower, hta_upper, color='grey', alpha=0.3, linestyle='--', label='95% CI (HAZ)')

# Plot WAZ with confidence intervals
plt.plot(wta_fit_x, wta_fit_y, label='WAZ', color='green')
plt.fill_between(wta_fit_x, wta_lower, wta_upper, color='lightgray', alpha=0.3, linestyle='--', label='95% CI (WAZ)')

# Customize plot
plt.title('WHO Z-score by Age of Child')
plt.xlabel('Age of the child in months')
plt.ylabel('WHO Z-score')
plt.legend()
plt.grid(True, linestyle='--', color='lightgray')

# Save plot in different formats
plt.savefig('ageprofiles02.eps', format='eps')
plt.savefig('ageprofiles02.pdf', format='pdf')
plt.show()


# -------------------------------------------------------------
# Affected region
# Define subgroups and cases to be analyzed
subgroup_column = 'ar'  # Column that holds subgroups (e.g., 'affected region')
cases = {'HAZ': 'hta', 'WAZ': 'wta'}  # Dictionary of cases to plot (e.g., height-for-age, weight-for-age)

# Plotting loop for subgroups and cases
plt.figure(figsize=(12, 8))

# Loop over each subgroup (e.g., different regions or urban/rural)
for subgroup, subgroup_data in data.groupby(subgroup_column):
    # Define x_eval (same for all cases)
    x_eval = np.linspace(subgroup_data['childage'].min(), subgroup_data['childage'].max(), 500)

    # Loop over each case (e.g., HAZ, WAZ)
    for case_name, case_column in cases.items():
        y = subgroup_data[case_column]
        x = subgroup_data['childage']
        
        # Perform local polynomial regression and get the smoothed points
        case_fit_x, case_fit_y = local_poly_regression(x, y, degree=3, frac=0.3, x_eval=x_eval)
        
        # Compute confidence intervals using bootstrap
        case_lower, case_upper = bootstrap_ci(x, y, degree=3, frac=0.3, x_eval=x_eval)
        
        # Plot the results for this subgroup and case
        plt.plot(case_fit_x, case_fit_y, label=f'{case_name} ({subgroup})')
        plt.fill_between(case_fit_x, case_lower, case_upper, alpha=0.3, label=f'95% CI {case_name} ({subgroup})')

# Customize the plot
plt.title('WHO Z-score by Age of Child by affected region')
plt.xlabel('Age of the child in months')
plt.ylabel('WHO Z-score')
plt.legend()
plt.grid(True, linestyle='--', color='lightgray')

# Save plot in different formats
plt.savefig('ageprofiles_multiple_cases.eps', format='eps')
plt.savefig('ageprofiles_multiple_cases.pdf', format='pdf')
plt.show()


# -------------------------------------------------------------
# Gender
# Define subgroups and cases to be analyzed
subgroup_column = 'male'  # Column that holds subgroups (e.g., 'born during')
cases = {'HAZ': 'hta', 'WAZ': 'wta'}  # Dictionary of cases to plot (e.g., height-for-age, weight-for-age)

# Plotting loop for subgroups and cases
plt.figure(figsize=(12, 8))

# Loop over each subgroup (e.g., different regions or urban/rural)
for subgroup, subgroup_data in data.groupby(subgroup_column):
    # Define x_eval (same for all cases)
    x_eval = np.linspace(subgroup_data['childage'].min(), subgroup_data['childage'].max(), 500)

    # Loop over each case (e.g., HAZ, WAZ)
    for case_name, case_column in cases.items():
        y = subgroup_data[case_column]
        x = subgroup_data['childage']
        
        # Perform local polynomial regression and get the smoothed points
        case_fit_x, case_fit_y = local_poly_regression(x, y, degree=3, frac=0.3, x_eval=x_eval)
        
        # Compute confidence intervals using bootstrap
        case_lower, case_upper = bootstrap_ci(x, y, degree=3, frac=0.3, x_eval=x_eval)
        
        # Plot the results for this subgroup and case
        plt.plot(case_fit_x, case_fit_y, label=f'{case_name} ({subgroup})')
        plt.fill_between(case_fit_x, case_lower, case_upper, alpha=0.3, label=f'95% CI {case_name} ({subgroup})')

# Customize the plot
plt.title('WHO Z-score by Age of Child by gender')
plt.xlabel('Age of the child in months')
plt.ylabel('WHO Z-score')
plt.legend()
plt.grid(True, linestyle='--', color='lightgray')

# Save plot in different formats
plt.savefig('ageprofiles_gender_cases.eps', format='eps')
plt.savefig('ageprofiles_gender_cases.pdf', format='pdf')
plt.show()


# -------------------------------------------------------------
# Urban
# Define subgroups and cases to be analyzed
subgroup_column = 'urban'  # Column that holds subgroups (e.g., 'born during')
cases = {'HAZ': 'hta', 'WAZ': 'wta'}  # Dictionary of cases to plot (e.g., height-for-age, weight-for-age)

# Plotting loop for subgroups and cases
plt.figure(figsize=(12, 8))

# Loop over each subgroup (e.g., different regions or urban/rural)
for subgroup, subgroup_data in data.groupby(subgroup_column):
    # Define x_eval (same for all cases)
    x_eval = np.linspace(subgroup_data['childage'].min(), subgroup_data['childage'].max(), 500)

    # Loop over each case (e.g., HAZ, WAZ)
    for case_name, case_column in cases.items():
        y = subgroup_data[case_column]
        x = subgroup_data['childage']
        
        # Perform local polynomial regression and get the smoothed points
        case_fit_x, case_fit_y = local_poly_regression(x, y, degree=3, frac=0.3, x_eval=x_eval)
        
        # Compute confidence intervals using bootstrap
        case_lower, case_upper = bootstrap_ci(x, y, degree=3, frac=0.3, x_eval=x_eval)
        
        # Plot the results for this subgroup and case
        plt.plot(case_fit_x, case_fit_y, label=f'{case_name} ({subgroup})')
        plt.fill_between(case_fit_x, case_lower, case_upper, alpha=0.3, label=f'95% CI {case_name} ({subgroup})')

# Customize the plot
plt.title('WHO Z-score by Age of Child by urban vs rural region')
plt.xlabel('Age of the child in months')
plt.ylabel('WHO Z-score')
plt.legend()
plt.grid(True, linestyle='--', color='lightgray')

# Save plot in different formats
plt.savefig('ageprofiles_gender_cases.eps', format='eps')
plt.savefig('ageprofiles_gender_cases.pdf', format='pdf')
plt.show()



"""
twoway lpolyci hta childage if ar==0, degree(3) legend(label(1 "95% CI") label(2 "not born in the affected region")) clcolor(gs0) alcolor(gs3) alpattern(dash) fcolor(none)||lpolyci hta childage if ar==1, degree(3) clcolor(gs7) alcolor(gs10) alpattern(dash) fcolor(none) legend(label(3 "95% CI") label(4 "born in the affected region")) ytitle(Height for age WHO Z-score) xtitle(Age of the child in months) plotregion(fcolor(white)) graphregion(lcolor(black) fcolor(white)) bgcolor(none) ylabel(, grid glcolor(gs13))
graph save ar02, replace
graph export ar02.eps, replace
graph export ar02.pdf, replace
"""





