import numpy as np
from scipy import optimize, signal
import numpy as np
from scipy import optimize
import pandas as pd

# ---------------------- Equation for exponential, can always define new equation and call in the plot data col function ---------------

def guess_exponential(x,A,B):
    y = A*np.exp(-B*x)
    return y

# --------- Generic function used to fit curve to plot. The fit should be easily changed by calling a different equation into 'fit_type). -----------------

def fit_curve_to_plot(df, fit_type, x, y, data):
    est_x = df['timepoint_plot'].to_numpy()
    est_y = df[f'{data}'].to_numpy()
    # actual code for fitting
    paramaters, covariance = optimize.curve_fit(fit_type, est_x, est_y)
    fit_A = paramaters[0]
    fit_B = paramaters[1]
    print(fit_A)
    print(fit_B)
    perr = np.sqrt(np.diag(covariance))
    fit_A_error = perr[0]  # Error for fit_A
    fit_B_error = perr[1]  # Error for fit_B (rate constant)

    # Print parameters and their standard deviations
    print(f"fit_A (amplitude): {fit_A} ± {fit_A_error}")
    print(f"fit_B (rate constant): {fit_B} ± {fit_B_error}")
    #plotting data from fit
    fit_x_values = np.linspace(x,y,1000)
    fit_y = fit_type(fit_x_values, fit_A, fit_B)
    return fit_x_values, fit_y, fit_A, fit_B, fit_A_error, fit_B_error


def concat_df_fit_data(df, save_loc):
    mean_data = df.groupby(['treatment_name', 'timepoint_plot', 'protein', 'treatment'])['FRET_time_below_thresh'].mean().reset_index()
    std_err_data = df.groupby(['treatment_name', 'timepoint_plot', 'protein', 'treatment'])['FRET_time_below_thresh'].sem().reset_index()
    test = pd.merge(mean_data, std_err_data, on=['treatment_name', 'timepoint_plot', 'protein', 'treatment'])
    test.rename(columns={'FRET_time_below_thresh_x':'Mean','FRET_time_below_thresh_y':'Std_error'},inplace=True)
    test = test.dropna()
    test = test.sort_values(by=['timepoint_plot']).reset_index()
    norm = []
    for (protein, treatment), dfs in test.groupby(['protein', 'treatment']):
        dfs['normalised'] = (dfs['Mean']/(dfs['Mean'].iloc[0]))*100
        norm.append(dfs)
    normed = pd.concat(norm)       
    normed.drop('index', axis=1, inplace=True)
    normed.to_csv(f'{save_loc}/mean.csv')
    return normed

