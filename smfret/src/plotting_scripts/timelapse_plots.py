import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from smfret.src.processing_scripts import timelapse_hist_processing as ps
from smfret.src.Utilities import Data_analysis as util
from scipy import optimize, signal
import numpy as np
from scipy import optimize

def plot_data_col(df, thresh, fit_type, xlim, ylim, save_loc, palette, data):
    """Plots the proportion of time each molecule spends below a threshold (defined previously in 1A-plot-histogram) as the mean +- SE as a function of time. This function
    is predominantly designed to collate timelapse data from different experiments and present it within a single plot. It also fits a generic curve to the data, which has been 
    defined above in fit_curve_to_plot function. 

    Args:
        df (dataframe): collated dataset of all treatments to compare
        fit_type (func): the type of fit you wish to plot. If you want a different function, define elsewhere and call the function here to plot.
        xlim (float): minimum x-axis value used to define fit
        ylim (float): maximum x-axis value used to define fit
        data (str, optional): changes if you want to plot normalised data 'normalised' or raw data 'Mean'. Defaults to 'Mean'.
    """
    fig, ax = plt.subplots()
    fit_dict = {}
    for (protein, treatment), dfs in df.groupby(['protein', 'treatment']):
        if data == 'Mean':
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        else:
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        plt.errorbar(dfs['timepoint_plot'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
        sns.scatterplot(data=dfs, x='timepoint_plot', y=data, color=palette[protein], edgecolor='black')
        plt.plot(fitx, fity,'k', color='black')
        fit_dict[(protein, treatment)] = (fit_A, fit_B, fit_A_error, fit_B_error)        
    plt.ylabel(f'Proportion of time spent < {thresh} FRET (mean)')
    plt.xlabel('Time (min)')
    plt.legend(title='')
    plt.savefig(f'{save_loc}/timelapse_proportion_mol_below_thresh.svg', dpi=600)
    plt.show()
    return fit_dict

def plot_data_col_sep(df, thresh, fit_type, xlim, ylim, save_loc, palette, data, markersize):
    """Plots the proportion of time each molecule spends below a threshold (defined previously in 1A-plot-histogram) as the mean +- SE as a function of time. This function
    is predominantly designed to collate timelapse data from different experiments and present it within a single plot. It also fits a generic curve to the data, which has been 
    defined above in fit_curve_to_plot function. 

    Args:
        df (dataframe): collated dataset of all treatments to compare
        fit_type (func): the type of fit you wish to plot. If you want a different function, define elsewhere and call the function here to plot.
        xlim (float): minimum x-axis value used to define fit
        ylim (float): maximum x-axis value used to define fit
        data (str, optional): changes if you want to plot normalised data 'normalised' or raw data 'Mean'. Defaults to 'Mean'.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)
    fit_dict = {}
    for (protein, treatment), dfs in df.groupby(['protein', 'treatment']):
        if data == 'Mean':
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        else:
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)

        if treatment=='KJE':
            ax1.errorbar(dfs['timepoint_plot'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
            sns.scatterplot(data=dfs, x='timepoint_plot', y=data, color=palette[protein], edgecolor='black', ax=ax1, s=markersize)
            ax1.plot(fitx, fity,'k', color='black')
        else:
            ax2.errorbar(dfs['timepoint_plot'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
            sns.scatterplot(data=dfs, x='timepoint_plot', y=data, color=palette[protein], edgecolor='black', ax=ax2, s=markersize)
            ax2.plot(fitx, fity,'k', color='black')


        fit_dict[(protein, treatment)] = (fit_A, fit_B, fit_A_error, fit_B_error)        

# Remove individual y-axis labels
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Add a centered y-axis label across both subplots
    fig.text(0.04, 0.5, f'Proportion of time spent < {thresh} FRET (mean)', va='center', rotation='vertical')        
    plt.xlabel('Time (min)')
    plt.legend(title='')
    plt.savefig(f'{save_loc}/timelapse_proportion_mol_below_thresh.svg', dpi=600)
    plt.show()
    return fit_dict



def timelapse_mapping(directory, timepoint):
    """
    Loads and processes FRET histogram and filtered data for timelapse plotting.
    Parameters
    ----------
    directory : str
        Path to the directory containing 'Cleaned_FRET_histogram_data.csv' and 'filt_dfs.csv'.
    timepoint : dict
        Dictionary mapping timepoint labels (as found in 'treatment_name') to numeric values for plotting.
    Returns
    -------
    compiled_df : pandas.DataFrame
        DataFrame containing processed FRET histogram data with added columns:
        - 'timepoint': extracted from 'treatment_name'
        - 'protein': extracted from 'treatment_name'
        - 'timepoint_plot': mapped numeric timepoint value
        - 'time_rand': randomized value for plotting
    filt_dfs : pandas.DataFrame
        DataFrame containing processed filtered data with the same added columns as above.
    Notes
    -----
    - Assumes 'treatment_name' column exists in both CSV files and follows the format 'protein_xxx_timepoint'.
    - The 'time_rand' column is generated by sampling uniformly between the mapped timepoint and (timepoint - 6).
    """
    compiled_df = pd.read_csv(f'{directory}/Cleaned_FRET_histogram_data.csv')
    compiled_df['timepoint'] = compiled_df['treatment_name'].str.split('_').str[-1]
    compiled_df['protein'] = compiled_df['treatment_name'].str.split('_').str[0]
    compiled_df['timepoint_plot'] = compiled_df['timepoint'].map(timepoint)
    compiled_df['time_rand'] = compiled_df['timepoint_plot'].apply(lambda y: np.random.uniform(y, y - 6))

    filt_dfs = pd.read_csv(f'{directory}/filt_dfs.csv')
    filt_dfs['timepoint'] = filt_dfs['treatment_name'].str.split('_').str[-1]
    filt_dfs['protein'] = filt_dfs['treatment_name'].str.split('_').str[0]
    filt_dfs['timepoint_plot'] = filt_dfs['timepoint'].map(timepoint)
    filt_dfs['time_rand'] = filt_dfs['timepoint_plot'].apply(lambda y: np.random.uniform(y, y - 6))
    return compiled_df, filt_dfs

def heatmap_timelapse(output_folder, compiled_df, palette, str_identifier='KJEG', x='time_rand', y='FRET'):
    """
    Generates and saves heatmap timelapse plots for each protein in the provided DataFrame.
    For each protein, two subplots are created:
        - The first subplot visualizes the kernel density estimation (KDE) of samples 
          not containing the specified treatment identifier.
        - The second subplot visualizes the KDE of samples containing the specified treatment identifier.
    Both subplots display the density of the specified x and y columns (default: 'time_rand' and 'FRET').
    The resulting plots are saved as SVG files in the specified output folder.
    Args:
        output_folder (str): Path to the folder where the output SVG files will be saved.
        compiled_df (pd.DataFrame): DataFrame containing the data to plot. Must include columns for 'protein',
            'treatment_name', and the specified x and y columns.
        palette (dict): Dictionary mapping protein names to colormap names or color palettes for plotting.
        str_identifier (str, optional): String identifier to filter treatment names. Default is 'KJEG'.
        x (str, optional): Column name to use for the x-axis. Default is 'time_rand'.
        y (str, optional): Column name to use for the y-axis. Default is 'FRET'.
    Saves:
        SVG files named 'histogram_heatmap_{protein}.svg' in the output_folder for each protein.
    """
    for protein, df in compiled_df.groupby('protein'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, sharey=True)
        sns.kdeplot(data=df[~df['treatment_name'].str.contains(str_identifier)], x=x, y=y, cmap='Greys', shade=bool, cbar=False, cbar_kws={'format': '%.0f%%', 'ticks': [0, 50]}, gridsize=50, levels=6, thresh=0.05, ax=ax1)
        sns.kdeplot(data=df[~df['treatment_name'].str.contains(str_identifier)], x=x, y=y, color='grey', gridsize=50, levels=6, thresh=0.05, ax=ax1)

        sns.kdeplot(data=df[df['treatment_name'].str.contains(str_identifier)], x=x, y=y, cmap=palette[protein], shade=bool, cbar=False, cbar_kws={'format': '%.0f%%', 'ticks': [0, 25]}, gridsize=50, levels=6, thresh=0.05, ax=ax2)
        sns.kdeplot(data=df[df['treatment_name'].str.contains(str_identifier)], x=x, y=y, color='black', gridsize=50, levels=6, thresh=0.05, ax=ax2)

        fig.text(0.04, 0.5, y, va='center', rotation='vertical', fontsize=12)
        plt.tight_layout(rect=[0.06, 0, 1, 1])
        plt.ylim(-0.2, 1.2)
        plt.xlabel('Time (s)')
        plt.savefig(f'{output_folder}/histogram_heatmap_{protein}.svg', dpi=600)
        plt.show()


def plot_rate_constants(output_folder, timelapse_colors, fit_dict, x='protein', y='rate_constant'):
    """
    Plots bar charts of rate constants with error bars for different proteins and treatments.
    Parameters
    ----------
    output_folder : str
        Path to the folder where the output plots (SVG and PNG) will be saved.
    timelapse_colors : dict
        Dictionary mapping protein names to their corresponding colors for plotting order.
    fit_dict : dict
        Dictionary containing fit results. Keys should be tuples of (protein, treatment), and values should contain
        fit parameters including 'fit_A', 'rate_constant', 'fit_A_error', and 'rate_constant_error'.
    x : str, optional
        Column name to use for the x-axis (default is 'protein').
    y : str, optional
        Column name to use for the y-axis (default is 'rate_constant').
    Returns
    -------
    fit_df_sorted : pandas.DataFrame
        DataFrame containing the sorted fit results used for plotting.
    Saves
    -----
    rate_constant.svg, rate_constant.png : Plots saved in the specified output folder.
    Notes
    -----
    - The function uses seaborn and matplotlib for plotting.
    - Error bars are added manually to each bar to represent the 'rate_constant_error'.
    - The order of proteins in the plot is determined by the keys of `timelapse_colors`.
    """
    order = list(timelapse_colors.keys())
    fit_df = pd.DataFrame(fit_dict).transpose().reset_index()
    fit_df.columns = ['protein', 'treatment', 'fit_A', 'rate_constant', 'fit_A_error', 'rate_constant_error']
    fit_df['protein'] = pd.Categorical(fit_df['protein'], categories=order, ordered=True)
    # Now sort the dataframe by the 'protein' column (and optionally by 'treatment')
    fit_df_sorted = fit_df.sort_values(by=['protein', 'treatment']).reset_index(drop=True)
    # now label the plot position of each bar to match with errorbars
    fit_df_sorted['bar_pos']=fit_df.index


    plt.figure()
    ax = sns.barplot(
        x='protein', 
        y='rate_constant', 
        hue='treatment', 
        data=fit_df_sorted, 
        palette='muted', 
        ci=None, 
        edgecolor='black'
    )

    # Get the positions of the bars
    bar_positions = [(p.get_x() + 0.5 * p.get_width(), p.get_height()) for p in ax.patches]
    # Create a mapping of (protein, treatment) to error values
    error_mapping = fit_df_sorted.set_index(['protein', 'treatment'])['rate_constant_error'].to_dict()
    # Iterate over the DataFrame in the same order as the bars
    for i, (protein, treatment, bar_pos) in enumerate(fit_df_sorted.groupby(['protein', 'treatment', 'bar_pos']).groups.keys()):
        # Get the corresponding error value
        error = error_mapping.get((protein, treatment), 0)  # Default to 0 if not found
        # Height of the current bar
        x_position, height = bar_positions[bar_pos]
        # Place the error bar at the center of the current bar
        ax.errorbar(
            x_position,  # center of the bar
            height,  # height of the bar
            yerr=error,  # corresponding error
            fmt='none',  # no marker
            ecolor='black',  # color of the error bar
            capsize=5,  # size of caps on error bars
            elinewidth=2  # width of error bar lines
        )
    # Set labels and title
    plt.ylabel('Rate constant')
    plt.xlabel('')
    plt.legend(title='')
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{output_folder}/rate_constant.svg', dpi=600)
    plt.savefig(f'{output_folder}/rate_constant.png', dpi=600)
    plt.show()
    return fit_df_sorted


# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_timelapse_func(filt_dfs, thresh=0.3, xlim_min=6, xlim_max=60, output_folder='Experiment_1-description/python_results', palette='BuPu', data_type='Mean', markersize=5, split=True):
    filt_dfs['protein'] = filt_dfs['treatment_name'].str.split('_').str[0]
    filt_dfs['treatment'] = filt_dfs['treatment_name'].str.split('_').str[1]
    timepoint_plotdata = ps.concat_df_fit_data(filt_dfs, output_folder)
    if split==False:
        fit_dict = plot_data_col(timepoint_plotdata, thresh, ps.guess_exponential, xlim_min, xlim_max, output_folder, palette, data_type)
    else:
        fit_dict = plot_data_col_sep(timepoint_plotdata, thresh, ps.guess_exponential, xlim_min, xlim_max, output_folder, palette, data_type, markersize)
    return filt_dfs, timepoint_plotdata, fit_dict



