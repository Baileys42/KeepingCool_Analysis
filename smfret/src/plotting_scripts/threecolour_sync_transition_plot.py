import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from smfret.src.processing_scripts import threecolour_processing as ps


def plot_synchronised_transition_3color(dfs, index_to_plot, exposure_seconds, list_to_drop, order, frame_from_trans, save_loc, label=''):
    """plots the FRET values either side of a transition type of interest

    Args:
        dfs (df): dataframe containing the raw FRET values (generated after the calculate dwells function)
        index_to_plot (list): list of index values that met the criteria defined in 'filt_df_to_plot' that will the be mapped to df for plotting of raw FRET values
        exposure_seconds (float): exposure in seconds used to convert frames to a unit of time
        list_to_drop (list): list containing the name of all treatments to be dropped
        frame_from_trans (int, optional): should be the same as 'min_dwell_before' variable in the 'filt_df_to_plot' function, basically sets the xlims. Defaults to 80.
    """
    combined_mini = []
    for df in index_to_plot:
        print(df)
        lower = df - frame_from_trans
        upper = df + (frame_from_trans+1)
        mini_df = dfs.iloc[lower:upper].reset_index()
        mini_df['time_from_trans'] = np.arange(-(frame_from_trans/(1/exposure_seconds)), (frame_from_trans/(1/exposure_seconds))+exposure_seconds, exposure_seconds)
        combined_mini.append(mini_df)
    combined_mini = pd.concat(combined_mini)
    filt_data = combined_mini[~combined_mini['treatment'].isin(list_to_drop)]
    fig, axes = plt.subplots()
    sns.set(style='ticks')
    sns.lineplot(data=filt_data, x='time_from_trans', y='FRET Cy3 to AF647', hue='treatment', palette='BuPu', hue_order=order)
    plt.xlabel('Time (s)')
    plt.legend(title='',loc='best')
    fig.savefig(f'{save_loc}/synchronised_release{"_"+label}.svg', dpi=600)
    plt.show()


def plot_synchronised_fluorescence_3color(dfs, index_to_plot, exposure_seconds, list_to_drop, frame_from_trans, save_loc, label=''):
    """plots the FRET values and total fluorescence of all dyes following excitation at 488 nm either side of a transition type of interest

    Args:
        dfs (df): dataframe containing the raw FRET values (generated after the calculate dwells function)
        index_to_plot (list): list of index values that met the criteria defined in 'filt_df_to_plot' that will the be mapped to df for plotting of raw FRET values
        exposure_seconds (float): exposure in seconds used to convert frames to a unit of time
        list_to_drop (list): list containing the name of all treatments to be dropped
        frame_from_trans (int, optional): should be the same as 'min_dwell_before' variable in the 'filt_df_to_plot' function, basically sets the xlims. Defaults to 80.
    """
    combined_mini = []
    for df in index_to_plot:
        print(df)
        lower = df - frame_from_trans
        upper = df + (frame_from_trans+1)
        mini_df = dfs.iloc[lower:upper].reset_index()
        mini_df['time_from_trans'] = np.arange(-(frame_from_trans/(1/exposure_seconds)), (frame_from_trans/(1/exposure_seconds))+exposure_seconds, exposure_seconds)
        combined_mini.append(mini_df)
    combined_mini = pd.concat(combined_mini)
    filt_data = combined_mini[~combined_mini['treatment'].isin(list_to_drop)]
    for treatment, df in filt_data.groupby('treatment'):
        fig, axes = plt.subplots()
        sns.set(style='ticks')
        ax2 = axes.twinx()
        sns.lineplot(data=df, x='time_from_trans', y='FRET Cy3 to AF647', color='black', ax=axes)
        sns.lineplot(data=df, x='time_from_trans', y='normalised_summed_fluorescence', color='skyblue', ax=ax2)
        axes.set_xlabel('Time (s)')
        ax2.set_ylabel('Normalised total fluorescence (a.u.)')  
        axes.set_ylabel('FRET')  
        axes.legend(['FRET'], loc='upper left')
        ax2.legend(['Fluorescence'])
        fig.savefig(f'{save_loc}/{treatment}_synchronised_release_fluorescence{"_"+label}.svg', dpi=600)
        plt.show()

# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_3color_synch_transitions(output_folder='Experiment_1-description/python_results', exposure=0.2, frames_to_plot=50, FRET_before=0.5, FRET_after=0.5, order=['treatment1', 'treatment2'], list_to_drop=['']):
    plot_export = f'{output_folder}/synchronised_transitions/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)
    compiled_df_HMM = pd.read_csv(f'{output_folder}/compiled_df_HMM.csv')
    compiled_df_HMM_dropped = compiled_df_HMM.dropna()

    calculated_transitions = []
    for treatment, df in compiled_df_HMM_dropped.groupby('treatment'):
        dwell_df = ps.calculate_dwells_3color(df)
        transition_df = ps.generate_transitions_3color(dwell_df)
        calculated_transitions.append(transition_df)
    calculated_transitions_df = pd.concat(calculated_transitions)
    calculated_transitions_df.reset_index(inplace=True)
    calculated_transitions_df.drop('index', axis=1, inplace=True)

    font = {'weight' : 'normal', 'size'   : 12 }
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'

    dnak_stable_release = ps.filt_df_to_plot_3color(calculated_transitions_df, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    plot_synchronised_transition_3color(calculated_transitions_df, dnak_stable_release, exposure, list_to_drop, order, frames_to_plot, save_loc=plot_export, label='release')
    

    bottle = []
    for (treatment, molecule), df in calculated_transitions_df.groupby(['treatment', 'cumulative_molecule']):
        df['normalised_summed_fluorescence'] = df['probe_summed_fluorescence']/df['probe_summed_fluorescence'].max()
        bottle.append(df)
    calculated_transitions_df_normalised = pd.concat(bottle)

    dnak_stable_release_norm = ps.filt_df_to_plot_3color(calculated_transitions_df_normalised, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    dnak_stable_binding_norm = ps.filt_df_to_plot_3color(calculated_transitions_df_normalised, FRET_before, FRET_after, 'high_to_low', frames_to_plot)

    plot_synchronised_fluorescence_3color(calculated_transitions_df_normalised, dnak_stable_release_norm, exposure, list_to_drop, frames_to_plot, save_loc=plot_export, label='release')
    plot_synchronised_fluorescence_3color(calculated_transitions_df_normalised, dnak_stable_binding_norm, exposure, list_to_drop, frames_to_plot, save_loc=plot_export, label='binding')

