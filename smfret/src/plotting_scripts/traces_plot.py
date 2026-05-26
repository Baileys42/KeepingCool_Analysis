import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from scipy.signal import savgol_filter
import os
from smfret.src.processing_scripts import trace_processing as ps


# ------------------ If you want to plot all molecules with a time threshold greater than a defined point ------------------------

def plot_FRET_test(df, molecule, min_trace_length):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_style("whitegrid", {'grid.linestyle':'--'})
    plot2, ax = plt.subplots(figsize=(7, 1.8))
    plt.xlim(0, min_trace_length, 10)
    plt.ylim(0, 1.1, 0.2)
    sns.lineplot(x=df["Time"], y=df["smoothed_FRET"], color='black')
    sns.lineplot(x=df["Time"], y=df["idealized FRET"], color='darkorange')
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plt.xlabel("Time (s)")
    plt.ylabel("FRET")
    plt.title(f'{molecule}')
    plt.show()
    return plot2, ax

def plot_all(dfs):
    for treatment, df in dfs.groupby('treatment_name'):
        """Plots all molecules with a threshold longer than a defined length. It is arranged so that each plot
    is always 4 molecules wide and as long as there are molecules to fill.
    Args:
        dfincompiled_df (_type_): _description_
        dfincompiled_df_long_renumbered (_type_): _description_
        ax (_type_, optional): _description_. Defaults to plt.subplots(figsize=(5, 2))plt.xlim(0, 200, 10)plt.ylim(0, 1.1, 0.2)sns.lineplot(x=df["Time"], y=df["smoothed_FRET"], color='black')sns.lineplot(x=df["Time"], y=df["idealized FRET"], color='darkorange')[x.set_linewidth(2) for x in ax.spines.values()][x.set_color('black') for x in ax.spines.values()]plt.xlabel("Time (s)")plt.ylabel("FRET")plt.title(f'{treatment}-{molecule}')plt.show()returnplot2.
        ax (_type_, optional): _description_. Defaults to plot_FRET_test(df, treatment, molecule)plot.savefig(f'{output_folder}/{treatment}_Trace_{molecule}.svg', dpi=600)else:continuefilt_data=[]for(treatment, molecule).
    """
        df_filt_mol_list = list(df['Change_Count'].unique())
        nrow = math.ceil(len(df_filt_mol_list)/4)
        fig, axes = plt.subplots(nrow, 4, figsize=(16, 2*nrow), sharex=True, sharey=True)
        sns.set_style('ticks')
        axes = axes.flatten()
        for i, mol_number in enumerate(df_filt_mol_list):
            data = df[df['Change_Count'] == mol_number]
            sns.lineplot(x=data["Time"], y=data["smoothed_FRET"], color='black', ax=axes[i])
            sns.lineplot(x=data["Time"], y=data["idealized FRET"], color='darkorange', ax=axes[i])
            axes[i].axvline(x=0, linestyle='--', color='grey')
            plt.xlim(0, 200)
            plt.ylim(0, 1)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('FRET')
            axes[i].set_title(f"{treatment}")
        plt.show()


# ------------------- Code to plot FRET and/or intensity traces -------------------------

def plot_intensity(treatment):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_style("whitegrid", {'grid.linestyle':'--'})
    plot1, ax = plt.subplots(figsize=(5, 2))
    plt.xlim(0, 150, 10)
    plt.ylim(0, 4000, 0.2)
    sns.lineplot(x=treatment["Time"], y=treatment["donor"], color='green')
    sns.lineplot(x=treatment["Time"], y=treatment["acceptor"], color='purple')
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plt.xlabel("Time (s)")
    plt.ylabel("FRET")
    plt.show()
    return plot1

def plot_FRET(treatment):
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_style("whitegrid", {'grid.linestyle':'--'})
    plot2, ax = plt.subplots(figsize=(5, 2))
    plt.xlim(0, 200, 10)
    plt.ylim(0, 1.1, 0.2)
    sns.lineplot(x=treatment["Time"], y=treatment["smoothed_FRET"], color='black')
    sns.lineplot(x=treatment["Time"], y=treatment["idealized FRET"], color='darkorange')
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plt.xlabel("Time (s)")
    plt.ylabel("FRET")
    plt.show()
    return plot2

def plot_both(df):
    fig, ax = plt.subplots(2, 1, sharex=True)
    sns.set(style='ticks', font_scale=1)
    sns.lineplot(x=df["Time"], y=df["smoothed_FRET"], color='black', ax=ax[1])
    sns.lineplot(x=df["Time"], y=df["idealized FRET"], color='darkorange', ax=ax[1])
    sns.lineplot(x=df["Time"], y=df["donor"], color='green', ax=ax[0])
    sns.lineplot(x=df["Time"], y=df["acceptor"], color='purple', ax=ax[0])
    plt.xlim(0, 200)
    ax[1].set_ylabel('FRET')
    ax[0].set_ylabel('Intensity (a.u.)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    return fig


# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_plot_traces_func(input_folder, exposure=0.2, min_trace_length=300):
    output_folder = f'{input_folder}/Traces'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    compiled_df = pd.read_csv(f'{input_folder}/Cleaned_FRET_histogram_data.csv')
    compiled_df['Time'] = compiled_df['frames']/(1/exposure)
    compiled_df["smoothed_FRET"] = savgol_filter(compiled_df["FRET"], 5, 1) 

    for (treatment, molecule), df in compiled_df.groupby(['treatment_name', 'unique_id']):
        if df.Time.iloc[-1] > min_trace_length:
            plot, ax = plot_FRET_test(df,molecule, min_trace_length)
            plot.savefig(f'{output_folder}/{treatment}_Trace_{molecule}.svg', dpi=600)
        else:
            continue

    test = []
    for (treatment, molecule), df in compiled_df.groupby(['treatment_name', 'unique_id']):
        if df.Time.iloc[-1] > min_trace_length:
            test.append(df)
        else:
            continue
        testies = pd.concat(test)

    renumbered_mol = []
    for treatment, df in testies.groupby('treatment_name'):
        mask = df['unique_id'].ne(df['unique_id'].shift())
        df['Change_Count'] = mask.cumsum()
        renumbered_mol.append(df)
    compiled_df_long_renumbered = pd.concat(renumbered_mol)
    plot_all(compiled_df_long_renumbered)
    return 

def master_plot_individual_trace(data_paths,input_folder, exposure=0.2):
    output_folder = f'{input_folder}/Traces'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    compiled_df = []
    for data_name, data_path in data_paths.items():
        imported_data = ps.load_data(data_path, exposure)
        imported_data["treatment_name"] = data_name
        compiled_df.append(imported_data)
    compiled_df = pd.concat(compiled_df) 

    for data_name, data_path in data_paths.items():
        treatment = compiled_df[compiled_df["treatment_name"] == data_name]
        mol_ident = data_path.split('/')[-1].split('_')[0]
        plot_both(treatment).savefig(f'{output_folder}/{data_name}_Trace_{mol_ident}_both.svg', dpi=600)
        plot_FRET(treatment).savefig(f'{output_folder}/{data_name}_Trace_{mol_ident}.svg', dpi=600)
        plot_intensity(treatment).savefig(f'{output_folder}/{data_name}_Trace_{mol_ident}_intensity.svg', dpi=600)
