import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from deepfret_nm import DeepFRET_NM
from smfret.src.Utilities import Data_analysis as util
from smfret.src.processing_scripts import threecolour_processing as ps 

# ------------------- plots two graphs, one with the PPR FRET and another with whatever FRET you want to look at --------------------

def plot_FRET_multiple(df, top_FRET, bottom_FRET, save_loc):
    data_hist = df[(df[top_FRET] > -0.2) & (df[top_FRET] < 1.2) & (df[bottom_FRET] > -0.2) & (df[bottom_FRET] < 1.2)]
    fig, axes = plt.subplots(2, 1, sharex= True)
    sns.set_style("ticks",{'figure.figsize':(8,5.5),'font_scale':1.5} )
    sns.histplot(data=data_hist, x=top_FRET, binrange=(0, 1), binwidth=0.05, kde=True, ax=axes[0], stat='density', fill=False, hue='treatment', common_norm=False, palette='gray')
    sns.histplot(data=data_hist, x=bottom_FRET, binrange=(0, 1), binwidth=0.05, kde=True, ax=axes[1], stat='density', fill=False, hue='treatment', common_norm=False, palette='gray')
    for x in axes:
        [y.set_linewidth(1) for y in x.spines.values()]
        [y.set_color('black') for y in x.spines.values()]
    axes[0].set_title(top_FRET)
    axes[1].set_title(bottom_FRET)
    plt.xlabel('FRET')
    plt.xlim(0, 1)
    plt.savefig(f'{save_loc}/FRET-histograms.svg', dpi=600)
    plt.show()



# ------ Plots FRET distribution. 'Bar' is to plot multiple treatments in the one graph 'anything else' is to plot each treatment independently. -------

def plot_hist_type(df, save_loc, kind='kde', palette='BuPu'):
    plot_hist, ax = plt.subplots()
    sns.set_style("ticks",{'font_scale':1} )
    plt.xlim(0, 1, 10)
    plt.xlabel("FRET")
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    if kind == 'kde':
        sns.kdeplot(
            data=df, 
            palette=palette, 
            x="FRET Cy3 to AF647",
            hue="treatment",
            common_norm=False, 
            fill=True, 
            linewidth=1.5, 
            alpha=0.25)
        plt.show()
    if kind == 'bar':
        sns.histplot(
            data=df, 
            x="FRET Cy3 to AF647",
            common_norm=False, 
            stat='density',
            hue='treatment',
            palette=palette,
            binwidth=0.05,
            fill=False, 
            kde=True,
            linewidth=1.5, 
            alpha=0.25)
        plt.show()
    else: 
        for treatment, dfs in df.groupby('treatment'):
            df_filt = dfs[dfs['treatment']==treatment]
            sns.histplot(
                data=df_filt, 
                x="FRET Cy3 to AF647",
                common_norm=False, 
                stat='density',
                color='black',
                binwidth=0.05,
                fill=False, 
                kde=True,
                linewidth=1.5, 
                alpha=0.25)
            plt.xlim(0, 1, 10)
            plt.xlabel("FRET")
            [x.set_linewidth(2) for x in ax.spines.values()]
            [x.set_color('black') for x in ax.spines.values()]
            plot_hist.savefig(f'{save_loc}/Histogram_{treatment}.svg', dpi=600)
            plt.show()
    plot_hist.savefig(f'{save_loc}/Histogram_{kind}.svg', dpi=600)
    

# ---------- plots the fluorescence intensity of a single dye for each oligo in the PPR bound or unbound state --------------------------

def plot_intensity_for_treatments(df, intensity_type, palette):
    treatment_list = list(df['treatment'].unique())
    fig, axes = plt.subplots(len(treatment_list), 1, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]  # Convert single Axes object to a list
    sns.set_style('ticks',{'grid.linestyle':'--', 'font_scale': 1.5})
    for (i, label) in enumerate(treatment_list):
        treatment_data=df[df['treatment'] == label]
        sns.histplot(data=treatment_data, 
                     x=intensity_type, 
                     kde=True, 
                     stat='density', 
                     fill=False, 
                     common_norm=False, 
                     palette=palette[label], 
                     binwidth=250,
                     binrange=(-5000, 20000),
                     hue='bound',
                     ax=axes[i])
        axes[i].set_title(f'{label}')
        axes[i].legend(['Unbound', 'RNA-bound'])   
    axes[0].set_xlabel(intensity_type)
    plt.tight_layout()
    plt.xlim(-5000, 20000)
    plt.show()
    return fig, axes



# --------- For each treatment it will plot the fluorescence intensity of the dyes excited by 488 nm laser in the bound and unbound PPR state --------

def plot_each_intensity(df, treatment, save_loc, palette='BuPu'):
    treatment_list = ['AF488 at 488', 'Cy3 at 488', 'AF647 at 488']
    fig, axes = plt.subplots(3, 1, sharex=True)
    sns.set_style('ticks',{'grid.linestyle':'--', 'font_scale': 1.5})
    for (i, label) in enumerate(treatment_list):
        sns.histplot(data=df, 
                     x=label, 
                     kde=True, 
                     stat='density', 
                     fill=False, 
                     common_norm=False, 
                     palette=palette[label], 
                     binwidth=500,
                     binrange=(-5000, 20000),
                     hue='bound',
                     hue_order=['RNA-bound', 'Unbound'],
                     ax=axes[i], 
                     legend=True)
        axes[i].set_title(f'{label} for {treatment}')
    axes[2].set_xlabel('Total fluorescence (a.u.)')
    plt.tight_layout()
    plt.xlim(-5000, 20000)
    fig.savefig(f'{save_loc}/intensity_bound_{treatment}.svg', dpi=600)
    plt.show()
    return fig, axes


def master_fit_HMM(dict_to_concat, output_folder="Experiment_1-description/python_results/", FRET_thresh=0.5):
    saved_folders = f'{output_folder}/organized_csvs/'
    if not os.path.exists(saved_folders):
        os.makedirs(saved_folders)

    compiled_df = []
    for data_name, data_path in dict_to_concat.items():
        imported_data = util.file_reader_3colour(data_path, 'hist')
        imported_data["treatment"] = data_name
        compiled_df.append(imported_data)
    compiled_df = pd.concat(compiled_df) 
    compiled_df['Cy3 FRET'] = compiled_df['Cy3 at 488']/(compiled_df['Cy3 at 488']+compiled_df['AF488 at 488']+compiled_df['AF647 at 488'])
    compiled_df['AF647 FRET'] = compiled_df['AF647 at 488']/(compiled_df['Cy3 at 488']+compiled_df['AF488 at 488']+compiled_df['AF647 at 488'])
    compiled_df['Cy3 FRET cascade'] = (compiled_df['AF647 at 488'] + compiled_df['Cy3 at 488'])/(compiled_df['Cy3 at 488']+compiled_df['AF488 at 488']+compiled_df['AF647 at 488'])
    compiled_df['AF647 FRET cascade'] = (compiled_df['AF647 at 488'])/(compiled_df['Cy3 at 488']+compiled_df['AF647 at 488'])
    compiled_df['Cy3 FRET Lee'] = (compiled_df['Cy3 at 488'])/(compiled_df['Cy3 at 488']+(compiled_df['AF488 at 488']*(1 - compiled_df['FRET Cy3 to AF647'])))
    compiled_df['probe_summed_fluorescence'] = compiled_df['Cy3 at 488'] + compiled_df['AF488 at 488'] + compiled_df['AF647 at 488']

    FRET_bound = compiled_df[compiled_df['FRET Cy3 to AF647']>FRET_thresh]
    compiled_df['bound'] = np.where(compiled_df['FRET Cy3 to AF647']>FRET_thresh, 'RNA-bound', 'Unbound')
    print(FRET_bound.groupby('treatment')['cumulative_molecule'].nunique())
    print(compiled_df.groupby('treatment')['cumulative_molecule'].nunique())
    compiled_df_filt = compiled_df.drop(['Frame at 532'], axis=1)

    list(compiled_df_filt.columns)
    compiled_df_filt = compiled_df_filt[[
    'Time at 532',
    'Cy3 at 532',
    'AF647 at 532',
    'AF488 at 532',
    'Time at 488',
    'Frame at 488',
    'AF488 at 488',
    'Cy3 at 488',
    'AF647 at 488',
    'FRET AF488 to Cy3',
    'Idealized FRET AF488 to Cy3',
    'FRET AF488 to AF647',
    'Idealized FRET AF488 to AF647',
    'FRET Cy3 to AF647',
    'Idealized FRET Cy3 to AF647',
    'molecule number',
    'movie_number',
    'cumulative_molecule',
    'treatment',
    'Cy3 FRET',
    'AF647 FRET',
    'Cy3 FRET cascade',
    'AF647 FRET cascade',
    'Cy3 FRET Lee',
    'probe_summed_fluorescence',
    'bound'
    ]]

    for (treatment, molecule), df in compiled_df_filt.groupby(['treatment', 'cumulative_molecule']):
        df_filt = df[['Time at 532', 'Cy3 at 532', 'AF647 at 532']]
        df_filt.to_csv(f'{saved_folders}/{treatment}_molecule{molecule}.dat', index=False, header=False, sep=' ')

    compiled_HMM = []
    for (treatment, molecule), df in compiled_df_filt.groupby(['treatment', 'cumulative_molecule']):
        df
        trace = DeepFRET_NM.import_data(f'{output_folder}/organized_csvs/{treatment}_molecule{molecule}.dat')
        traces = [trace]
        df['e_pred_global'] = DeepFRET_NM.fit_HMM_NM(traces)['e_pred_global']
        compiled_HMM.append(df)
    compiled_df_HMM = pd.concat(compiled_HMM)
    compiled_df_HMM.to_csv(f'{output_folder}/compiled_df_HMM.csv', index=False)
    print(compiled_df_HMM)


def plot_three_colour_traces(output_folder):
    plot_export = f'{output_folder}/traces/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)

    compiled_df_HMM = pd.read_csv(f'{output_folder}/compiled_df_HMM.csv')

    for (treatment, molecule), df in compiled_df_HMM.groupby(['treatment','cumulative_molecule']):
        fig, ax = plt.subplots(3, 1, sharex=True)
        sns.set_style("whitegrid",{'figure.figsize':(8,5.5), 'grid.linestyle':'--', 'font_scale':1.5} )
        
        sns.lineplot(data=df, x='Time at 532', y='Cy3 at 532', ax=ax[0], color='green')
        sns.lineplot(data=df, x='Time at 532', y='AF647 at 532', ax=ax[0], color='purple')

        sns.lineplot(data=df, x='Time at 488', y='AF488 at 488', ax=ax[1], color='royalblue')
        sns.lineplot(data=df, x='Time at 488', y='Cy3 at 488', ax=ax[1], color='olivedrab')
        sns.lineplot(data=df, x='Time at 488', y='AF647 at 488', ax=ax[1], color='orange')
        sns.lineplot(data=df, x='Time at 488', y='probe_summed_fluorescence', ax=ax[1], color='darkgrey')

        sns.lineplot(data=df, x='Time at 532', y='FRET Cy3 to AF647', ax=ax[2], color='black')
        sns.lineplot(data=df, x='Time at 532', y='e_pred_global', ax=ax[2], color='orange')

        ax[2].set_ylim(0, 1)
        ax[2].set_xlim(0, 300)
        for x in ax:
            [y.set_linewidth(2) for y in x.spines.values()]
            [y.set_color('black') for y in x.spines.values()]
        ax[0].set_ylabel('')
        ax[1].set_ylabel('')
        ax[2].set_ylabel('FRET')
        ax[2].set_xlabel('Time (s)')
        fig.text(0.04, 0.65, 'Fluorescence intensity (a.u.)', ha='center', va='center', rotation='vertical')
        ax[0].set_title(f'{treatment} molecule {molecule}')
        plot_dir = f'{plot_export}/{treatment}/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        fig.savefig(f'{plot_dir}/molecule_{molecule}_from{treatment}.svg', dpi=600)
    plt.show()


def plot_3colour_fret_hist(output_folder="Experiment_1-description/python_results/", FRET_subplot1='Cy3 FRET cascade', FRET_subplot2='AF647 FRET cascade', FRET_thresh=0.5, palette='BuPu'):
    compiled_df_HMM, FRET_bound, plot_export = ps.annotate_data(output_folder=output_folder, FRET_thresh=FRET_thresh)
    plot_FRET_multiple(FRET_bound, FRET_subplot1, FRET_subplot2, save_loc=plot_export)
    plot_hist_type(compiled_df_HMM, plot_export, 'bar', palette)
    return 

def plot_3colour_by_treatment(output_folder="Experiment_1-description/python_results/", FRET_thresh=0.5, dye='AF647 at 488', palette='BuPu'):
    compiled_df_HMM, FRET_bound, plot_export = ps.annotate_data(output_folder, FRET_thresh)

    fig, axes = plot_intensity_for_treatments(compiled_df_HMM, dye, palette)

def plot_3colour_by_dye(output_folder="Experiment_1-description/python_results/", FRET_thresh=0.5, palette='BuPu'):
    compiled_df_HMM, FRET_bound, plot_export = ps.annotate_data(output_folder=output_folder, FRET_thresh=FRET_thresh)

    for treatment, df in compiled_df_HMM.groupby('treatment'):
        fig, axes = plot_each_intensity(df, treatment, plot_export, palette)

