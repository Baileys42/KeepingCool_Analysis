import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from smfret.src.Utilities import Data_analysis as util
from smfret.src.processing_scripts import kinetics_processing as ps


def plot_fret_trans(df, order, save_loc, FRET_state='after', to_drop='none', threshold=0.5, palette='BuPu'):
    """Function to plot the FRET state before or after a transition above or below a defined FRET state

    Args:
        df (dataframe): dataframe that contains the concatenated dataset of all treatments, should be TDP_data
        FRET_state (str, optional): Will determine whether or not you are looking at the FRET state 'before' or 'after' the transition. Defaults to 'after'.
        to_drop (str, optional): Can input a list with the datasets that you want to drop from the plot. Will need to use those categories within the 'treatment_name' column within df. Defaults to 'none'.
        threshold (_type_, optional): The FRET state that determines the kind of transitions you are looking at. If set to 0.3, and FRET_state is='before', this will plot the FRET state before transition to below 0.3 FRET. Defaults to Transition_threshold.
        palette (str, optional): Choose colour scheme to plot. Defaults to 'mako'.
    """
    sns.set(style="ticks", font_scale=1)
    if to_drop == 'none':
        if FRET_state == 'after':
            plot1, ax = plt.subplots()
            sns.set(style="ticks", font_scale=1.5)
            sns.violinplot(data=df, x='treatment_name', y='FRET_after', palette=palette, order=order)
            sns.stripplot(data=df, x='treatment_name', y='FRET_after', color='black', alpha=0.25, order=order)
            plt.ylabel(f'FRET state after transition from < {threshold}')
        elif FRET_state == 'before':
            plot1, ax = plt.subplots()
            sns.set(style="ticks", font_scale=1.5)
            sns.violinplot(data=df, x='treatment_name', y='FRET_before', palette=palette, order=order)
            sns.stripplot(data=df, x='treatment_name', y='FRET_before', color='black', alpha=0.25, order=order)
            plt.ylabel(f'FRET state before transition to < {threshold}')
    else:
        dropped = df[~df['treatment_name'].isin(to_drop)].dropna()
        plot1, ax = plt.subplots()
        sns.set(style="ticks", font_scale=1.5)
        sns.violinplot(data=dropped, x='treatment_name', y='FRET_before')
        sns.stripplot(data=dropped, x='treatment_name', y='FRET_before', color='black', alpha=0.25)
    plt.rcParams['svg.fonttype']='none'
    plt.xlabel('Treatment')
    plt.ylim(-0.1, 1.2)
    plt.xticks(rotation=45)
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plt.xlabel('')
    plot1.savefig(f'{save_loc}/FRET_{FRET_state}_trans_{threshold}.svg', dpi=600)
    plt.show()

 
def plot_binding_release(df, order, save_loc, chaperone='binding', palette='BuPu'):
    # sourcery skip: switch
    """Plots the number or rate of chaperone binding and/or release events per molecule

    Args:
        df (dataframe): dataframe that contains the number of 'binding' or 'release' events per molecule normalised to duration of molecule. Done by using 'count_chaperone_events' function and subsequent code.
        chaperone (str, optional): string that determine what to plot. Can input any of the 'if' chaperone == options. Defaults to 'binding'.
        order (bool, optional): defines what order to plot datasets. Defaults to False.
        palette (str, optional): what palette to use when plotting. Defaults to 'mako'.
    """
    if chaperone == 'binding':
        ycol = 'FRET_after_normalised'
        ylabel = '# of chaperone binding events/min/molecule'
        title = 'chaperone_binding_rate_per_molecule'
    if chaperone == 'release':
        ycol = 'FRET_before_normalised'
        ylabel = '# of chaperone release events/min/molecule'
        title = 'chaperone_release_rate_per_molecule'
    if chaperone == 'binding_events':
        ycol = 'FRET_after'
        ylabel = '# of chaperone binding events/molecule'
        title = 'chaperone_binding_events_per_molecule'
    if chaperone == 'binding_and_release':
        ycol = 'bind_and_release_overtime'
        ylabel = '# of chaperone binding and release events/molecule/min'
        title = 'chaperone_binding_and_release_events_per_molecule_min'
    sns.set(style="ticks", font_scale=1)
    plot1, ax = plt.subplots()
    sns.set(font_scale=1.5, style='ticks')
    sns.violinplot(data=df, y=ycol, x='treatment', cut=0, order=order, palette=palette, scale='width')
    sns.stripplot(data=df, y=ycol, x='treatment', color='black', alpha=0.25, order=order)
    plt.xticks(rotation=45)
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plt.ylabel(f'{ylabel}')
    plt.xlabel('')
    plot1.savefig(f'{save_loc}/{title}.svg', dpi=600)
    plt.show()
   

def plot_large_transitions(df, order, save_loc, type='transition_prob', palette='BuPu'):
    sns.set(style="ticks", font_scale=1)
    if type == 'transition_prob': 
        ycol = 'proportion_of_large_transitions'
    if type == 'proportion_of_mol': 
        ycol = 'proportion_mol_large_transition'
    plot, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df, 
        y=ycol, 
        x='treatment',
        order=order, 
        palette=palette, 
        edgecolor='black')
    plt.xticks(rotation=45)
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plt.xlabel('')
    plot.savefig(f'{save_loc}/{ycol}.svg', dpi=600)
    plt.show()


def plot_violin(data, save_loc, palette='BuPu', scale="y_axis"):
    if scale == "y_axis":
        plt.rcParams['svg.fonttype'] = 'none'
        plot1 = plt.figure()
        sns.set(style="ticks", font_scale=1)
        sns.violinplot(
            data=data, 
            x= "transition_name", 
            y="y_axis",
            palette= palette, 
            hue="treatment", 
            log_scale=True,
            cut=0)
        plt.ylabel("Residence time (s)")
        plt.xlabel("Transition class")
        plt.legend(title='',loc="upper right")
        plot1.savefig(f"{save_loc}/Violin_plot_normal.svg", dpi=600)
        plt.show()
    if scale == "y_axis_log10":
        plt.rcParams['svg.fonttype']='none'
        plot2 = plt.figure()
        sns.set(style="ticks", font_scale=1)
        sns.violinplot(
            data=data, 
            x= "transition_name", 
            y="y_axis_log10",
            palette= 'mako', 
            hue="treatment", 
            log_scale=True)
        plt.ylabel("Log residence time (s)")
        plt.xlabel("Transition class")
        plt.legend(title='',loc="upper left", bbox_to_anchor=(1,1), ncol =1)
        plot2.savefig(f"{save_loc}/Violin_plot_log.svg", dpi=600)
        plt.show()
    if scale == 'split':     
        f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})
        sns.set(style='ticks')
        sns.violinplot(x="transition_name", y="y_axis", hue="treatment",data=data, ax=ax_top, palette='BuPu', scale='width')
        sns.violinplot(x="transition_name", y="y_axis", hue="treatment",data=data, ax=ax_bottom, cut=0, palette='BuPu', scale='width')
        ax_top.set_ylim(bottom=40)   # those limits are fake
        ax_bottom.set_ylim(0,40)
        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs=dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        #remove one of the legend
        ax_bottom.legend_.remove()
        ax_top.set_xlabel('')
        plt.xlabel('')
        ax_top.tick_params(bottom=False)
        ax_top.set_ylabel('')
        ax_bottom.set_ylabel('')
        f.text(0.04, 0.5, 'Residence time (s)', ha='center', va='center', rotation='vertical')
        ax_top.legend(title='')
        f.savefig(f"{save_loc}/Violin_plot_splitaxis.svg", dpi=600)
        plt.show()


def plot_bar_with_sem(df, summary_df, order, save_loc, y_axis='y_axis', palette='mako'):
    ###### prepare order of datasets so that sem are correctly mapped onto the correct dataset
    list_to_order = list(np.arange(0, len(order), 1))
    dict_to_order = dict(zip(order, list_to_order))
    summary_df['plot_order'] = summary_df['treatment'].map(dict_to_order)
    collated_sorted = summary_df.sort_values(['plot_order', 'transition_type'])
    sorted_df = df.sort_values(['treatment', 'transition_type'])
    ###### now plot the figure
    fig, ax = plt.subplots()
    sns.set(style='ticks', font_scale=1)
    sns.barplot(x='transition_name', y=y_axis, data=sorted_df, hue='treatment', palette=palette, ci =None, hue_order=order, edgecolor='black')
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=collated_sorted["sem"], fmt="none",  elinewidth=2, capsize=4, color='black')
    plt.ylabel('Residence time (s)')
    plt.xlabel('')
    plt.legend(title='')
    fig.savefig(f'{save_loc}/mean_residence_withSEM_{y_axis}.svg', dpi=600)
    plt.show()


def plot_residence_time_of_class(df, binwidth, transition, save_log, plot_type='KDE', log=False):
    if plot_type == 'KDE':
        for transition, dfs in df.groupby('transition_type'):
            filt_trans = dfs[dfs['transition_type'] ==  transition]
            sns.kdeplot(data=filt_trans, x='y_axis', hue='treatment', fill=False, log_scale=True, common_norm=False, palette='mako')
            plt.title(transition)
            plt.show()
    if plot_type == 'individual':
        for treatment, dfs in df.groupby('treatment'):
            fig, axes = plt.subplots(10, 1, figsize=(8, 18), sharex=True)
            axes = axes.flatten()
            for i, transition in enumerate(list(dfs['transition_type'].unique())):
                fig = sns.histplot(data=dfs[dfs['transition_type']==transition], x='y_axis', binwidth=binwidth, kde=True, stat='density', log_scale=log, ax=axes[i])
                axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
                axes[i].set_title(f'{treatment} and {transition}')
                plt.xlim(0, 50)
            plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
            plt.show()
    if plot_type == 'collated':
        fig, axes = plt.subplots(10, 1, sharex=True)
        axes = axes.flatten()
        for i, treatment in enumerate(list(df['treatment'].unique())):
            dfs = df[df['treatment']==treatment]
            df2 = dfs[dfs['transition_type']==transition]
            fig = sns.histplot(
                data=df2, 
                hue='treatment', 
                x='y_axis',
                binwidth=binwidth, 
                stat='density', 
                log_scale=log, 
                ax=axes[i], 
                common_norm=False, 
                fill=False, 
                palette='mako')
            axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
            axes[i].set_title(f'{treatment}', loc='center')
            plt.xlim(0, 200)
        plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
        plt.show()
    if plot_type == 'cum_dwell':
        fig, axes = plt.subplots(10, 1, sharex=True)
        axes = axes.flatten()
        for i, treatment in enumerate(list(df['treatment'].unique())):
            dfs = df[df['treatment']==treatment]
            df2 = dfs[dfs['transition_type']==transition]
            fig = sns.histplot(
                data=df2, 
                hue='treatment', 
                x='CumulativeTime(s)',
                binwidth=binwidth, 
                kde=True, 
                stat='density', 
                log_scale=log, 
                ax=axes[i], 
                common_norm=False, 
                fill=False, 
                palette='mako')
            axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
            axes[i].set_title(f'{treatment}')
            axes[i].legend('')
            plt.xlim(0, 200)
        plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
        plt.show()
    plt.show


def transition_frequency_plot(output_folder, order, FRET_thresh=0.5):
    headers_withsum = [f"< {FRET_thresh} to < {FRET_thresh}", f"< {FRET_thresh} to > {FRET_thresh}", f"> {FRET_thresh} to > {FRET_thresh}", f"> {FRET_thresh} to < {FRET_thresh}", "sum", "sample"]
    imported_df = util.file_reader(f'{output_folder}/Dwell_frequency', 'transition_frequency', False, headers_withsum) ##### NEED TO ALTER EACH TIME
    imported_df.drop(columns=["sum"], inplace=True)
    imported_df = imported_df.reindex(order) ### need to do this to plot in order that you would like
    imported_df.to_csv(f'{output_folder}/Transition_frequency.csv')

    # -------------- code to prepare for plotting ---------------------
    total = [i+j+k+m for i,j,k, m in zip(imported_df[f'< {FRET_thresh} to < {FRET_thresh}'], imported_df[f'< {FRET_thresh} to > {FRET_thresh}'], imported_df[f'> {FRET_thresh} to > {FRET_thresh}'], imported_df[f"> {FRET_thresh} to < {FRET_thresh}"])]
    lowtolow = [i / j * 100 for i,j in zip(imported_df[f'< {FRET_thresh} to < {FRET_thresh}'], total)]
    lowtohigh = [i / j * 100 for i,j in zip(imported_df[f'< {FRET_thresh} to > {FRET_thresh}'], total)]
    hightohigh = [i / j * 100 for i,j in zip(imported_df[f'> {FRET_thresh} to > {FRET_thresh}'], total)]
    hightolow = [i / j * 100 for i,j in zip(imported_df[f'> {FRET_thresh} to < {FRET_thresh}'], total)]
    # ----------------- plotting code -----------------------
    labels = imported_df["sample"].to_list()
    barWidth = 0.85
    plt.rcParams['svg.fonttype']='none'
    sns.set(style="ticks", color_codes='pastel')
    # --------------- Create green Bars ---------------------
    plot1, ax = plt.subplots()
    plt.bar(labels, lowtolow, color='skyblue', edgecolor='black', width=barWidth, label=f"< {FRET_thresh} to < {FRET_thresh}" )
    # ------------- Create orange Bars ----------------------
    plt.bar(labels, lowtohigh, bottom=lowtolow, color='royalblue', edgecolor='black', width=barWidth, label=f"< {FRET_thresh} to > {FRET_thresh}")
    # -------------- Create blue Bars ----------------------
    plt.bar(labels, hightohigh, bottom=[i+j for i,j in zip(lowtolow, lowtohigh)], color='mediumpurple', edgecolor='black', width=barWidth, label=f"> {FRET_thresh} to > {FRET_thresh}")
    # --------------- Create blue Bars ---------------------
    plt.bar(labels, hightolow, bottom=[i+j+k for i,j,k in zip(lowtolow, lowtohigh, hightohigh)], color='indigo', edgecolor='black', width=barWidth,label=f"> {FRET_thresh} to < {FRET_thresh}")
    plt.legend(title='', loc="upper left", bbox_to_anchor=(1,1), ncol =1)
    plt.ylabel("Transition probability (%)")
    plt.xticks(rotation=45)
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plot1.savefig(f'{output_folder}/Transition_frequency_plot.svg', dpi=600)
    plt.show()

# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_dwell_time_func(order, output_folder='Experiment_1-description/python_results', palette_to_use='BuPu', FRET_thresh=0.5, thresh_for_events=0.5, fps=5, thresh=0.8, Transition_threshold=0.5, event='binding_and_release'):
    
    plot_folder = f'{output_folder}/dwell_analysis_figs' 
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    filename = f'{output_folder}/TDP_cleaned.csv'
    
    headers = [f"< {FRET_thresh} to < {FRET_thresh}", f"< {FRET_thresh} to > {FRET_thresh}", f"> {FRET_thresh} to > {FRET_thresh}", f"> {FRET_thresh} to < {FRET_thresh}"]
    TDP_data = pd.read_csv(filename, header="infer")

    for treatment_name, df in TDP_data.groupby("treatment_name"):
        initial_data = df[df["treatment_name"] == treatment_name]
        cleaned_data = util.cleanup_dwell(initial_data, fps, thresh, 'keep') ##### to keep the first dwell state, simply change code to "cleanup_dwell(initial_data, "keep")
        filtered_data = util.filter_dwell(cleaned_data, FRET_thresh, headers)
        filtered_data.to_csv(f"{output_folder}/Dwell_times/Filtered_dwelltime_{treatment_name}.csv", index=False)
        mean_dwell = util.calculate_mean(filtered_data, treatment_name)
        mean_dwell.to_csv(f"{output_folder}/Mean_dwell/Filtered_meandwell_{treatment_name}.csv", index=False)
        dwell_frequency = util.transition_frequency(filtered_data)
        dwell_frequency["sample"]=treatment_name
        dwell_frequency.to_csv(f"{output_folder}/Dwell_frequency/Filtered_dwellfrequency_{treatment_name}.csv", index=False, header=None)

    FRET_value_after_transition = util.fret_state_trans(TDP_data, Transition_threshold, fps, FRET_thresh, 'after')
    plot_fret_trans(FRET_value_after_transition, order, save_loc=plot_folder, FRET_state='after')
    FRET_value_before_transition = util.fret_state_trans(TDP_data, Transition_threshold, fps, FRET_thresh, 'before')
    plot_fret_trans(FRET_value_before_transition, order, save_loc=plot_folder, FRET_state='before')

    org_chap_events = ps.count_chaperone_events(dfs=TDP_data, thresh=thresh_for_events, fps_clean=fps, thresh_clean=FRET_thresh)
    org_chap_events['FRET_after_normalised'] = org_chap_events['FRET_after']/org_chap_events['Total Molecule Lifetime (min)']
    org_chap_events['FRET_before_normalised'] = org_chap_events['FRET_before']/org_chap_events['Total Molecule Lifetime (min)']
    org_chap_events['bind_and_release'] = org_chap_events[['FRET_after', 'FRET_before']].min(axis=1) ### bind and release event defined here as the minimum number of binding or release events
    org_chap_events['bind_and_release_overtime'] = (org_chap_events['bind_and_release']/org_chap_events['Total Molecule Lifetime (min)'])
    org_chap_events['bind_and_release_overtime'] = org_chap_events['bind_and_release_overtime'].replace(0, np.nan)
    org_chap_events.to_csv(f"{output_folder}/chaperone_events_per_molecule.csv", index=False)


    plot_binding_release(org_chap_events, order, save_loc=plot_folder, chaperone=event, palette=palette_to_use)

    mean_chaperone_bind_release_mean = org_chap_events.groupby('treatment')['bind_and_release_overtime'].mean()
    mean_chaperone_bind_release_sem = org_chap_events.groupby('treatment')['bind_and_release_overtime'].sem()
    mean_chaperone_bind_release_N = org_chap_events.groupby('treatment')['bind_and_release_overtime'].count()
    bind_release_col = pd.concat([mean_chaperone_bind_release_mean,mean_chaperone_bind_release_sem, mean_chaperone_bind_release_N], axis=1)
    bind_release_col.columns = ['mean_bind_and_release', 'sem', 'n']
    bind_release_col.reset_index(inplace=True)
    bind_release_col.to_csv(f"{output_folder}/bind_release_col.csv", index=False)

    TDP_data['FRET_trans_difference'] = abs(TDP_data['FRET_before'] - TDP_data['FRET_after'])

    large_transitions_to_plot = ps.find_large_transitions(TDP_data, Transition_threshold)
    plot_large_transitions(large_transitions_to_plot, order, save_loc=plot_folder, palette=palette_to_use)
    return bind_release_col


def master_dwell_time_func_tristate(
    order,
    output_folder='Experiment_1-description/python_results',
    palette_to_use='BuPu',
    low_thresh=0.3,
    high_thresh=0.5,
    fps=5,
    thresh=0.8,
    Transition_threshold=0.5,
    event='binding_and_release',
    event_thresh=0.5
):
    """
    Tri-state version of master_dwell_time_func.

    States:
        low  : FRET < low_thresh
        mid  : low_thresh <= FRET < high_thresh
        high : FRET >= high_thresh

    Transition types:
        low-low, low-mid, low-high,
        mid-low, mid-mid, mid-high,
        high-low, high-mid, high-high

    Reads TDP_cleaned_tristate_raw.csv (produced by master_tdp_cleanup_func_tristate).
    """

    import os
    import numpy as np
    import pandas as pd

    plot_folder = f'{output_folder}/dwell_analysis_figs_tristate'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    filename = f'{output_folder}/TDP_cleaned_tristate_raw.csv'
    TDP_data = pd.read_csv(filename, header="infer")

    def fret_state(f):
        if f < low_thresh:
            return 'low'
        elif f < high_thresh:
            return 'mid'
        else:
            return 'high'

    all_tri_dwell = []

    dwell_col_guess = None
    for candidate in ['dwell_time', 'res_time', 'residence_time', 'Time_s', 'Time']:
        if candidate in TDP_data.columns:
            dwell_col_guess = candidate
            break

    if dwell_col_guess is None:
        raise KeyError(
            "Could not infer dwell-time column in TDP_data. "
            "Please update master_dwell_time_func_tristate to use the correct column name "
            "(e.g. 'res_time', 'dwell_time', etc.)."
        )

    dwell_col = dwell_col_guess

    for treatment_name, df_t in TDP_data.groupby("treatment_name"):
        initial_data = df_t[df_t["treatment_name"] == treatment_name].copy()
        cleaned_data = util.cleanup_dwell(initial_data, fps, thresh, 'keep')

        cleaned_data['state_before'] = cleaned_data['FRET_before'].apply(fret_state)
        cleaned_data['state_after']  = cleaned_data['FRET_after'].apply(fret_state)
        cleaned_data['transition_type'] = (
            cleaned_data['state_before'] + '-' + cleaned_data['state_after']
        )
        cleaned_data['treatment_name'] = treatment_name
        all_tri_dwell.append(cleaned_data)

    if not all_tri_dwell:
        raise ValueError("No dwell data found after tri-state classification.")

    tri_dwell = pd.concat(all_tri_dwell, ignore_index=True)
    tri_dwell.to_csv(f"{output_folder}/Dwell_times_tristate_all.csv", index=False)

    summary_rows = []
    for (treatment_name, tr_type), df_tr in tri_dwell.groupby(['treatment_name', 'transition_type']):
        dvals = df_tr[dwell_col].values
        dvals = dvals[np.isfinite(dvals) & (dvals > 0)]
        if len(dvals) == 0:
            continue
        summary_rows.append({
            'treatment': treatment_name,
            'transition_type': tr_type,
            'mean_dwell': np.mean(dvals),
            'sem': np.std(dvals, ddof=1) / np.sqrt(len(dvals)) if len(dvals) > 1 else np.nan,
            'n': len(dvals)
        })

    dwell_summary = pd.DataFrame(summary_rows)
    dwell_summary.to_csv(f"{output_folder}/Mean_dwell_tristate.csv", index=False)

    freq_rows = []
    for treatment_name, df_t in tri_dwell.groupby('treatment_name'):
        counts = df_t['transition_type'].value_counts()
        total  = counts.sum()
        for tr_type, count in counts.items():
            freq_rows.append({
                'treatment': treatment_name,
                'transition_type': tr_type,
                'count': count,
                'frequency': count / total if total > 0 else np.nan
            })

    dwell_frequency = pd.DataFrame(freq_rows)
    dwell_frequency.to_csv(f"{output_folder}/Dwell_frequency_tristate.csv", index=False)

    FRET_value_after_transition = util.fret_state_trans(
        TDP_data, Transition_threshold, fps, low_thresh, 'after'
    )
    plot_fret_trans(FRET_value_after_transition, order, save_loc=plot_folder, FRET_state='after')

    FRET_value_before_transition = util.fret_state_trans(
        TDP_data, Transition_threshold, fps, low_thresh, 'before'
    )
    plot_fret_trans(FRET_value_before_transition, order, save_loc=plot_folder, FRET_state='before')

    org_chap_events = ps.count_chaperone_events(
        dfs=TDP_data, thresh=event_thresh, fps_clean=fps, thresh_clean=low_thresh
    )

    org_chap_events['FRET_after_normalised'] = (
        org_chap_events['FRET_after'] / org_chap_events['Total Molecule Lifetime (min)']
    )
    org_chap_events['FRET_before_normalised'] = (
        org_chap_events['FRET_before'] / org_chap_events['Total Molecule Lifetime (min)']
    )
    org_chap_events['bind_and_release'] = org_chap_events[['FRET_after', 'FRET_before']].min(axis=1)
    org_chap_events['bind_and_release_overtime'] = (
        org_chap_events['bind_and_release'] / org_chap_events['Total Molecule Lifetime (min)']
    )
    org_chap_events['bind_and_release_overtime'] = (
        org_chap_events['bind_and_release_overtime'].replace(0, np.nan)
    )

    plot_binding_release(org_chap_events, order, save_loc=plot_folder, chaperone=event, palette=palette_to_use)

    mean_chaperone_bind_release_mean = org_chap_events.groupby('treatment')['bind_and_release_overtime'].mean()
    mean_chaperone_bind_release_sem  = org_chap_events.groupby('treatment')['bind_and_release_overtime'].sem()
    mean_chaperone_bind_release_N    = org_chap_events.groupby('treatment')['bind_and_release_overtime'].count()

    bind_release_col = pd.concat(
        [mean_chaperone_bind_release_mean, mean_chaperone_bind_release_sem, mean_chaperone_bind_release_N],
        axis=1
    )
    bind_release_col.columns = ['mean_bind_and_release', 'sem', 'n']
    bind_release_col.reset_index(inplace=True)
    bind_release_col.to_csv(f"{output_folder}/bind_release_col_tristate.csv", index=False)

    TDP_data['FRET_trans_difference'] = abs(TDP_data['FRET_before'] - TDP_data['FRET_after'])
    large_transitions_to_plot = ps.find_large_transitions(TDP_data, Transition_threshold)
    plot_large_transitions(large_transitions_to_plot, order, save_loc=plot_folder, palette=palette_to_use)

    return dwell_summary, dwell_frequency, bind_release_col
