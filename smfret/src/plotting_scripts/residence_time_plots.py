import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import smfret.src.Utilities.Data_analysis as uda
import os
from smfret.src.processing_scripts import residence_time_processing as ps 
import scipy

def plot_violin(data, save_loc, palette='BuPu', scale="y_axis"):
    if scale=="y_axis":
        plt.rcParams['svg.fonttype'] = 'none'
        plot1 = plt.figure()
        sns.set(style="ticks", font_scale=1)
        sns.violinplot(
            data=data, 
            x="transition_name", 
            y="y_axis",
            palette=palette, 
            hue="treatment", 
            log_scale=True,
            cut=0)
        plt.ylabel("Residence time (s)")
        plt.xlabel("Transition class")
        plt.legend(title='',loc="upper right")
        plot1.savefig(f"{save_loc}/Violin_plot_normal.svg", dpi=600)
        plt.show()
    if scale=="y_axis_log10":
        plt.rcParams['svg.fonttype'] = 'none'
        plot2 = plt.figure()
        sns.set(style="ticks", font_scale=1)
        sns.violinplot(
            data=data, 
            x="transition_name", 
            y="y_axis_log10",
            palette=palette, 
            hue="treatment", 
            log_scale=True)
        plt.ylabel("Log residence time (s)")
        plt.xlabel("Transition class")
        plt.legend(title='',loc="upper left", bbox_to_anchor=(1,1), ncol =1)
        plot2.savefig(f"{save_loc}/Violin_plot_log.svg", dpi=600)
        plt.show()
    if scale=='split':     
        f, (ax_top, ax_bottom)=plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})
        sns.set(style='ticks')
        sns.violinplot(x="transition_name", y="y_axis", hue="treatment",data=data, ax=ax_top, palette=palette, scale='width')
        sns.violinplot(x="transition_name", y="y_axis", hue="treatment",data=data, ax=ax_bottom, cut=0, palette=palette, scale='width')
        ax_top.set_ylim(bottom=40)   # those limits are fake
        ax_bottom.set_ylim(0,40)
        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
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
    # -------- prepare order of datasets so that sem are correctly mapped onto the correct dataset ------
    list_to_order = list(np.arange(0, len(order), 1))
    dict_to_order = dict(zip(order, list_to_order))
    summary_df['plot_order'] = summary_df['treatment'].map(dict_to_order)
    collated_sorted = summary_df.sort_values(['plot_order', 'transition_type'])
    sorted_df = df.sort_values(['treatment', 'transition_type'])
    # ----------- now plot the figure -------------------
    fig, ax = plt.subplots()
    sns.set(style='ticks', font_scale=1)
    sns.barplot(x='transition_name', y=y_axis, data=sorted_df, hue= 'treatment', palette=palette, ci =None, hue_order=order, edgecolor='black')
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=collated_sorted["sem"], fmt="none",  elinewidth=2, capsize=4, color='black')
    plt.ylabel('Residence time (s)')
    plt.xlabel('')
    plt.legend(title='')
    fig.savefig(f'{save_loc}/mean_residence_withSEM_{y_axis}.svg', dpi=600)
    plt.show()


def plot_residence_time_of_class(df, binwidth, transition, save_log, plot_type='KDE', log=False):
    num_subplots = df['treatment'].nunique()
    if plot_type=='KDE':
        for transition, dfs in df.groupby('transition_type'):
            filt_trans = dfs[dfs['transition_type']== transition]
            sns.kdeplot(data=filt_trans, x='y_axis', hue='treatment', fill=False,  log_scale=True, common_norm=False, palette='mako')
            plt.title(transition)
            plt.show()
    if plot_type=='individual':
        for treatment, dfs in df.groupby('treatment'):
            fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 18), sharex=True)
            axes = axes.flatten()
            for i, transition in enumerate(list(dfs['transition_type'].unique())):
                fig = sns.histplot(data=dfs[dfs['transition_type']==transition], x='y_axis', binwidth=binwidth, kde=True, stat='density', log_scale=log, ax=axes[i])
                axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
                axes[i].set_title(f'{treatment} and {transition}')
                plt.xlim(0, 50)
            plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
            plt.show()
    if plot_type=='collated':
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
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
    if plot_type=='cum_dwell':
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
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


def plot_all_treatments_positive_CDF(
    fits_df,
    filt_df,
    xlim=300,
    binwidth=5,
    figsize=(7, 5)
):
    """
    Plots the positive empirical CDF + positive single/double exponential fits
    for all treatments overlaid in one figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for transition in sorted(filt_df['transition_type'].unique()):
        df_trans = filt_df[filt_df['transition_type'] == transition]
        df_fit   = fits_df[fits_df['transition_type'] == transition]

        for treatment in sorted(df_trans['treatment'].unique()):
            subset = df_trans[df_trans['treatment'] == treatment]["CumulativeTime(s)"]
            bins = np.arange(0, xlim + binwidth, binwidth)
            hist_counts, hist_edges = np.histogram(subset, bins=bins)
            cdf = np.cumsum(hist_counts) / np.sum(hist_counts)
            x_vals = hist_edges[:-1]

            ax.plot(x_vals, cdf,
                    label=f"{treatment} ({transition}) data",
                    linewidth=1.2)

            df_single = df_fit[df_fit['treatment'] == treatment]
            if len(df_single) > 0:
                x_fit = df_single['x_bins'].values
                fit_single_pos = df_single['fit_pos_single'].values
                ax.plot(x_fit, fit_single_pos,
                        linestyle='--', color='orange', linewidth=1.0,
                        label=f"{treatment} ({transition}) 1-exp")

                fit_double_pos = df_single['fit_pos_double'].values
                ax.plot(x_fit, fit_double_pos,
                        linestyle='--', color='blue', linewidth=1.0,
                        label=f"{treatment} ({transition}) 2-exp")

    ax.set_xlim(0, xlim)
    ax.set_xlabel("Residence Time (s)")
    ax.set_ylabel("Positive CDF")
    ax.set_title("Positive CDF + Fitted Exponentials (All Treatments)")
    ax.legend(fontsize='small')
    plt.tight_layout()
    plt.show()




def cumulative_hist_residence_plot(filt_df, palette, save_loc, binwidth=5, xlim=300, plot_biexponentials=True, biexponential_params=None, single_exp=None, figsize=(6, 4)):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=figsize)
    treatment_list = sorted(list(filt_df['transition_type'].unique()))
    for i, transition in enumerate(treatment_list):
        if plot_biexponentials and biexponential_params is not None:
            x_vals = np.linspace(0, xlim, 500)
            for treatment in filt_df['treatment'].unique():
                params = biexponential_params[(treatment, transition)]
                if params is not None:
                    k1, k2, frac = params
                    biexponential_cdf = frac * (1 - np.exp(-k1 * x_vals)) + (1 - frac) * (1 - np.exp(-k2 * x_vals))
                    sns.lineplot(x=x_vals, y=biexponential_cdf, label=f'{treatment}_biexp', linestyle='--', color='black', linewidth=0.5, ax=axes[i])
        elif single_exp is not None:
            for treatment in single_exp['treatment'].unique():
                single_exp_data = single_exp[(single_exp['transition_type'] == transition) & (single_exp['treatment'] == treatment)]
                sns.lineplot(data=single_exp_data, x='x_bins', y='fit', label=f'{treatment}_single_exp', linestyle='--', color='black', linewidth=0.5, ax=axes[i])
        sns.histplot(data=filt_df[filt_df['transition_type'] == transition], 
                x='CumulativeTime(s)', 
                stat='density', 
                cumulative=True, 
                hue='treatment', 
                fill=False, 
                common_norm=False, 
                element='step', 
                binwidth=binwidth, 
                palette=palette,
                ax=axes[i], 
                legend=True,
                zorder=1, 
                linewidth=1.5)
        handles, labels = axes[i].get_legend_handles_labels()
        # axes[i].legend(title=None)
        # axes[i].legend(title=None, handles=handles[len(filt_df['treatment'].unique()):], labels=labels[len(filt_df['treatment'].unique()):], loc='lower right', fontsize='small')
        legend = axes[i].get_legend()
        handles = legend.legendHandles
        labels = [label.get_text() for label in legend.get_texts()]
        axes[i].legend(title=None, handles=handles, labels=labels, loc='lower right', fontsize='small')
        axes[i].set_xlim(-5, xlim)

    plt.xlabel('Residence time (s)')
    plt.tight_layout()
    fig.savefig(f'{save_loc}/cumulative_histogram_residencetime_with_biexp.svg', dpi=600)
    plt.show()



def biexponential_fits(cumulative_dwell_filt, n_bootstrap=1000):
    # Generate the biexponential_params dictionary
    biexponential_params = {}
    r_squared_values = []
    bootstrap_results = []

    for (treatment, transition), group_df in cumulative_dwell_filt.groupby(['treatment', 'transition_type']):
        # Extract the 'CumulativeTime(s)' column for the current treatment
        dwell_times = group_df['CumulativeTime(s)'].dropna().values  # Ensure no NaN values

        # CDF model
        def CDF_mixture(x, k1, k2, frac):
            return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))

        # MLE cost function
        def MLE_cdf(params, x):
            k1, k2, frac = params
            if not (0 < frac < 1):
                return np.inf
            if k1 <= 0 or k2 <= 0:
                return np.inf
            pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
            pdf_vals = np.clip(pdf_vals, 1e-12, None)
            return -np.sum(np.log(pdf_vals))

        # Initial guess and bounds
        initial_guess = [1/300, 1/20, 0.5]  # Ensure k1 < k2 in the initial guess
        bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]

        # Optimization
        result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
        k1, k2, frac = result.x

        # Ensure k1 is the smaller rate (slowest rate)
        if k1 > k2:
            k1, k2 = k2, k1
            frac = 1 - frac

        # Store the parameters in the dictionary
        biexponential_params[(treatment, transition)] = (k1, k2, frac)

        # Calculate R-squared
        x_vals = np.sort(dwell_times)
        observed_cdf = np.arange(1, len(x_vals) + 1) / len(x_vals)
        predicted_cdf = CDF_mixture(x_vals, k1, k2, frac)
        ss_res = np.sum((observed_cdf - predicted_cdf) ** 2)
        ss_tot = np.sum((observed_cdf - np.mean(observed_cdf)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values.append({'treatment': treatment, 'transition_type': transition, 'r_squared': r_squared})

        # Bootstrapping to estimate errors
        bootstrap_k1 = []
        bootstrap_k2 = []
        bootstrap_frac = []
        for _ in range(n_bootstrap):
            resampled_dwell_times = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
            bootstrap_result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(resampled_dwell_times,), bounds=bounds)
            bk1, bk2, bfrac = bootstrap_result.x
            if bk1 > bk2:
                bk1, bk2 = bk2, bk1
                bfrac = 1 - bfrac
            bootstrap_k1.append(bk1)
            bootstrap_k2.append(bk2)
            bootstrap_frac.append(bfrac)

        # Calculate bootstrap errors
        k1_error = np.std(bootstrap_k1)
        k2_error = np.std(bootstrap_k2)
        frac_error = np.std(bootstrap_frac)
        bootstrap_results.append({'treatment': treatment, 'transition_type': transition, 'k1_error': k1_error, 'k2_error': k2_error, 'frac_error': frac_error})

    rate_df = pd.DataFrame([
        {'treatment': treatment, 'transition_type': transition, 'k1': params[0], 'k2': params[1], 'proportion': params[2]}
        for (treatment, transition), params in biexponential_params.items()
    ])

    r_squared_df = pd.DataFrame(r_squared_values)
    bootstrap_df = pd.DataFrame(bootstrap_results)
    rate_df = rate_df.merge(r_squared_df, on=['treatment', 'transition_type'])
    rate_df = rate_df.merge(bootstrap_df, on=['treatment', 'transition_type'])

    return biexponential_params, rate_df

def master_residence_time_func(output_folder, data_paths_violin, order, palette='BuPu', FRET_thresh=0.3, binwidth=10, cumulative_hist_binwidth=2, fit_xlim=300, plot_xlim=300, func=ps.one_phase_association, biexponential=True, figsize=(6, 4)):
    plot_export = f'{output_folder}/Residence_time_plots/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)
    cumulative_dwell = pd.read_csv(f'{output_folder}/cumulative_dwell.csv')
    cumulative_dwell['repeat'] = cumulative_dwell['Molecule'].str.split('_').str[-1]
    test=[]
    for data_name, data_path in data_paths_violin.items():
        data = uda.file_reader(data_path, 'other')
        compiled_data = ps.compiled(data, data_name, FRET_thresh)
        test.append(compiled_data)
    final = pd.concat(test)
    final["y_axis_log10"]=np.log10(final['y_axis']) ## if need to plot in log scale
    dict_for_label={
    f"< {FRET_thresh} to < {FRET_thresh}":'$T_{low-low}$',
    f"< {FRET_thresh} to > {FRET_thresh}":'$T_{low-high}$',
    f"> {FRET_thresh} to > {FRET_thresh}":'$T_{high-high}$',
    f"> {FRET_thresh} to < {FRET_thresh}":'$T_{high-low}$'
    }
    final['transition_name'] = final['transition_type'].map(dict_for_label)
    plot_violin(final, plot_export, palette, 'split')
    # ---------------------- Generate and collate summary statistics of residence times ---------------------
    mean = final.groupby(['treatment', 'transition_type']).mean()
    sem =  final.groupby(['treatment', 'transition_type']).sem()
    final_drop = final.drop('transition_name', axis=1)
    N = final_drop.groupby(['treatment', 'transition_type']).count()
    collated = pd.concat([mean,sem, N], axis=1)
    collated.drop([col for col in collated.columns.tolist() if 'y_axis_log10' in col], axis=1, inplace=True)
    collated.columns = ['mean_residence_time', 'sem', 'n']
    collated.reset_index(inplace=True)
    collated.to_csv(f"{output_folder}/summary.csv", index=False)
    collated_filt = collated[(collated['transition_type']==f'< {FRET_thresh} to > {FRET_thresh}')|(collated['transition_type']==f'> {FRET_thresh} to < {FRET_thresh}')]
    final_filt = final[(final['transition_type']==f'< {FRET_thresh} to > {FRET_thresh}')|(final['transition_type']==f'> {FRET_thresh} to < {FRET_thresh}')]
    # -------------------- plot residence time as a conventional histogram ------------------------
    plot_bar_with_sem(final, collated, order,plot_export, 'y_axis',palette)
    plot_bar_with_sem(final_filt, collated_filt, order,plot_export, 'y_axis', palette)
    plot_residence_time_of_class(final, binwidth, f'< {FRET_thresh} to > {FRET_thresh}', plot_export, 'collated', False)
    plot_residence_time_of_class(final, 0.2, f'< {FRET_thresh} to > {FRET_thresh}', plot_export, 'logged', False)
    print('this is the one right here officer')
    # -------------------- Prepare cumulative dwell time data for plotting (i.e., convert to two-state system) ---------------------
    dict_for_label_cum_dwell={
    'low-low':'$T_{low-low}$',
    'low-high':'$T_{low-high}$',
    'high-high':'$T_{high-high}$',
    'high-low':'$T_{high-low}$'
    }
    cumulative_dwell = cumulative_dwell[cumulative_dwell['treatment_name'].isin(order)]
    cumulative_dwell['transition_name'] = cumulative_dwell['transition_type'].map(dict_for_label_cum_dwell)
    mean = cumulative_dwell.groupby(['treatment_name', 'transition_type'])['CumulativeTime(s)'].mean()
    sem =  cumulative_dwell.groupby(['treatment_name', 'transition_type'])['CumulativeTime(s)'].sem()
    cumulative_dwell_drop = cumulative_dwell.drop('transition_name', axis=1)
    N = cumulative_dwell_drop.groupby(['treatment_name', 'transition_type'])['CumulativeTime(s)'].count()
    col_cum_dwell = pd.concat([mean,sem, N], axis=1)
    col_cum_dwell.drop([col for col in col_cum_dwell.columns.tolist() if 'y_axis_log10' in col], axis=1, inplace=True)
    col_cum_dwell.columns = ['mean_residence_time', 'sem', 'n']
    col_cum_dwell.reset_index(inplace=True)
    cumulative_dwell.columns =['molecule', 'FRET_before', 'FRET_after', 'Time', 'number_of_frames','treatment', 'repeat', 'transition_type', 'shift', 'is_in_sequence','CumulativeTime', 'CumulativeTime(s)','transition_name']
    col_cum_dwell.columns = ['treatment', 'transition_type', 'mean_residence_time', 'sem', 'n']
    col_cum_dwell.to_csv(f"{output_folder}/summary_cum_dwell.csv", index=False)
    col_cum_dwell_filt = col_cum_dwell[(col_cum_dwell['transition_type']=='low-high')|(col_cum_dwell['transition_type']=='high-low')]
    cumulative_dwell_filt = cumulative_dwell[(cumulative_dwell['transition_type']=='low-high')|(cumulative_dwell['transition_type']=='high-low')]
    cumulative_dwell_filt.to_csv(f"{output_folder}/cumulative_dwell_filt.csv", index=False)
    # -------------------- plot cumulative residence time as a conventional histogram ------------------------
    plot_bar_with_sem(cumulative_dwell_filt, col_cum_dwell_filt, order,plot_export,'CumulativeTime(s)', palette)
    plot_residence_time_of_class(cumulative_dwell_filt, binwidth, 'low-high', plot_export, 'cum_dwell', False)
    fits_df, halftime_summary, bootstrap_df = ps.cumulative_residence_fitting(cumulative_dwell_filt,
                                                                    plot_export,
                                                                    bin_width=cumulative_hist_binwidth,
                                                                    xlim = fit_xlim,
                                                                    func=func)
    
    biexponential_params, bioexp_rate_df = biexponential_fits(cumulative_dwell_filt)
    if biexponential == True:
        cumulative_hist_residence_plot(cumulative_dwell_filt, 
                                    palette=palette, 
                                    save_loc=plot_export, 
                                    binwidth=1, 
                                    xlim=plot_xlim, 
                                    plot_biexponentials=True, 
                                    biexponential_params=biexponential_params, 
                                    figsize=figsize)
    else:
        single_exp_fits_df, halftime_summary, _ = ps.cumulative_residence_fitting(cumulative_dwell_filt,
                                                                plot_export,
                                                                bin_width=cumulative_hist_binwidth,
                                                                xlim = fit_xlim,
                                                                func=ps.one_phase_association)
        cumulative_hist_residence_plot(cumulative_dwell_filt, 
                            palette=palette, 
                            save_loc=plot_export, 
                            binwidth=1, 
                            xlim=plot_xlim, 
                            plot_biexponentials=False, 
                            biexponential_params=biexponential_params, 
                            single_exp=single_exp_fits_df, 
                            figsize=figsize)

    return final, collated, cumulative_dwell_filt, fits_df, halftime_summary