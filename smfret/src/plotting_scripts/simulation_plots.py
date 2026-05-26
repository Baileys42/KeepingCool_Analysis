import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os as os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from smfret.src.processing_scripts import simulation_scripts as aa

def plot_state_simulations(plot_export, data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Time', y='State')
    plt.xlabel('Time (s)')
    plt.ylabel('')
    plt.yticks(range(5), [f'State {i}' for i in range(5)])
    plt.savefig(f'{plot_export}/simulated_states.svg', dpi=300)
    plt.show()

def simple_sim_test(plot_export, num_states=5, rate_up=0.4, rate_down=0.11, simulation_time=360, time_step=1, seed=None):
    data = aa.simulate_state_transitions(num_states=num_states, rate_up=rate_up, rate_down=rate_down, simulation_time=simulation_time, 
                                time_step=time_step, seed=seed)

    plot_state_simulations(plot_export, data)
    data = aa.plot_binary_state(data, plot_export, threshold_state=1, noise_level=.05)
    mean_dwell_time, concatenated_data, fits = aa.analyze_dwell_times(plot_export=plot_export, threshold_state=1, num_simulations=300, num_states=num_states, rate_up=rate_up, rate_down=rate_down,
                    simulation_time=360, time_step=1)

    average_state_df = aa.calculate_average_state_per_simulation(concatenated_data, plot_export)

    # Plot the average state per simulation
    plt.figure(figsize=(10, 6))
    sns.histplot(data=average_state_df, x='Average State', color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Average State')
    plt.ylabel('Frequency')
    plt.title('Average State Per Simulation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plot_export}/test_simulation.svg', dpi=300)
    plt.show()

def plot_average_state_hist(combined_average_state_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_average_state_df, x='Average State', hue='num_states', palette='viridis', edgecolor='black', alpha=0.7, bins=30)
    plt.xlabel('Average State')
    plt.ylabel('Frequency')
    plt.title('Average State Distribution Across Simulations (Grouped by Num States)')
    plt.tight_layout()
    plt.show()

def scatter_plot_rate_constant(fits_data_melted, plot_export, palette='BuPu'):
    """
    Creates a vertical scatter plot with error bars for rate constants, grouped by rate type and colored by the number of states.
    Each data point represents a mean rate constant with its standard error, jittered horizontally to avoid overlap.
    Points are colored according to the 'num_states' column, and a custom legend is provided.
    Parameters
    ----------
    fits_data_melted : pandas.DataFrame
        Melted DataFrame containing at least the columns 'Rate Type', 'Mean Rate Constant', 'Rate Std Error', and 'num_states'.
    plot_export : str
        Directory path where the resulting SVG plot will be saved.
    palette : str, optional
        Name of the seaborn color palette to use for coloring points by 'num_states' (default is 'BuPu').
    Returns
    -------
    None
        The function saves the plot as an SVG file and displays it.
    """
    
    plt.figure(figsize=(3, 5))
    # Map Rate Type to numeric for jittering
    rate_type_map = {rate: i for i, rate in enumerate(fits_data_melted['Rate Type'].unique())}
    jitter_strength = 0.1  # Increase for more spread
    # Assign randomly jittered x positions for each datapoint
    rng = np.random.default_rng(20)  # For reproducibility
    x_vals = []
    for i, row in fits_data_melted.iterrows():
        base_x = rate_type_map[row['Rate Type']]
        jitter = rng.uniform(-jitter_strength, jitter_strength)
        x_vals.append(base_x + jitter)
    # Plot scatter with error bars using jittered x positions
    ax = plt.gca()
    unique_states = sorted(fits_data_melted['num_states'].unique())
    n_states = len(unique_states)
    for idx, (i, row) in enumerate(fits_data_melted.iterrows()):
        color_idx = unique_states.index(row['num_states'])
        # Plot errorbars first (zorder=1)
        ax.errorbar(
            x=x_vals[idx],
            y=row['Mean Rate Constant'],
            yerr=row['Rate Std Error'],
            fmt='none',
            ecolor='black',
            capsize=5,
            linewidth=1.5,
            zorder=1
        )
        # Plot scatter on top (zorder=2)
        ax.scatter(
            x_vals[idx],
            row['Mean Rate Constant'],
            color=sns.color_palette(palette, n_states)[color_idx],
            s=100,
            edgecolor='black',
            label=row['num_states'] if idx == 0 else "",
            zorder=2
        )

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Num States', loc='upper right')
    plt.ylabel(r'Mean rate constant (s$^{-1}$)')
    plt.xticks([rate_type_map[k] for k in rate_type_map.keys()],
               [r'$k_{\mathit{1}}$', r'$k_{\mathit{2}}$'], ha='center')
    plt.gca().tick_params(axis='x', pad=10)
    plt.xlabel('')
    x_min, x_max = min(rate_type_map.values()), max(rate_type_map.values())
    plt.gca().set_xlim(x_min - 0.5, x_max + 0.5)
    y_min, y_max = plt.gca().get_ylim()
    plt.gca().set_ylim(y_min - 0.0, y_max + 0.1)
    plt.tight_layout()
    plt.savefig(f'{plot_export}/mean_double_exponential_rates_scatter_vertical.svg', dpi=300)
    plt.show()

def plot_kinetic_map(results_df, plot_export, palette='Greys'):
    for rate_up in results_df['rate_up'].unique():
        subset = results_df[results_df['rate_up'] == rate_up]
    
        # Plot k1 with num_states as hue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x='rate_down', y='k1', hue='num_states', palette=palette)
        plt.title(f'k1 vs Rate Down (Rate Up = {rate_up})')
        plt.xlabel('Rate Down')
        plt.ylabel('k1 (Rate Constant)')
        plt.legend(title='Num States')
        plt.tight_layout()
        plt.savefig(f'{plot_export}/k1_analysis_rate_up_{rate_up}.svg', dpi=300)
        plt.show()
    
        # Plot k2 with num_states as hue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x='rate_down', y='k2', hue='num_states', palette=palette)
        plt.title(f'k2 vs Rate Down (Rate Up = {rate_up})')
        plt.xlabel('Rate Down')
        plt.ylabel('k2 (Rate Constant)')
        plt.legend(title='Num States')
        plt.tight_layout()
        plt.savefig(f'{plot_export}/k2_analysis_rate_up_{rate_up}.svg', dpi=300)
        plt.show()

        # Plot k2 with num_states as hue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x='rate_down', y='prop_fast', hue='num_states', palette=palette)
        plt.title(f'proportion of k1 (Rate Up = {rate_up})')
        plt.xlabel('Rate Down')
        plt.ylabel('proportion of k1')
        plt.legend(title='Num States')
        plt.tight_layout()
        plt.savefig(f'{plot_export}/k2_analysis_rate_up_{rate_up}.svg', dpi=300)
        plt.show()

def plot_bar_by_num_states_and_rate_up(
    data, x, y, hue, palette, xlabel, ylabel, legend_title, filename, plot_export, figsize=(10, 6), plot_err=False, err_col='std_error'
):
    """
    Plots a barplot of y vs x, grouped by hue, with optional error bars, custom labels, and saves the figure.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x (str): Column name for x-axis.
        y (str or int): Column name or index for y-axis.
        hue (str): Column name for hue grouping.
        palette (dict or str): Color palette for hue.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        legend_title (str): Title for the legend.
        filename (str): Filename for saving the plot (without extension).
        plot_export (str): Directory to save the plot.
        figsize (tuple): Figure size.
        plot_err (bool): Whether to plot error bars using err_col.
        err_col (str): Column name for standard error values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if plot_err and err_col in data.columns:
        # Draw the barplot
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            edgecolor='black',
            ax=ax,
            ci=None
        )
        # Add error bars at the correct positions
        # Get the positions of each bar
        n_hue = data[hue].nunique()
        n_x = data[x].nunique()
        width = 0.8 / n_hue  # seaborn default total bar width is 0.8

        for i, (_, group) in enumerate(data.groupby(x, sort=False)):
            for hue_val, row in group.set_index(hue).iterrows():
                hue_idx = list(data[hue].unique()).index(hue_val)
                # Calculate the center of the bar
                bar_center = i - 0.4 + width/2 + hue_idx*width
                ax.errorbar(
                    bar_center,
                    row[y],
                    yerr=row[err_col],
                    fmt='none',
                    ecolor='black',
                    capsize=5,
                    linewidth=1.5,
                    zorder=10
                )
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(data[x].unique())
    else:
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            edgecolor='black',
            ax=ax
        )
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(title=legend_title, fontsize=14, title_fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{plot_export}/{filename}.svg', dpi=300)
    plt.show()

def plot_time_in_each_state(plot_export, data, palette='BuPu'):
    state_list = [col for col in data.columns if isinstance(col, int)]
    data.groupby(['num_states', 'Simulation', 'rate_up_down'])[state_list].mean().reset_index()
    data['total'] = data[state_list].sum(axis=1)
    for col in state_list:
        data[f'prop_{col}'] = (data[col] / data['total'])*100
    # Melt the DataFrame to long format for easier plotting
    prop_cols = [f'prop_{col}' for col in state_list if f'prop_{col}' in data.columns]
    melted = data.melt(
        id_vars=['num_states', 'Simulation', 'rate_up_down'],
        value_vars=prop_cols,
        var_name='state',
        value_name='prop')

    melted['state'] = melted['state'].str.extract(r'prop_(\d)').astype(int)
    # Now you can group by num_states, state, and Simulation to keep simulation info
    summary = melted.groupby(['num_states', 'state', 'Simulation', 'rate_up_down'])['prop'].mean().reset_index()

    for rate_up, df in summary.groupby('rate_up_down'):
        mean_std = df.groupby(['num_states', 'state'])['prop'].agg(['mean', 'std']).reset_index()
        unique_num_states = sorted(mean_std['num_states'].unique())
        nrows = len(unique_num_states)

        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(6, 1.25 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        sns.set_style('ticks')
        # Get BuPu colors for the number of states
        num_states_per_plot = mean_std['state'].nunique()
        bupu = cm.get_cmap(palette, num_states_per_plot)
        state_colors = [mcolors.to_hex(bupu(i)) for i in range(num_states_per_plot)]

        for idx, num_states in enumerate(unique_num_states):
            ax = axes[idx]
            subset = mean_std[mean_std['num_states'] == num_states]
            # Barplot of mean proportions for this num_states, with BuPu colors per state
            sns.barplot(
            data=subset,
            x='state',
            y='mean',
            ax=ax,
            palette=state_colors,
            capsize=0.1, 
            edgecolor='black')
            # Add STD error bars manually
            x = subset['state'].values
            y = subset['mean'].values
            yerr = subset['std'].values
            ax.errorbar(x, y, yerr=yerr, fmt='none', c='black', capsize=5, lw=1)
            ax.set_ylabel('')
            ax.set_ylim(0, 70)
            if idx == nrows - 1:
                ax.set_xlabel('State')
            else:
                ax.set_xlabel('')

        plt.tight_layout()
        fig.text(0.0, 0.5, 'Time spent in each state (%)', va='center', rotation='vertical', fontsize=12)

        fig.savefig(f'{plot_export}/{rate_up}_state_proportions_by_num_states.svg', dpi=300)
        plt.show()

def master_simplot(results, plot_export, rate_up_palette='BuPu', other_pal='BuPu'):
    (combined_average_state_df, 
    combined_concat, 
    proportion_time_combined, 
    transitions_x_to_y_rate, 
    mean_dwell_times_by_num_states_rate_up, 
    fits_data_melted) = aa.df_processing_for_plots(results, state_x=2, state_y=1)

    plot_average_state_hist(combined_average_state_df)
    # average_state_df = aa.calculate_average_state_per_simulation2(combined_concat, plot_export, palette='BuPu')

    plot_bar_by_num_states_and_rate_up(
    data=proportion_time_combined,
    x='num_states',
    y=0,
    hue='rate_up',
    palette=rate_up_palette,
    xlabel='# possible states',
    ylabel='Proportion of time in state 0 or 1',
    legend_title='Rate Up',
    filename='percent_time_state_0_1_combined_rate_up',
    plot_export=plot_export)

    plot_bar_by_num_states_and_rate_up(
    data=transitions_x_to_y_rate,
    x='num_states',
    y=0,
    hue='rate_up',
    palette=rate_up_palette,
    xlabel='# possible states',
    ylabel=r'$k_{2\to1}$ (s$^{-1}$)',
    legend_title='Rate Up',
    filename='average_transitions_2_to_1_rate_rate_up',
    plot_export=plot_export)


    plot_bar_by_num_states_and_rate_up(
    data=mean_dwell_times_by_num_states_rate_up,
    x='num_states',
    y='mean_dwell_time',
    hue='rate_up',
    palette=rate_up_palette,
    xlabel='# possible states',
    ylabel='Mean dwell time (s)',
    legend_title='Rate Up',
    filename='mean_dwell_time_by_num_states_rate_up',
    plot_export=plot_export, 
    plot_err=True,
    err_col='std_error')


    plot_time_in_each_state(plot_export, combined_average_state_df, palette=other_pal)
    return combined_average_state_df,combined_concat, proportion_time_combined, transitions_x_to_y_rate, mean_dwell_times_by_num_states_rate_up, fits_data_melted


