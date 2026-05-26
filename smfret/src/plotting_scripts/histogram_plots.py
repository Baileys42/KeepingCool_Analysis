import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from smfret.src.Utilities import Data_analysis as util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from smfret.src.processing_scripts import timelapse_hist_processing as ps


def ridgeline_plot(df, save_loc, dict_key, font, palette='BuPu'):
    # Sets order for ridgeline plot
    ridgeline_keys = list(range(1,len(dict_key)+1))
    str_ridgeline_keys = list(map(str, ridgeline_keys))
    data_paths_ridgeline =  dict(zip(str_ridgeline_keys,dict_key))

    n_colors = len(data_paths_ridgeline)
    pal = sns.color_palette(palette=palette, n_colors=n_colors)
    matplotlib.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'
    g = sns.FacetGrid(df, row='treatment_name', hue='treatment_name', aspect=10, height=5, palette=pal)
    # then we add the densities kdeplots for each condition
    g.map(sns.kdeplot, 'FRET',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=1, linewidth=10)
    # here we add a black line that represents the contour of each kdeplot
    g.map(sns.kdeplot, 'FRET', 
      bw_adjust=1, clip_on=False, 
      color="black", linewidth=10)
    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0,
      lw=2, clip_on=False)
    # we loop over the FacetGrid figure axes (g.axes.flat) and add the condition as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for i, ax in enumerate(g.axes.flat):
        ax.text(-0.5, .5, data_paths_ridgeline[f'{i+1}'],
            fontweight='bold', fontsize=100,
            color=ax.lines[-1].get_color())
        ax.set_facecolor((0, 0, 0, 0))  ### removes background so KDE plots are not obstructed when stacked
    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.7)
    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=([]))
    g.despine(bottom=True, left=True)
    plt.setp(ax.get_xticklabels(), fontsize=100, fontweight='bold')
    plt.xlabel('FRET', fontweight='bold', fontsize=100)
    plt.xlim(-.4, 1.2)
    g.savefig(f'{save_loc}/Histogram-ridgeline.svg', dpi=600)
    plt.show()

# ---------------------- Code to plot regular histogram -------------------------------

def plot_hist_type(df, order, labels, save_loc, kind='bar'):
    plot_hist, ax = plt.subplots()
    sns.set_style("ticks",{'font_scale':1} )
    if kind == 'kde':
        sns.kdeplot(
            data=df, 
            palette='BuPu', 
            x="FRET",
            hue="treatment_name",
            hue_order=order,
            common_norm=False, 
            fill=True, 
            linewidth=1.5, 
            alpha=0.25)
    if kind == 'bar':
        sns.histplot(
            data=df,
            palette='BuPu',
            x="FRET",
            hue="treatment_name",
            hue_order=order,
            common_norm=False, 
            stat='density',
            binwidth=0.05,
            fill=True, 
            kde=True,
            linewidth=1.5, 
            alpha=0.25, 
            element='step')
    plt.xlim(0, 1, 10)
    plt.xlabel("FRET")
    plt.legend([labels[treatment] for treatment in reversed(order)], loc='best', fontsize=12, title='')    
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plot_hist.savefig(f'{save_loc}/Histogram_{kind}.svg', dpi=600)
    plt.show()

# --------------------- code to calculate and plot the proportion of time each molecules spends below a defined threshold ------------------------
# 
def heatmap_timelapse(output_folder, compiled_df, palette, discriminator='KJEG', subplot_id='native'):
    for protein, df in compiled_df.groupby('protein'):
        # Create a gridspec with 2 rows and 2 columns, where the right column is much thinner

        fig = plt.figure(figsize=(8, 4))  # wider to accommodate extra subplot
        gs = GridSpec(2, 2, width_ratios=[8, 1], height_ratios=[1, 1], wspace=0.05)

        # Main subplots (left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)

        # Thin subplots (right)
        ax1_side = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax2_side = fig.add_subplot(gs[1, 1], sharey=ax2)

        # Main plots (as before, but with 'native' instead of 'KJEG')
        sns.kdeplot(data=df[~df['treatment_name'].str.contains(f'{discriminator}|{subplot_id}', case=False, na=False)], x='time_rand', y='FRET', cmap='Greys', shade=bool, cbar=False, cbar_kws={'format': '%.0f%%', 'ticks': [0, 50]}, gridsize=50, levels=6, thresh=0.05, ax=ax1)
        sns.kdeplot(data=df[~df['treatment_name'].str.contains(f'{discriminator}|{subplot_id}', case=False, na=False)],x='time_rand', y='FRET', color='grey', gridsize=50, levels=6, thresh=0.05, ax=ax1)

        sns.kdeplot(data=df[df['treatment_name'].str.contains(f'{discriminator}', case=False, na=False)],x='time_rand', y='FRET', cmap=palette[protein], shade=bool, cbar=False,cbar_kws={'format': '%.0f%%', 'ticks': [0, 25]}, gridsize=50, levels=6, thresh=0.05, ax=ax2)
        sns.kdeplot(data=df[df['treatment_name'].str.contains(f'{discriminator}', case=False, na=False)],x='time_rand', y='FRET', color='black', gridsize=50, levels=6, thresh=0.05, ax=ax2)

        # Side plots (thinner, e.g. marginal distributions)
        sns.kdeplot(data=df[df['treatment_name'].str.contains(f'{subplot_id}', case=False, na=False)], x='time_rand', y='FRET', cmap='Greys', shade=bool, cbar=False, cbar_kws={'format': '%.0f%%', 'ticks': [0, 50]}, gridsize=50, levels=5, thresh=0.05, ax=ax1_side)
        sns.kdeplot(data=df[df['treatment_name'].str.contains(f'{subplot_id}', case=False, na=False)],x='time_rand', y='FRET', color='grey', gridsize=50, levels=5, thresh=0.05, ax=ax1_side)

        sns.kdeplot(data=df[df['treatment_name'].str.contains(f'{subplot_id}', case=False, na=False)],x='time_rand', y='FRET', cmap=palette[protein], shade=bool, cbar=False,cbar_kws={'format': '%.0f%%', 'ticks': [0, 25]}, gridsize=50, levels=5, thresh=0.05, ax=ax2_side)
        sns.kdeplot(data=df[df['treatment_name'].str.contains(f'{subplot_id}', case=False, na=False)],x='time_rand', y='FRET', color='black', gridsize=50, levels=5, thresh=0.05, ax=ax2_side)
        # Remove x/y labels from side plots for clarity
        # Remove x/y labels from side plots for clarity
        ax1_side.set_xlabel('')
        ax1_side.set_ylabel('')
        ax2_side.set_xlabel('')
        ax2_side.set_ylabel('')

        # Ensure main plots show y-axis label and ticks with numbers
        ax1.set_ylabel('FRET')
        ax2.set_ylabel('FRET')
        ax1.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks_position('left')
        ax1.yaxis.set_label_position('left')
        ax2.yaxis.set_label_position('left')
        ax1.set_yticks(np.linspace(-0.2, 1.2, 8))
        ax2.set_yticks(np.linspace(-0.2, 1.2, 8))
        ax1.set_yticklabels([f"{x:.1f}" for x in np.linspace(-0.2, 1.2, 8)])
        ax2.set_yticklabels([f"{x:.1f}" for x in np.linspace(-0.2, 1.2, 8)])

        # Hide x-tick labels and y-tick labels for side plots (but keep ticks if present)
        ax1_side.set_xticklabels([])
        ax1_side.set_yticklabels([])
        ax2_side.set_xticklabels([])
        ax2_side.set_yticklabels([])

        # Show x-ticks only on ax2 (bottom plot)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel('')

        ax2.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.ylim(-0.2, 1.2)
        plt.xlabel('Native')
        plt.savefig(f'{output_folder}/histogram_heatmap_{protein}.svg', dpi=600)
        plt.show()

def timelapse_mapping(directory, timepoint):
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
    

def plot_time_below_thresh(df, order, thresh, save_loc, swarmplot=False, palette='BuPu'):
    fig, ax = plt.subplots()
    sns.violinplot(data=df,
                y='FRET_time_below_thresh',
                x='treatment_name',
                palette=palette,
                scale='width',
                order=order)
    if swarmplot:
        sns.swarmplot(data=df,
                    y='FRET_time_below_thresh',
                    x='treatment_name',
                    color='grey',
                    order=order,
                    alpha=0.25)
    plt.xlabel('')
    plt.ylabel(f'Proportion of time spent < {thresh} FRET')
    fig.savefig(f'{save_loc}/Time_below_thresh_per_molecule.svg', dpi=600)
    plt.xticks(rotation=45)
    plt.show()

    
# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_histogram_func(data_paths, output_folder="Experiment_X-description/python_results", thresh=0.2, swarmplot=False,bruh=False):
    if isinstance(data_paths, dict):    # --------- the first item in the tuple will be the name that goes into the graph legend -------------
        dict_key = list(data_paths.keys())
        # -------- Data from all data sets in the dict will be imported and concatenated into a single dataframe. Outliers wil be removed -----------
        compiled_df = []
        for data_name, (label, data_path) in data_paths.items():
            imported_data = util.file_reader(data_path, 'hist')
            cleaned_raw = util.remove_outliers(imported_data, 'hist') # add "idealized" after imported_data to get idealized histograms
            cleaned_raw["treatment_name"] = data_name
            compiled_df.append(cleaned_raw)
        compiled_df = pd.concat(compiled_df)   #### .rename(columns = {1:"test", 3:"test2"}) ## can rename individually if needed
        compiled_df.columns = ["frames", "donor", "acceptor", "FRET", "idealized FRET", 'molecule_number', "treatment_name"]
        compiled_df.to_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv', index=False)
        labels = {data_name:label for data_name, (label, data_path) in data_paths.items()}
        order = list(reversed(dict_key))
    elif bruh==True:
        compiled_df = pd.read_csv(f'{output_folder}/sep_repeat.csv')
        labels = {val: val for val in compiled_df['treatment_name'].unique()}
        order = list(reversed(labels.keys()))

    else:
        compiled_df = pd.read_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv')
        labels = {val: val for val in compiled_df['treatment_name'].unique()}
        order = list(reversed(labels.keys()))

    # ------------------------- To plot ridgeline histograms -------------------------

    font = {'weight' : 'normal', 'size'   : 12 }
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'
    ridgeline_plot(compiled_df, output_folder, labels, font, 'BuPu')


    plot_hist_type(compiled_df, order, labels, output_folder,kind='bar')
    filt_df = []
    order_normal = list(reversed(order))
    for (treatment, molecule), df in compiled_df.groupby(['treatment_name', 'unique_id']):
        mol_total = len(df)
        mol_below_thresh = len(df[df['FRET']<thresh])
        percent_below_thresh = (mol_below_thresh/mol_total)*100
        df['FRET_time_below_thresh'] = percent_below_thresh
        df = df.iloc[[0]]
        filt_df.append(df)
    filt_dfs = pd.concat(filt_df)
    filt_dfs.to_csv(f'{output_folder}/filt_dfs.csv')
    plot_time_below_thresh(filt_dfs, order_normal, thresh, save_loc=output_folder, palette='BuPu', swarmplot=swarmplot)
    return compiled_df, filt_dfs


def plot_fret_vs_distance(output_folder, R0=51, distance_range=(0, 120), num_points=500, fret_values=(0.05, 0.1), distance_to_mark=None):
    """
    Plot FRET efficiency vs. distance curve and mark specific FRET efficiency values or a specific distance.

    Parameters:
    - output_folder (str): Path to save the plot.
    - R0 (float): Förster radius in Ångströms.
    - distance_range (tuple): Range of distances (start, end) in Ångströms.
    - num_points (int): Number of points to calculate in the distance range.
    - fret_values (tuple): Specific FRET efficiency values to mark on the plot.
    - distance_to_mark (float, optional): Specific distance in Ångströms to mark on the plot.

    Returns:
    - dict: A dictionary containing distances corresponding to the provided FRET efficiency values.
    - float (optional): FRET efficiency at the specified distance_to_mark (if provided).
    """
    # Define distance range
    r = np.linspace(distance_range[0], distance_range[1], num_points)

    # Calculate FRET efficiency
    E = 1 / (1 + (r / R0) ** 6)

    # Function to calculate distance from FRET efficiency
    def calculate_r(E, R0):
        if not (0 < E < 1):  # Ensure E is within valid range
            raise ValueError("E must be between 0 and 1 (exclusive).")
        return R0 * ((1 - E) / E) ** (1 / 6)

    # Calculate distances for the specified FRET efficiency values
    r_values = {E: calculate_r(E, R0) for E in fret_values}

    # Plot the FRET efficiency vs. distance curve
    plt.figure(figsize=(8, 5))
    plt.plot(r, E, label=f'Förster Radius = {R0} Å', color='b')

    # Mark specific FRET efficiency values and corresponding distances
    for E_value, r_value in r_values.items():
        plt.axhline(E_value, linestyle='dashed', color='red', label=f'E = {E_value}')
        plt.axvline(r_value, linestyle='dashed', color='red', label=f'Distance = {r_value:.2f} Å')

    # If a specific distance is provided, calculate and mark the corresponding FRET efficiency
    fret_at_distance = None
    if distance_to_mark is not None:
        fret_at_distance = 1 / (1 + (distance_to_mark / R0) ** 6)
        plt.axvline(distance_to_mark, linestyle='dotted', color='black', label=f'Distance = {distance_to_mark} Å')
        plt.axhline(fret_at_distance, linestyle='dotted', color='black', label=f'E = {fret_at_distance:.2f}')
        print(f"FRET efficiency at distance {distance_to_mark} Å: {fret_at_distance:.2f}")

    plt.xlabel("Distance (Å)")
    plt.ylabel("FRET Efficiency")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/FRET_vs_distance.svg', dpi=600)
    plt.show()

    # Return the distances and optionally the FRET efficiency at the specified distance
    if distance_to_mark is not None:
        return r_values, fret_at_distance
    return r_values

