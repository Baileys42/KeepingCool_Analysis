import pandas as pd
import numpy as np
import os


def annotate_data(output_folder="Experiment_1-description/python_results/", FRET_thresh=0.5):
    """Adds a column to a dataframe to denote whether the protein is in a certain state (e.g., bound => FRET_thresh, unbound <= FRET_thresh)

    Args:
        output_folder (str, optional): where to import data from. Defaults to "Experiment_1-description/python_results/".
        FRET_thresh (float, optional): used to set a FRET threshold to define bound or unbound. Defaults to 0.5.

    Returns:
        df: returns annotated dataframe
    """
    plot_export = f'{output_folder}/histograms/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)
    compiled_df_HMM = pd.read_csv(f'{output_folder}/compiled_df_HMM.csv')

    # ---------- Identify when molecule is in a certain state (e.g., bound or unbound) based on FRET efficiency and label when it is in that state --------
    FRET_bound = compiled_df_HMM[compiled_df_HMM['FRET Cy3 to AF647']>FRET_thresh]
    compiled_df_HMM['bound'] = np.where(compiled_df_HMM['FRET Cy3 to AF647']>FRET_thresh, 'RNA-bound', 'Unbound')
    return compiled_df_HMM, FRET_bound, plot_export


# ---- Calculates the dwell time for each e_pred_global state based on the HMM fits and then appends the dwell duration to the cleaned FRET histogram data.
# ---- The dwell state (i.e., state 1, state 2, etc) is also appended to the dataset. This code also removes molecules that have only a single dwell state 
# ---- (i.e., does not exhibit a transition)

def calculate_dwells_3color(dfs):
    """Calculate the dwell duration and number for each molecule and then appends it to the cleaned histogram dataset

    Args:
        dfs (df): dataframe containing the cleaned FRET histogram data. This dataset is produced in the 1A-plot_histogram script

    Returns:
        df: returns the same dataframe with addition columns containing the dwell state and the duration of that state
    """
    compiled = []
    for molecule, df in dfs.groupby(['cumulative_molecule']):
        frame_length = len(df[df['cumulative_molecule']==molecule])
        df['dwell_steady_state'] = df['e_pred_global'].ne(df['e_pred_global'].shift()).cumsum()
        test_df = pd.DataFrame(df.groupby([df['e_pred_global'].ne(df['e_pred_global'].shift()).cumsum(), 'e_pred_global']).size())
        test_df.index.names = ["transition", "e_pred_global"]
        test_df.reset_index(inplace = True)
        test_df.columns = ["transition", "e_pred_global", 'dwell']
        dict_dwell = dict(zip(test_df['transition'], test_df['dwell']))
        df['dwell'] = df['dwell_steady_state'].map(dict_dwell)
        df['frame_length'] = frame_length
        compiled.append(df)
    compiled_df = pd.concat(compiled)
    # ---- This next section removes any molecules that have only a single dwell state (i.e., they photobleach before any transition occcurs) ------------
    filtered = [df2 for (molecule, treatment), df2 in compiled_df.groupby(['cumulative_molecule', 'treatment']) if df2['dwell_steady_state'].nunique() > 1]
    return pd.concat(filtered)


# ------- Identifies the transition point (the point at which the e_pred_global changes), and then assigns that transition a dwell time and the --------
# ------- FRET states before and after a transition. ---------------------------------------------------------------------------------------------------

def generate_transitions_3color(dfs):
    """identifies the time at which a transition occurs and provides the FRET state before (FRET_before) and after (FRET_after) a transition occurs.

    Args:
        dfs (df): dataframe containing the cleaned FRET histogram data with the dwell time of each state for each molecule. Generated using the calculate dwells function.

    Returns:
        df: dataframe that contains extra columns, which include the transition point (i.e., the point at which the e_pred_global changes, is either True or False), transition dwell (the duration of FRET_before prior to a True transition) and the FRET_before or FRET_after a transition
    """
    compiled_transition = []
    for molecule, df in dfs.groupby(['cumulative_molecule']):
        dwell_steady_state_list = list(df['dwell_steady_state'].unique())
        df['transition_point'] = df['e_pred_global'].ne(df['e_pred_global'].shift())
        df['transition_point'].iloc[0] = False
        df['column_for_dict'] = df['dwell_steady_state']-1
        steady_dwell = df[['dwell_steady_state', 'dwell']]
        dwell_dict = dict(zip(steady_dwell['dwell_steady_state'], steady_dwell['dwell']))
        df['transition_dwell'] = df['column_for_dict'].map(dwell_dict)
        steadyFRET = df[['dwell_steady_state', 'e_pred_global']]
        test_dict = dict(zip(steadyFRET['dwell_steady_state'], steadyFRET['e_pred_global']))
        df['FRET_before'] = df['column_for_dict'].map(test_dict)
        df['FRET_after'] = df['e_pred_global']
        df.drop('column_for_dict', axis = 1, inplace = True)
        compiled_transition.append(df)
    return pd.concat(compiled_transition)




# -------- Filters the dataset prior to plotting. Will filter based on the type of transition (low-to-high or vice-versa) and for how long the -----
# --------  dwell state exists prior to a transition. The next function will then plot all the transitions that meet the filtered criteria. -------------


def filt_df_to_plot_3color(df, FRET_before, FRET_after, transition_type='low_to_high', min_dwell_before=0):
    """will filter the dataframe according to the transition of interest and the dwell time of the FRET state prior to that transition. Returns a list of indexes that meet the transition criteria

    Args:
        df (df): dataframe containing the cleaned FRET data with transition information
        FRET_before (float): FRET state prior to transition, used to filter data
        FRET_after (float): FRET state after transtion, used to filter data
        transition_type (str, optional): determines what kind of transitions you want to look into (e.g., low-to-high transitions where low is below FRET_before and high is above FRET_after). Defaults to 'low_to_high'.
        min_dwell_before (int, optional): variable that defines for how long a FRET state must have existed before the transition. Defaults to 0.

    Returns:
        list: returns a list of index values where the above transition criteria is true. This list is then used to identify transition points within the cleaned histogram data and plot.
    """
    transitions_to_plot = df[df['transition_point'] == True]
    if transition_type == 'low_to_high':
        index_to_plot = transitions_to_plot[((transitions_to_plot['FRET_before'] < FRET_before) & (transitions_to_plot['transition_dwell'] > min_dwell_before)) & (transitions_to_plot['FRET_after'] > FRET_after)].index
    elif transition_type == 'high_to_low':
        index_to_plot = transitions_to_plot[((transitions_to_plot['FRET_before'] > FRET_before) & (transitions_to_plot['transition_dwell'] > min_dwell_before)) & (transitions_to_plot['FRET_after'] < FRET_after)].index
    return index_to_plot

