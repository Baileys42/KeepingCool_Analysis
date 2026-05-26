import pandas as pd
import glob as glob
import os as os

# --------- Read in heatmap data, remove outliers and concatenate multiple treatments into a single data frame for plotting in 1D-heatmap-plotting -------

def remove_outliers(compiled, plot_type='hist', data_type="raw"):
    """removes outliers from dataframe

    Args:
        compiled (dataframe): raw dataframe containing outliers to be removed
        plot_type (str): string can either be 'hist' for histogram data or 'TDP' for TDP data
        data_type (str, optional): removes either raw FRET values or 'idealized' FRET values]. Defaults to "raw".

    Returns:
        (dataframe): returns cleaned data without outliers
    """
    if plot_type == 'hist':
        if data_type == "raw":
            return compiled[(compiled['FRET'] > -0.5) & (compiled['FRET'] < 1.5)].copy()
        if data_type == "idealized":
            return compiled[(compiled[4] > -0.5) & (compiled[4] < 1.5)].copy()
    elif plot_type == 'TDP':
        outliers = compiled[(compiled["FRET before transition"] < -0.5)|(compiled["FRET before transition"] > 1.5)|(compiled["FRET after transition"] < -0.5) | (compiled["FRET after transition"] > 1.5)].index
        compiled.drop(outliers, inplace=True)
        return compiled
    else:
        print('invalid plot type, please set plot_type as "hist" or "TDP" - you idiot')


# ------ Calculate the dwell time of each state, identify/generate transitions, remove outliers, filter for transitions of interest (e.g., low to high), ---
# ------- identify the first transition that meets criteria and export csv files for plotting in 1D-heatmap-plotting --------------------------------------

def calculate_dwell_time(df):
    """Function to convert raw idealized data to a form in which the duration of each idealized state is calculated

    Args:
        df (dataframe): dataframe containing each molecule and the idealized fret 

    Returns:
        dataframe: returns dataframe containing the duration of each FRET state from each molecule
    """
    df_test2 = []
    for Molecule, dfs in df.groupby('molecule_number'):
        frame_length = len(dfs)
        test = dfs.groupby([dfs['idealized FRET'].ne(dfs['idealized FRET'].shift()).cumsum(), 'idealized FRET']).size()
        test = test.reset_index(level=1, drop=False)
        test['Molecule'] = Molecule
        test['number_of_frames'] = frame_length
        df_test2.append(test)
    df_test3 = pd.concat(df_test2)
    df_test3.columns = ['FRET_state', 'Time', 'Molecule', 'number_of_frames']
    df_test3 = df_test3.reset_index().drop('idealized FRET', axis=1)
    return df_test3[df_test3.groupby('Molecule').Molecule.transform('count') > 1]

def generate_transitions(df):
    """Converts the duration of each FRET state into a transition, whereby the FRET state before, the FRET state after
    and the duration of the FRET state before a transition is given in a single line. Each line represents a single transition.

    Args:
        df (dataframe): dataframe generated following 'calculate_dwell_time' function in which the duration of a certain
        FRET state is given for each FRET state for all molecules

    Returns:
        dataframe: returns a dataframe in which each row represents a transition, with FRET before transition, FRET after transition
        and duration of FRET state before transition (given in number of frames in column Time) provided
    """
    df_toconcat = []
    for molecule, dfs in df.groupby('Molecule'):
        thing1 = dfs.assign(FRET_after=dfs.FRET_state.shift(-1)).dropna()
        df_toconcat.append(thing1)
    compiled_df = pd.concat(df_toconcat).reset_index(drop=True)
    compiled_final = compiled_df[['Molecule', 'FRET_state', 'FRET_after', 'Time', 'number_of_frames']]
    compiled_final.columns = ['Molecule', 'FRET_before', 'FRET_after', 'Time', 'number_of_frames']
    return compiled_final

def remove_outliers2(compiled_TDP):
    outliers = compiled_TDP[(compiled_TDP["FRET_before"] < -0.5)|(compiled_TDP["FRET_before"] > 1.5)|(compiled_TDP["FRET_after"] < -0.5) | (compiled_TDP["FRET_after"] > 1.5)].index
    compiled_TDP.drop(outliers, inplace=True)
    return compiled_TDP

def filter_FRET_trans_if(dfs, thresh, trans_type='low_to_high'):
    """This function has several roles. Firstly, for each molecule in the dataframe it will add a column with the cumulative
    sum of all residence times that will be used in later functions. Secondly, depending on what kind of transitions
    you are interested in, it will filter the dataset to include only those transitions (e.g., low to high)

    Args:
        dfs (dataframe): Dataframe containing the molecules, fret before, fret after, idealized fret, fret and time
        thresh (value): The FRET value at which to set the transition threshold. Will only find those that are
        lower than the thresh going to above the thresh (or vice versa)
        trans_type (str, optional): _description_. Defaults to 'low_to_high'. Dictates if youu want to look at high-to-low
        or low-to-high transitions. This is set as variable at the top of the script.

    Returns:
        dataframe: Will return a filtered dataframe with the transitions of interest as well as the cumulative sum of time
        at which each transition occurs (essentially how long into imaging does the transition appear)
    """
    comb = []
    for Molecule, df in dfs.groupby('Molecule'):
        df['cum_sum'] = df['time (s)'].cumsum()
        comb.append(df)
    combined = pd.concat(comb)
    if trans_type  == 'high_to_low':
        filt_data = combined[(combined['FRET_before'] > thresh) & (combined['FRET_after'] < thresh)]
    elif trans_type == 'low_to_high':
        filt_data = combined[(combined['FRET_before'] < thresh) & (combined['FRET_after'] > thresh)]
    return filt_data

def select_first_transition(dfs, time_thresh, injection_time):
    """Will find the first transition for each molecule. Important to note that this function should be run after the 
    'filter_FRET_trans_if' function, which filters for only those transitions that meet a criteria. This function 
    will essentially then find the first transition for a molecule that meets a defined criteria (e.g., first low-to-high 
    transition)

    Args:
        dfs (dataframe): Filtered dataframe containing only transitions of interest. Same as that returned after 
        executing 'filter_FRET_trans_if'

    Returns:
        dataframe: Dataframe containing the first transition of each molecule and the cumulative time that this occurs.
    """
    first_trans = []
    for molecule, df in dfs.groupby('Molecule'):
        if df['cum_sum'].min() < time_thresh:
            continue
        first_trans_above_timethresh = df[df['cum_sum']==df['cum_sum'].min()]
        first_trans_above_timethresh['cum_sum'] = first_trans_above_timethresh['cum_sum']-injection_time
        first_trans.append(first_trans_above_timethresh) 
    return pd.concat(first_trans)


# -------------- Synchronize the first transition of each molecule to time=0 and then export csv files for plotting in 1D-heatmap-plotting ---------------

def normalise_to_event(df1, df2, FRET_thresh, transition_type='low_to_high'):
    """ This function uses two dataframes to normalise the x-axis for each molecule so that the first transition
    that meets a criteria (filtered for and identified using the 'filter_FRET_trans_if' and 'select_first_transition'
    functions) is set to 0. This should allow the first transition between molecules to be synchronised to potentially
    observe changes in FRET that occur immediately prior to or after the transition that is normally hidden by
    the asynchronous timing of transitions between molecules.

    Args:
        df1 (dataframe): Dataframe that has not been filtered. Contains all transitions for all molecules and treatments
        df2 (dataframe): Filtered dataframe. Contains only the first transition for each molecule that meets the criteria.
        Also contains the cumulative sum of time at which that transition occurs, which is then subtracted from all
        timepoints for the corresponding molecule in df1 

    Returns:
        dataframe: returns a dataframe that contains all molecules that contain the transition of interest with an extra 
        column containing the normalised time (time for molecule minus time of first transition)
    """
    collated = []
    for (treatment, mol), df in df2.groupby(['treatment', 'Molecule']):
        norm_df = df1[(df1['treatment_name'] == treatment) & (df1['molecule_number'] == mol)]
        norm_df['normalised_to_event'] = norm_df['time']-(float(df[(df['Molecule'] == mol) & (df['treatment'] == treatment)]['cum_sum']))
        if transition_type == 'low_to_high' and norm_df['idealized FRET'].iloc[0] <= FRET_thresh:
            collated.append(norm_df)
        elif transition_type == 'high_to_low' and norm_df['idealized FRET'].iloc[0] >= FRET_thresh:
            collated.append(norm_df)
    return pd.concat(collated)


