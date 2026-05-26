import pandas as pd


# -------Identifies the transition point (the point at which the idealized FRET changes), and then assigns that transition a dwell time 
# ------ and the FRET states before and after a transition.

def generate_transitions_sync(dfs):
    """identifies the time at which a transition occurs and provides the FRET state before (FRET_before) and after (FRET_after) a transition occurs.

    Args:
        dfs (df): dataframe containing the cleaned FRET histogram data with the dwell time of each state for each molecule. Generated using the calculate dwells function.

    Returns:
        df: dataframe that contains extra columns, which include the transition point (i.e., the point at which the idealized FRET changes, is either True or False), transition dwell (the duration of FRET_before prior to a True transition) and the FRET_before or FRET_after a transition
    """
    compiled_transition = []
    for molecule, df in dfs.groupby(['unique_id']):
        df['transition_point'] = df['idealized FRET'].ne(df['idealized FRET'].shift())
        df['transition_point'].iloc[0] = False
        df['column_for_dict'] = df['dwell_steady_state']-1
        steady_dwell = df[['dwell_steady_state', 'dwell']]
        dwell_dict = dict(zip(steady_dwell['dwell_steady_state'], steady_dwell['dwell']))
        df['transition_dwell'] = df['column_for_dict'].map(dwell_dict)
        steadyFRET = df[['dwell_steady_state', 'idealized FRET']]
        test_dict = dict(zip(steadyFRET['dwell_steady_state'], steadyFRET['idealized FRET']))
        df['FRET_before'] = df['column_for_dict'].map(test_dict)
        df['FRET_after'] = df['idealized FRET']
        df.drop('column_for_dict', axis=1, inplace=True)
        compiled_transition.append(df)
    return pd.concat(compiled_transition)


# ----- Filters the dataset prior to plotting. Will filter based on the type of transition (low-to-high or vice-versa) and for how long the dwell state 
# ----- exists prior to a transition. The next function will then plot all the transitions that meet the filtered criteria. -----------------------

def filt_df_to_plot(df, FRET_before, FRET_after, transition_type='low_to_high', min_dwell_before=0):
    """will filter the dataframe according to the transition of interest and the dwell time of the FRET state prior to that transition. Returns a list of indexes that meet the transition criteria

    Args:
        df (df): dataframe containing the cleaned FRET data with transition information
        FRET_before (float): FRET state prior to transition, used to filter data
        FRET_after (floar): FRET state after transtion, used to filter data
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


def determine_first_transition_in_sequence(dfs):
    """Function finds the first transition within a sequence of multiple consecutive increases in FRET. For each molecule, the function will identify a run of transitions where 
    FRET_increase is True and then it will create a new column that identifies the first transition within that sequence. This can happen multiple times within a single molecule.

    Args:
        dfs (df): Dataframe containing transition data, with a column that identifies whether a certain transition represents an increase in FRET.

    Returns:
        df: dataframe with column 'output_column' which denotes the presence of the first transition in a sequence with 1.
    """
    combined_data = []
    for molecule, df in dfs.groupby('unique_id'):
        trans_list = df['FRET_increase'].to_list()
        df['output_column'] = [
        1 if all([(x == 0), (trans_list[x] == True), (trans_list[x+1] == True)]) else 
        (1 if all([(trans_list[x-1] == False),(trans_list[x] == True), (trans_list[x+1] == True)]) else 0)
        for x in range(len(trans_list)-1)]+[0]
        combined_data.append(df)
    return pd.concat(combined_data)


# def concat_trans_proportion(dfs, raw_df, FRET_before, FRET_after):
#     """Function used to determine the proportion of FRET_increase events that are consecutive. Also subsets the data into two classes (1) transitions that are the first in a 
#     sequence of consecutive increases in FRET (2) those that are not. 

#     Args:
#         dfs (df): dataframe that has been returned from the 'determine_first_transition_in_sequence' function.
#         raw_df (df): raw dataframe containing all transitions.
#         FRET_before (float): used to look at only those transitions that originate below FRET_before
#         FRET_after (float): used to look at only those transitions that transition to greater than FRET_after

#     Returns:
#         df: returns dataframe containing all transtions that are either consecutive or non-consecutive, and also summarised details on the occurence of these transitions.
#     """
#     #### create dataframes filtered to have only consecutive or non-consecutive transitions
#     consecutive_trans = dfs[dfs['output_column']==True]
#     nonconsecutive_trans = dfs[dfs['output_column']==False]

#     ##### determine the total number of transitions and the number of transitions that meet a criteria
#     transitions_only = raw_df[raw_df['transition_point'] == True]
#     total_trans = transitions_only.groupby('treatment_name')['transition_point'].sum()
#     transitions_above_thresh = transitions_only[(transitions_only['FRET_before'] < FRET_before) & (transitions_only['FRET_after'] > FRET_after)]
#     trans_above_thresh = transitions_above_thresh.groupby('treatment_name')['transition_point'].sum()
#     percent_trans_meet_criteria = (trans_above_thresh/total_trans)*100
#     percent_trans_meet_criteria_df = pd.DataFrame(percent_trans_meet_criteria).reset_index()
#     percent_trans_meet_criteria_df.columns = ['treatment', '% trans DnaK release']

#     #### determine the total number of consecutive transitions that meet a criteria
#     number_consecutive_event = consecutive_trans.groupby('treatment_name')['output_column'].sum()
#     consecutive_event_above_thresh = consecutive_trans[(consecutive_trans['FRET_before'] < FRET_before) & (consecutive_trans['FRET_after'] > FRET_after)]
#     number_consecutive_event_meet_criteria = consecutive_event_above_thresh.groupby('treatment_name')['output_column'].sum()
#     percent_of_consecutive_event_from_DnaK = (number_consecutive_event_meet_criteria/number_consecutive_event)*100
#     percent_of_consecutive_event_from_DnaK = pd.DataFrame(percent_of_consecutive_event_from_DnaK).reset_index()

#     percent_of_DnaK_release_events_that_are_consecutive = pd.DataFrame((number_consecutive_event_meet_criteria/trans_above_thresh)*100).reset_index()
#     percent_of_DnaK_release_events_that_are_consecutive.columns = ['treatment', 'proportion_consecutive_from_DnaK']

#     ### merge columns
#     percent_trans_meet_criteria_df['% DnaK release are consecutive'] = percent_of_DnaK_release_events_that_are_consecutive['proportion_consecutive_from_DnaK']
#     percent_trans_meet_criteria_df['% consecutive events are DnaK release'] = percent_of_consecutive_event_from_DnaK['output_column']
#     return consecutive_trans, nonconsecutive_trans, percent_trans_meet_criteria_df


def ratio_consecutive_to_nonconsecutive(calculated_transitions_df, frames_to_plot, consecutive_trans, nonconsecutive_trans, FRET_before, FRET_after):
    consecutive_from_dnak_release = filt_df_to_plot(consecutive_trans, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    nonconsecutive_from_dnak_release = filt_df_to_plot(nonconsecutive_trans, FRET_before, FRET_after,'low_to_high', frames_to_plot)

    test = calculated_transitions_df.iloc[consecutive_from_dnak_release].groupby('treatment_name')['unique_id'].nunique()/(calculated_transitions_df.iloc[nonconsecutive_from_dnak_release].groupby('treatment_name')['unique_id'].nunique())
    testies = pd.DataFrame(test).reset_index()
    testies.columns = ['treatment', 'prop_consecutive_dnaK_release']
    return testies


def prop_DnaK_release_events_are_consecutive(calculated_transitions_df, frames_to_plot, consecutive_trans, nonconsecutive_trans, FRET_before, FRET_after):
    consecutive_from_dnak_release = filt_df_to_plot(consecutive_trans, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    nonconsecutive_from_dnak_release = filt_df_to_plot(nonconsecutive_trans, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    df = (calculated_transitions_df.iloc[consecutive_from_dnak_release].groupby('treatment_name')['unique_id'].nunique())/((calculated_transitions_df.iloc[nonconsecutive_from_dnak_release].groupby('treatment_name')['unique_id'].nunique())+calculated_transitions_df.iloc[consecutive_from_dnak_release].groupby('treatment_name')['unique_id'].nunique())
    df_final = pd.DataFrame(df).reset_index()
    df_final.columns = ['treatment', 'prop_consecutive_dnaK_release']
    return df_final


def calculate_dwells(dfs):
    """Calculate the dwell duration and number for each molecule and then appends it to the cleaned histogram dataset

    Args:
        dfs (df): dataframe containing the cleaned FRET histogram data. This dataset is produced in the 1A-plot_histogram script

    Returns:
        df: returns the same dataframe with addition columns containing the dwell state and the duration of that state
    """
    compiled = []
    for molecule, df in dfs.groupby(['unique_id']):
        frame_length = len(df[df['unique_id']== molecule])
        df['dwell_steady_state'] = df['idealized FRET'].ne(df['idealized FRET'].shift()).cumsum()
        test_df = pd.DataFrame(df.groupby([df['idealized FRET'].ne(df['idealized FRET'].shift()).cumsum(), 'idealized FRET']).size())
        test_df.index.names = ["transition", "idealized FRET"]
        test_df.reset_index(inplace = True)
        test_df.columns = ["transition", "idealized FRET", 'dwell']
        dict_dwell = dict(zip(test_df['transition'], test_df['dwell']))
        df['dwell'] = df['dwell_steady_state'].map(dict_dwell)
        df['frame_length'] = frame_length
        compiled.append(df)
    compiled_df = pd.concat(compiled)
    # ---------- This next section removes any molecules that have only a single dwell state (i.e., they photobleach ---------------------------
    # ---------------------------------------- before any transition occcurs) ------------------------------------------------------------------
    filtered = [df2 for (molecule, treatment), df2 in compiled_df.groupby(['unique_id', 'treatment_name']) if df2['dwell_steady_state'].nunique() > 1]
    return pd.concat(filtered)



def concat_trans_proportion(dfs, raw_df, FRET_before, FRET_after):
    """Function used to determine the proportion of FRET_increase events that are consecutive. Also subsets the data into two classes (1) transitions that are the first in a 
    sequence of consecutive increases in FRET (2) those that are not. 

    Args:
        dfs (df): dataframe that has been returned from the 'determine_first_transition_in_sequence' function.
        raw_df (df): raw dataframe containing all transitions.
        FRET_before (float): used to look at only those transitions that originate below FRET_before
        FRET_after (float): used to look at only those transitions that transition to greater than FRET_after

    Returns:
        df: returns dataframe containing all transtions that are either consecutive or non-consecutive, and also summarised details on the occurence of these transitions.
    """
    #### create dataframes filtered to have only consecutive or non-consecutive transitions
    consecutive_trans = dfs[dfs['output_column']==True]
    nonconsecutive_trans = dfs[dfs['output_column']==False]

    col_repeat = []
    ##### determine the total number of transitions and the number of transitions that meet a criteria
    transitions_only = raw_df[raw_df['transition_point'] == True]
    for repeat, df in transitions_only.groupby('repeat'):
        total_trans = df.groupby('treatment_name')['transition_point'].sum()
        transitions_above_thresh = df[(df['FRET_before'] < FRET_before) & (df['FRET_after'] > FRET_after)]
        trans_above_thresh = transitions_above_thresh.groupby('treatment_name')['transition_point'].sum()
        percent_trans_meet_criteria = (trans_above_thresh/total_trans)*100
        percent_trans_meet_criteria_df = pd.DataFrame(percent_trans_meet_criteria).reset_index()
        percent_trans_meet_criteria_df.columns = ['treatment', '% trans DnaK release']

    #### determine the total number of consecutive transitions that meet a criteria
        consecutive_trans_filt = consecutive_trans[consecutive_trans['repeat']==repeat]
        number_consecutive_event = consecutive_trans_filt.groupby('treatment_name')['output_column'].sum()
        consecutive_event_above_thresh = consecutive_trans_filt[(consecutive_trans_filt['FRET_before'] < FRET_before) & (df['FRET_after'] > FRET_after)]
        number_consecutive_event_meet_criteria = consecutive_event_above_thresh.groupby('treatment_name')['output_column'].sum()
        percent_of_consecutive_event_from_DnaK = (number_consecutive_event_meet_criteria/number_consecutive_event)*100
        percent_of_consecutive_event_from_DnaK = pd.DataFrame(percent_of_consecutive_event_from_DnaK).reset_index()

        percent_of_DnaK_release_events_that_are_consecutive = pd.DataFrame((number_consecutive_event_meet_criteria/trans_above_thresh)*100).reset_index()
        percent_of_DnaK_release_events_that_are_consecutive.columns = ['treatment', 'proportion_consecutive_from_DnaK']
    ### merge columns
        percent_trans_meet_criteria_df['% DnaK release are consecutive'] = percent_of_DnaK_release_events_that_are_consecutive['proportion_consecutive_from_DnaK']
        percent_trans_meet_criteria_df['% consecutive events are DnaK release'] = percent_of_consecutive_event_from_DnaK['output_column']
        percent_trans_meet_criteria_df['repeat'] = repeat
        col_repeat.append(percent_trans_meet_criteria_df)
    percent_trans_meet_criteria_df = pd.concat(col_repeat)

    return consecutive_trans, nonconsecutive_trans, percent_trans_meet_criteria_df

