import pandas as pd


# ------- Remove outliers from TDP data, calculate the dwell/residence times of each FRET state from HMM analysis and identify transitions. ------------
# ------- Concatenatea all datasets into a single dataframe. -------------------------------------------------------------------------------------------

def calculate_dwell_time(df):
    """Function to convert raw idealized data to a form in which the duration of each idealized state is calculated

    Args:
        df (dataframe): dataframe containing each molecule and the idealized fret 

    Returns:
        dataframe: returns dataframe containing the duration of each FRET state from each molecule
    """
    df_test2 = []
    for Molecule, dfs in df.groupby('unique_id'):
        frame_length = len(dfs)
        test = dfs.groupby([dfs['idealized FRET'].ne(dfs['idealized FRET'].shift()).cumsum(), 'idealized FRET']).size()
        test = test.reset_index(level=1, drop=False)
        test['Molecule']=Molecule
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
        # ----------- create transition and delete last state ------------------
        thing1 = dfs.assign(FRET_after=dfs.FRET_state.shift(-1)).dropna()
        df_toconcat.append(thing1)
    compiled_df = pd.concat(df_toconcat).reset_index(drop=True)
    compiled_final = compiled_df[['Molecule', 'FRET_state', 'FRET_after', 'Time', 'number_of_frames']]
    compiled_final.columns = ['Molecule', 'FRET_before', 'FRET_after', 'Time', 'number_of_frames']
    return compiled_final

def remove_outliers_tdp(compiled_TDP):
    outliers = compiled_TDP[(compiled_TDP["FRET_before"] < -0.5)|(compiled_TDP["FRET_before"] > 1.5)|(compiled_TDP["FRET_after"] < -0.5) | (compiled_TDP["FRET_after"] > 1.5)].index
    compiled_TDP.drop(outliers, inplace=True)
    return compiled_TDP


# -------------- calculate the proportion of molecules that go below a defined threshold -----------------------------------

def filter_tdp(dfs, FRET_value):
    """function that selects all those molecules that go below a defined FRET value at some point and subsets them into a dataframe.

    Args:
        dfs (df): Raw dataframe containing all molecules with all transitions denoted.
        FRET_value (float): value to subset data. Function will look for any state (either before or after transition) that goes below this state and filter.

    Returns:
        df: dataframe containing all molecules that have a defined state below the FRET_value (at some point)
    """
    filt_tdp = []
    for treatment, df in dfs.groupby('treatment_name'):
        mol_list = df[(df["FRET_before"] <= FRET_value)|(df["FRET_after"] <= FRET_value)].Molecule.unique().tolist()
        filtered = df[df["Molecule"].isin(mol_list)]
        filt_tdp.append(filtered)
    return pd.concat(filt_tdp)

def count_filt_mol(dfs, thresh):
    """Will count the number of molecules in which the idealized FRET will go below a defined threshold at some point before photobleaching

    Args:
        df (dataframe): Contains all the data required to plot TDP and identify transitions below threshold (i.e., FRET, idealized FRET, molecule)
        thresh (float): Threshold to set. If set to 0.5, function will count the number of molecules that go below 0.5 at some point
        dataname (dict): Dictionary containing keys for each treatment - used to find mol count for each treatment

    Returns:
        dataframe: Will return dataframe with raw mol count and also corrected mol count. Corrected mol count is calculated as the Raw mol count subtracted
        by the molcount of another treatment. The treatment to subtract is defined by 'order', which is the index of the treatment you want to subtract
    """
    filtered_data = filter_tdp(dfs, thresh)
    percent_mol_concat = []
    for treatment, df in dfs.groupby('treatment_name'):
        total_mol = len(df.Molecule.unique())
        filt_mol = len(filtered_data[filtered_data['treatment_name']==treatment].Molecule.unique())
        df['proportion_mol_below_thresh'] = (filt_mol/total_mol)*100
        percent_mol_concat.append(df)
    return pd.concat(percent_mol_concat)

def determine_if_in_sequence(df):
    """This function creates a new column that tells you if a row is part of a sequence or run of similar values in the 'transition type' column.
    It first creates a new column 'shift' which has the values from 'transition type' shifted down by 1 and then using a 
    set of criteria which is nested in the for loop it will create a new column 'is_in_sequence' to determine if each row is part of a sequence
    of identical 'transition type' values. 

    Args:
        df (dataframe): dataframe for a single molecule within a specific treatment. Needs to be done on a per molecule basis.

    Returns:
        dataframe: dataframe with the additional columns, which can be fed into the next function (determine_cumulative_sum_in_sequence)
    """
    df['shift'] = df["transition_type"].shift()
    df["transition_type"] == df["shift"]
    data = list(df["transition_type"] == df["shift"])
    data_short = data[:-1]

    data_df = []
    for x, dfs in enumerate(data_short):
        dfs
        if x == 0 and data[x+1] == True:
            value = True
            data_df.append(value)
        elif dfs == True:
            value = True
            data_df.append(value)
        elif dfs == True and (data[x+1] == False):
            value = True
            data_df.append(value)
        elif dfs == False and (data[x+1] == True):
            value = True
            data_df.append(value)
        else:     
            value = False
            data_df.append(value)
    if data[-1] == True:
        value=True
        data_df.append(value)
    elif data[-1] == False:
        value = False
        data_df.append(value)
    df['is_in_sequence'] = data_df
    return df

def determine_cumulative_sum_in_sequence(df, exposure):
    """Takes a dataframe and adds a column called 'CumulativeTime'. What this function does is that it uses the 'is_in_sequence'
    column and it will identify a 'run' of True values and calculate the cumulative sum of all True values in the 'run'
    of True values and update the 'CumulativeTime' value in the row that breaks the run (i.e., the first False value after
    a run of True values) - these are the low-high or high-low transitions preceeded by multiple low-low or high-high transitions.
    The code the finds all low-high or high-low that are not preceeded by a sequence of True values and updates the 
    CumulativeTime column with the value that is in the 'Time' column - i.e., the rest of the low-high or high-low transitions
    not encompassed by the cumulative residence times.

    Args:
        df (dataframe): dataframe containing 'is_in_sequence' column originated from the 'determine_if_in_sequence' function
        exposure (variable): defined at top of script (in seconds), used to convert CumulativeTime (which is actually in frames) to a unit of time.

    Returns:
        dataframe: dataframe containing the CumulativeTime of all low-high or high-low transitions. Note that although high-high
        or low-low transitions are still present, these will be equal to 0. 
    """
    df['CumulativeTime'] = 0
    # Initialize variables to track the current run of True values and the cumulative sum
    current_run = 0
    cumulative_sum = 0
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        if row['is_in_sequence']:
            current_run += 1
            cumulative_sum += row['Time']
        else:
            if current_run > 0:
                # Update the cumulative sum in the row that breaks the run
                df.at[index, 'CumulativeTime']=cumulative_sum
                current_run = 0
                cumulative_sum = 0
    mask = (df['CumulativeTime'] == 0) & (df['transition_type'].isin(['low-high', 'high-low']))
    df.loc[mask, 'CumulativeTime']=df.loc[mask, 'Time']
    df.dropna(subset=['Molecule'],inplace=True)
    df['CumulativeTime(s)'] = df['CumulativeTime']*exposure
    return df
