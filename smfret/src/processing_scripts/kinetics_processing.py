import pandas as pd
import functools
from smfret.src.Utilities.Data_analysis import cleanup_dwell

def count_chaperone_events(dfs, thresh, fps_clean, thresh_clean):
    """Function to count the number of times that each molecule will go below a defined threshold from above the set threshold 'i.e. chaperone on' and vice versa 'i.e. chaperone off'

    Args:
        dfs (dataframe): dataframe containing raw TDP data, will be TDP_data
        thresh (variable): defines the minimum duration of a FRET state that can be included for analysis. Any dwell time that is shorter than this variable (in seconds) is deleted and not used for subsequent analysis.
        fps_clean (variable): previously defined threshold outlining the exposure rate. Is used to convert the dataset dwell times from frames to units of time.
        thresh_clean (variable): variable that has been defined previously that dictates the threshold with which the FRET must cross to be counted

    Returns:
        dataframe: dataframe that contains all the molecules that meet the criteria. Columns contain 'molecule' which provide the molecule number, 'FRET_after' which indicates the number of events from 
        above threshold to below threshold, 'FRET_below' which indicates the number of events from below threshold to above threshold and 'Total Molecule Lifetime (min)' which is how long the molecule 
        was imaged before photobleaching occurs.
    """
    cleaned_df = []
    for treatment_name, df in dfs.groupby("treatment_name"):
        initial_data = df[df["treatment_name"] == treatment_name]    
        cleaned = cleanup_dwell(initial_data, fps_clean, thresh_clean)
        cleaned_df.append(cleaned)
    cleaned_concat = pd.concat(cleaned_df)
    cleaned_concat['Total Molecule Lifetime (min)'] = (cleaned_concat['number_of_frames']/5)/60
    filt = []
    for treatment_name, df in cleaned_concat.groupby("treatment_name"):
        treatment = treatment_name
        chaperone_on = df[(df['FRET_after'] <= thresh) & (df['FRET_before'] >= thresh)].groupby('Molecule').count()['FRET_after'].reset_index()
        chaperone_off = df[(df['FRET_after'] >= thresh) & (df['FRET_before'] <= thresh)].groupby('Molecule').count()['FRET_before'].reset_index()
        time = df.groupby('Molecule').mean()['Total Molecule Lifetime (min)'].reset_index()
        merged_test = functools.reduce(lambda left, right: pd.merge(left, right, on='Molecule', how='outer'), [chaperone_on, chaperone_off, time]) ### Really usefull code for merging multiple dfs
        merged_test['treatment'] = treatment
        filt.append(merged_test)
    count_data = pd.concat(filt)
    test = pd.DataFrame(count_data)
    test.dropna(subset=['FRET_after', 'FRET_before'], how='all', inplace=True)
    test.fillna(0, inplace=True)
    return test
 


def find_large_transitions(dfs, delta_thresh):
    """Finds the proportion of transitions that are larger than a defined FRET threshold

    Args:
        dfs (dataframe): dataframe containing raw TDP data
        delta_thresh (float): this variable denotes the minimum change in FRET state during a transition to be counted as a 'large transition'. 

    Returns:
        dataframe: generates new dataframe with columns that indicate the number of molecules containing large transitions and the proportion of total transitions that are large for each 
        treatment
    """
    mol_with_large_trans = []
    for treatment, df in dfs.groupby('treatment_name'):
        filt = df[df['FRET_trans_difference'] > delta_thresh]
        filt_count_mol = pd.DataFrame(filt[filt['FRET_trans_difference'] > delta_thresh].agg({"Molecule": "nunique"})/df.agg({"Molecule": "nunique"})*100)
        filt_count_mol['treatment'] = treatment
        filt_count_mol['proportion_of_mol'] = (filt['Molecule'].count()/df['Molecule'].count())*100
        mol_with_large_trans.append(filt_count_mol)
    dfs = pd.concat(mol_with_large_trans)
    dfs = pd.DataFrame(dfs).reset_index()
    dfs.drop(columns='index', inplace=True)
    dfs.columns=['proportion_mol_large_transition', 'treatment', 'proportion_of_large_transitions']
    return dfs


#### Directory for above should come from the Dwell_times folder in python_results

def compiled(df, data_name, FRET_thresh):
    """Will filter transitions dependent on a threshold defined above as FRET_thresh to calculate residenc time for each transition class

    Args:
        df (dataframe): dataset containing the residence times  for each treatment
        data_name (string): treatment name  

    Returns:
        dataframe: compiles all transition classes (with residence times) from all treatments together
    """
    violin_data_lowtolow = pd.DataFrame(df[f"< {FRET_thresh} to < {FRET_thresh}"])
    violin_data_lowtolow.columns = ["y_axis"]
    violin_data_lowtolow["transition_type"] = f"< {FRET_thresh} to < {FRET_thresh}"
    violin_data_lowtolow["treatment"] = data_name

    violin_data_lowtohigh = pd.DataFrame(df[f"< {FRET_thresh} to > {FRET_thresh}"])
    violin_data_lowtohigh.columns = ["y_axis"]
    violin_data_lowtohigh["transition_type"] = f"< {FRET_thresh} to > {FRET_thresh}"
    violin_data_lowtohigh["treatment"] = data_name

    violin_data_hightohigh = pd.DataFrame(df[f"> {FRET_thresh} to > {FRET_thresh}"])
    violin_data_hightohigh.columns = ["y_axis"]
    violin_data_hightohigh["transition_type"] = f"> {FRET_thresh} to > {FRET_thresh}"
    violin_data_hightohigh["treatment"] = data_name

    violin_data_hightolow = pd.DataFrame(df[f"> {FRET_thresh} to < {FRET_thresh}"])
    violin_data_hightolow.columns = ["y_axis"]
    violin_data_hightolow["transition_type"] = f"> {FRET_thresh} to < {FRET_thresh}"
    violin_data_hightolow["treatment"] = data_name
    return pd.concat(
        [
            violin_data_lowtolow,
            violin_data_lowtohigh,
            violin_data_hightohigh,
            violin_data_hightolow,
        ]
    )

