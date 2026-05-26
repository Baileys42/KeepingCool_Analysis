import shutil
import os
from smfret.src.Utilities import Data_analysis as util
import pandas as pd

def move_folders(input_folders, filetype, output_folder):
    for folder in input_folders:
        new_folder = folder.split("/")[-2]
        if not os.path.exists(f"{output_folder}/{new_folder}/"):
            os.makedirs(f"{output_folder}/{new_folder}/")
        filelist = [filename for filename in os.listdir(folder) if filetype in filename] # create list of files
        for filename in filelist:
            shutil.copyfile(f"{folder}{filename}", f"{output_folder}/{new_folder}/{filename}")

def locate_raw_drive_files(input_path='raw_data/raw_data.txt'):
    """Collects path to raw data from txt file containing plain text address
    Parameters
    ----------
    input_path : str, optional
        plain text address for raw_data files, by default 'raw_data/raw_data.txt'
    Returns
    -------
    str
        path to raw data
    """
    with open(input_path) as f:
        print(f"Raw data paths located in {input_path} have been collected.")
        data_path = [line[:-1] for line in f.readlines()]
    return data_path


def combine_technical_repeats(data, output_folder):
    data['Directory'] = data['Directory'].str.replace('\\', '/')
    data.dropna(inplace=True)
    compiled_df = []
    for (treatment, repeat, directory), df in data.groupby(['Treatment', 'Repeat', 'Directory']):
        imported_data = util.file_reader(directory, 'hist')
        print(f"Data for {treatment} repeat {repeat} imported from {directory}")
        cleaned_raw = util.remove_outliers(imported_data, 'hist') # add "idealized" after imported_data to get idealized histograms
        cleaned_raw["treatment_name"] = treatment
        cleaned_raw["repeat"] = repeat
        cleaned_raw["unique_id"] = cleaned_raw['molecule number'].astype(str) + '_' + cleaned_raw['treatment_name'] + '_' + cleaned_raw['repeat'].astype(str)
        compiled_df.append(cleaned_raw)
    compiled_df = pd.concat(compiled_df)   #### .rename(columns = {1:"test", 3:"test2"}) ## can rename individually if needed
    compiled_df.columns = ["frames", "donor", "acceptor", "FRET", "idealized FRET", 'molecule_number', "treatment_name", 'repeat', 'unique_id']
    compiled_df.to_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv', index=False)

    # Create sep_repeat dataframe with merged treatment_name and repeat
    sep_repeat = compiled_df.copy()
    sep_repeat['treatment_name'] = sep_repeat['treatment_name'].astype(str) + '_' + sep_repeat['repeat'].astype(str)
    sep_repeat.to_csv(f'{output_folder}/sep_repeat.csv', index=False)

    return compiled_df
