from smfret.src.processing_scripts import data_import_script as ps
import pandas as pd
rawdata_folder = "Experiment_1-description/raw_data"  ## Change for each experiment
input_folders = ps.locate_raw_drive_files(f'{rawdata_folder}/raw_data.txt')
ps.move_folders(input_folders, '.dat', rawdata_folder)

# -------- Data from all data sets in the dict will be imported and concatenated into a single dataframe. Outliers wil be removed -----------

output_folder = 'Experiment_1/python_results'
data = pd.read_csv('Experiment_1/raw_data/Example_data_file.csv')
compiled_df = ps.combine_technical_repeats(data, output_folder)
