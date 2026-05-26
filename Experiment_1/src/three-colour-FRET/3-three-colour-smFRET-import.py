from smfret.src.processing_scripts import histogram_processing as ps

rawdata_folder = "Experiment_1-description/raw_data/"  ### Change for each experiment

input_folders = [

'/',
'/',
'/',
'/',
'/',
'/',
'/',
'/',
'/',
'/',

]
            
ps.move_folders(input_folders, ".txt", rawdata_folder)

