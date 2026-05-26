import os as os
from smfret.src.processing_scripts import simulation_scripts as aa
from smfret.src.plotting_scripts import simulation_plots as pp

#  -------------------------------- Establish save locations and define rates --------------------------
plot_export = 'Fig3_Fluc-mutants-merged/python_results/Simulations/'
if not os.path.exists(plot_export):
    os.makedirs(plot_export)

col_export = 'Fig3_Fluc-mutants-merged/python_results/Simulations/col/'
if not os.path.exists(col_export):
    os.makedirs(col_export)

rate_up_palette = {
    0.52:'#1f77b4', 
    0.46:"#b2b8bd" }

pp.simple_sim_test(plot_export, 
                num_states=5, 
                rate_up=0.4, 
                rate_down=0.11, 
                simulation_time=360, 
                time_step=1, 
                seed=None)

results = aa.simulate_and_analyze_states(num_states_range=range(3, 7),
                            rate_up_list=[0.52, 0.46],  # Fast rate for IDS1_KJE
                            rate_down=0.17,  # Fast rate for cDS1_KJEG
                            simulation_time=360,
                            time_step=1,
                            threshold_state=1,
                            noise_level=0.05,
                            num_simulations=500,
                            identifier_prefix='IDS1_KJE',
                            plot_export=plot_export)


# -------------------------------- plot more formal analyses -----------------------------------------------

(combined_average_state_df,
 combined_concat, 
 proportion_time_combined, 
 transitions_x_to_y_rate, 
 mean_dwell_times_by_num_states_rate_up, 
 fits_data_melted) = pp.master_simplot(results, col_export, rate_up_palette)

#  --------------------------- large scale kinetic analysis - takes time -----------------------------------

results_df = aa.kinetic_sequence_space_analysis(plot_export, 
                                    rate_up_list=[0.2, 0.5, 0.05], 
                                    rate_down_list=[0.025, 0.2, 0.01], 
                                    nstates=[3, 6], 
                                    threshold_state=1,
                                    simulation_time=360,
                                    time_step=1)

pp.plot_kinetic_map(results_df, plot_export=col_export)

