from smfret.src.plotting_scripts import TDP_plots as pi
from smfret.src.plotting_scripts import dwelltime_plots as pj
from smfret.src.plotting_scripts import dwelltime_plots as pk
from smfret.src.plotting_scripts import residence_time_plots as pl

if __name__ == "__main__":

    output_folder = 'Experiment_1/python_results'
    transition_threshold = 0.5
    # ------------------------------- generate transitions and plot TDPs ----------------------------------------------------------------
    pi.master_tdp_cleanup_func(output_folder=output_folder, 
                            exposure=0.2, 
                            FRET_value_for_classes=transition_threshold,
                            FRET_to_filt_tdp=transition_threshold)


    TDP_palette = {}
    for i in order:
        TDP_palette[i] = 'BuPu'
    
    pi.master_TDP_plot(TDP_palette, input_folder=output_folder, filt=True, if_chap=False)


    # ----------------------------- plot dwell time data (e.g. bind-and-release, FRET before or after transition) -----------------------


    bind_release_col = pj.master_dwell_time_func(order=order, 
                        output_folder=output_folder, 
                        palette_to_use='BuPu', 
                        FRET_thresh=transition_threshold, #### FRET value at which to filter data above or below. 
                        thresh_for_events=transition_threshold, #### FRET value to threshold for count chaperone events (i.e., binding and unbinding)
                        fps=5,  ### frames per second
                        thresh=2, ### should be 10x expsoure if using NL515 smoothing on MASH FRET
                        Transition_threshold=transition_threshold,
                        event='binding_and_release'
    )     

    # compiled_df.groupby('treatment_name')['molecule_number'].nunique()
    # bind_release_col.groupby('treatment')['n'].max()
    # proportion_dynamic =  bind_release_col.groupby('treatment')['n'].max()/compiled_df.groupby('treatment_name')['molecule_number'].nunique()

    # ----------------------------- plot transition frequence data ---------------------------------------------------------------

    sorted_list = sorted(order)   
    index_mapping = {value: index for index, value in enumerate(sorted_list)}
    indexes = [index_mapping[value] for value in order]
    pk.transition_frequency_plot(output_folder, indexes, FRET_thresh=transition_threshold)

    # ---------------------------- plot residence time data (reports as mean +- SE, violin and cumulative histogram) -------------------------------------------
    
    data_paths_violin = {}
    for treatment in order:
        data_paths_violin[treatment] = f"{output_folder}/Dwell_times/Filtered_dwelltime_{treatment}.csv"


    
    final, collated, cumulative_dwell_filt, fits_df, halftime_summary = pl.master_residence_time_func(output_folder,
                                                                    data_paths_violin,
                                                                    order,
                                                                    palette='BuPu',
                                                                    FRET_thresh=transition_threshold,
                                                                    binwidth=10,
                                                                    cumulative_hist_binwidth=1,
                                                                    fit_xlim=300,
                                                                    plot_xlim=100,
                                                                    func='both',
                                                                    biexponential=True
                                                                    )

