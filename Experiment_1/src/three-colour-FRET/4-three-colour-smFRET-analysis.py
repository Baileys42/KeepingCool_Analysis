from smfret.src.plotting_scripts import threecolour_trace_histogram_plots as pa
from smfret.src.plotting_scripts import threecolour_sync_transition_plot as pb

if __name__ == "__main__":

    output_folder = "Experiment_1-description/python_results"

    # ------------------------------- Fit traces with HMM ---------------------------------------
    data = {
        'treatment':'raw_data',
        'treatment':'raw_data',
    }

    pa.master_fit_HMM(data, output_folder=f'{output_folder}/', FRET_thresh=0.5)


    # -------------------------------------- plot traces --------------------------------------

    pa.plot_three_colour_traces(output_folder=f'{output_folder}/')


    # ----------------------- plot FRET and fluorescence histograms from 3-colour data ------------------------------------------


    colors = {
        "test2": "black", 
        # "treatment2":"grey", 
    }
    pa.plot_3colour_fret_hist(output_folder=output_folder, FRET_subplot1='Cy3 FRET cascade', FRET_subplot2='AF647 FRET cascade', FRET_thresh=0.5, palette=colors)

    palette_for_treatment = {
        'test2':{'RNA-bound':'black', 'Unbound':'grey'}, 
        # 'RNA09-AF488':{'RNA-bound':'royalblue', 'Unbound':'skyblue'}, 
        # 'RNA09-duplex':{'RNA-bound':'purple', 'Unbound':'orchid'}, 
        # 'RNA01-AF488_col':{'RNA-bound':'black', 'Unbound':'grey'}, 
        }
    
    pa.plot_3colour_by_treatment(output_folder, FRET_thresh=0.5, dye='AF647 at 488', palette=palette_for_treatment)

    flurophore_palette = {
        'AF488 at 488':{'RNA-bound':'#0077b6', 'Unbound':'#caf0f8'}, 
        'Cy3 at 488':{'RNA-bound':'#31572c', 'Unbound':'#dde5b6'}, 
        'AF647 at 488':{'RNA-bound':'#ae2012', 'Unbound':'#e9d8a6'}, 
        # 'probe_summed_fluorescence':{'RNA-bound':'#ae2012', 'Unbound':'#e9d8a6'}, 
    }

    pa.plot_3colour_by_dye(output_folder, FRET_thresh=0.5, palette=flurophore_palette)

    # ----------------------------- plot both the change in FRET and fluorescence for synchronised transitions --------------------------------


    order = ['RNA01']
    list_to_drop = ['']

    pb.master_3color_synch_transitions(output_folder=output_folder, 
                                       exposure=0.2, 
                                       frames_to_plot=50, 
                                       FRET_before=0.5, 
                                       FRET_after=0.5, 
                                       order=order, 
                                       list_to_drop=list_to_drop)


