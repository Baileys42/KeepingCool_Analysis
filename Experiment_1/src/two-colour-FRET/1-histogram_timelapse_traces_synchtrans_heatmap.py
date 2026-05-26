from smfret.src.plotting_scripts import histogram_plots as pa
from smfret.src.plotting_scripts import synchronised_transition_plots as pg
from smfret.src.plotting_scripts import gaussian_plots as ph
from smfret.src.plotting_scripts import heatmap_liveflow_plots as pe
import pandas as pd


if __name__ == "__main__":

    output_folder = "Experiment_1/python_results"
    FRET_thresh = 0.5

    # ---------------------- plot histograms and gaussian fit --------------------------------------------------------
    compiled_df, filt_dfs = pa.master_histogram_func('test', output_folder=output_folder, thresh=FRET_thresh, swarmplot=False)
    compiled_df.groupby('treatment_name')['unique_id'].nunique()

    percent_df = ph.plot_gmm_fits(compiled_df, output_folder)

    # --------------------------------- synchronised transitions to generate heatmaps --------------------------------
    compiled_df = pd.read_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv')
    order = list(compiled_df['treatment_name'].unique())

    (percent_trans_meet_criteria_df, calculated_transitions_df,
     consecutive_from_dnak_release, nonconsecutive_from_dnak_release,
     filt_data, sycnchronised_data, combined_consec_nonconsec,
     heatmap_release_df, heatmap_capture_df, spec1, spec2) = pg.master_plot_synchronised_transitions(
        order=order,
        output_folder=output_folder,
        exposure=0.2,
        frames_to_plot=50,
        FRET_before=0.5,
        FRET_after=0.5,
        datatype="Proportion",
        filt=True,
        palette='mako',
        add_time=0,
        df=compiled_df
    )

    # --------------------------------- FRET transition KDE heatmaps -------------------------------------------------
    #change to heatmatp_release_df for release and heatmap_capture_df for capture
    final_post_transition_df = pe.plot_fret_transition_kde(heatmap_release_df, output_folder)
