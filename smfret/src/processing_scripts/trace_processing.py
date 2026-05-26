import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

def load_data(filename, exposure):
    trace_df = pd.DataFrame(np.loadtxt(filename))
    trace_df.columns = ["frames", "donor", "acceptor", "FRET", "idealized FRET"]
    trace_df["Time"] = trace_df["frames"]/(1/exposure)
    trace_df["smoothed_FRET"] = savgol_filter(trace_df["FRET"], 5, 1)
    return trace_df


