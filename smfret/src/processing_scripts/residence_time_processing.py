import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os as os 
import warnings

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


def compiled_tri(df, data_name, low_thresh, high_thresh):
    """
    Compile residence times for all tri-state transitions
    low  = FRET < low_thresh
    mid  = low_thresh <= FRET < high_thresh
    high = FRET >= high_thresh
    """

    # readable shorthands for the column keys
    low  = f"< {low_thresh}"
    mid  = f"{low_thresh}–{high_thresh}"
    high = f"> {high_thresh}"

    # create a lookup table of all 9 transitions expected in df
    transition_pairs = [
        (low,  low),
        (low,  mid),
        (low,  high),
        (mid,  low),
        (mid,  mid),
        (mid,  high),
        (high, low),
        (high, mid),
        (high, high)
    ]

    dfs = []
    for start, end in transition_pairs:
        col = f"{start} to {end}"
        if col not in df.columns:
            # if nothing in this class (or upstream extractor didn’t create it)
            continue

        tmp = pd.DataFrame(df[col]).copy()
        tmp.columns = ["y_axis"]
        tmp["transition_type"] = f"{start}-{end}".replace(" ", "")
        tmp["treatment"] = data_name
        dfs.append(tmp)

    if not dfs:
        raise ValueError(
            "No tri-state transition columns found. "
            "Check your thresholding/extraction stage."
        )

    return pd.concat(dfs, ignore_index=True)


def compiled_tri2(df, data_name, low_thresh, high_thresh):
    """
    Tri-state version of compiled():
    - Expects df to have columns named by transition_type:
      'low-low', 'low-mid', 'low-high',
      'mid-low', 'mid-mid', 'mid-high',
      'high-low', 'high-mid', 'high-high'
    - Returns long-format dataframe with columns:
      ['y_axis', 'transition_type', 'treatment'].
    """

    # all possible 3-state transitions
    states = ['low', 'mid', 'high']
    transition_types = [f"{s1}-{s2}" for s1 in states for s2 in states]

    dfs = []
    for tr in transition_types:
        if tr not in df.columns:
            # might be missing for some treatments → skip gracefully
            continue

        tmp = pd.DataFrame(df[tr]).copy()
        tmp.columns = ["y_axis"]
        tmp["transition_type"] = tr
        tmp["treatment"] = data_name
        dfs.append(tmp)

    if not dfs:
        raise ValueError(
            f"No tri-state transition columns found in df for treatment {data_name}. "
            f"Expected something like 'low-low', 'low-mid', ..., 'high-high'."
        )

    return pd.concat(dfs, ignore_index=True)


def one_phase_association(x, Y0, Plateau, K):
    return Y0 + (Plateau-Y0)*(1-np.exp(-K*x))

def MLE(x,params):
    k1,k2,frac = params
    if not (0 < frac < 1):
        return np.inf
    yPred = frac * k1 * np.exp(k1 * x) + (1 - frac) * k2 * np.exp(k2 * x)
    # Avoid zero or negative predictions
    yPred = np.clip(yPred, 1e-12, None)
    negLL = -np.sum(np.log(yPred))
    return negLL

def double_expon(x, a,b,c,d):
    return a*np.exp(b*x)+c*np.exp(d*x)
def MLE_cdf(params, x):
    """Negative Log Likelihood function for double exponential."""
    k1, k2, frac = params
    
    # Hard constraints
    if not (0.01 <= frac <= 0.99): 
        return np.inf
    if k1 <= 1e-5 or k2 <= 1e-5: 
        return np.inf
    
    # PDF: P(t) = frac*k1*exp(-k1*t) + (1-frac)*k2*exp(-k2*t)
    pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
    pdf_vals = np.clip(pdf_vals, 1e-12, None)
    return -np.sum(np.log(pdf_vals))
def CDF_mixture(x, k1, k2, frac):
    """Double exponential CDF mixture model."""
    return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))
# Assuming one_phase_association is defined elsewhere (e.g., globally or imported)

# Assuming one_phase_association is defined elsewhere (e.g., globally or imported)

def cumulative_residence_fitting_actualcodedontdelete(dfs, output_folder, bin_width, xlim, func=one_phase_association):
    """Function is used to fit cumulative histogram data with a one-phase association curve. The script will create bins from the raw data and create a cumulative histogram, which is then
    used to fit the curve to the data. Will return the fit (with half time, plateua, etc) and a an Rsquared value to provide a measure of goodness of fit.
    Args:
        dfs (df): dataframe containing raw data to be used for fitting.
        output_folder (str): where to save data.
        bin_width (float): bin_width used to calculate the fit. Recommended to use smaller bin_widths (especially if data is tightly distributed at low values), but note
                            smaller bin_widths will reduce the number of datapoints in each bin.
        xlim (float): value used to determine how far the fit will extend to. Recommended to extend to max possible bin value.
        func (float, optional): decide here what fit to use. Defaults to one_phase_association. Can call another fit as long as it has been previously defined in a function.
    Returns:
        df: returns the fits and also the summary data (containing the half-time for each treatment and residence time state) AND the raw bootstrap results.
    """
    data = []
    summary = []
    # --- ADDITION 1: Initialize list for raw bootstrap data ---
    all_bootstrap_data = [] 
    
    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        print(treatment)
        print(transition)
        bin_width = bin_width
        bin_edges = range(0, xlim+1, bin_width)
        # Bin the 'CumulativeTime(s)' column
        bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False) 
        # Count the number of values in each bin
        bin_counts = bins.value_counts().sort_index()
        # Calculate cumulative count
        cumulative_counts = bin_counts.cumsum()
        bin_edges_array = bin_edges[:-1]
        # Exclude the last edge to match the length of bin_counts
        cumulative_counts_array = cumulative_counts.values
        cumulative_counts_array = cumulative_counts_array/cumulative_counts_array.max()
        # plt.show()
        fits_output = []
        if func == 'both':
            print('single-fit first')
            p0 = (20, 100, 0.1) 
            params, cv = scipy.optimize.curve_fit(one_phase_association, bin_edges_array, cumulative_counts_array, p0)
            Y0, Plateau, K = params
            tauSec = (1 / K)
            half_time = np.log(2) / K
            # Calculate standard error of half-time
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2
            # 95% confidence interval
            alpha = 0.05
            z_score = scipy.stats.norm.ppf(1 - alpha / 2)
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            n_value = cumulative_counts.max()
            print(treatment)
            print(transition)
            print('bruh')
            print("Estimated Half-Time:", half_time)
            print("Standard Error of Half-Time:", se_half_time)
            print("95% Confidence Interval of Half-Time:", ci_half_time)
            # R² of single fit
            squaredDiffs = np.square(cumulative_counts_array - one_phase_association(bin_edges_array, Y0, Plateau, K))
            squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

            fitted_data = one_phase_association(bin_edges_array, Y0, Plateau, K)
            fitted_data_df = pd.DataFrame(fitted_data)
            bin_edges_hist = np.arange(0, xlim + bin_width, bin_width) # Define here for consistency, though not used in next 3 lines
            # bin_centers = bin_edges[:-1] + bin_width / 2 # This was correctly defined later, so leaving it out here.
            res_single = cumulative_counts_array - fitted_data
            x_bins = pd.DataFrame(bin_edges_array)
            test_single = pd.DataFrame(cumulative_counts_array)
            test_single = pd.concat([test_single, fitted_data_df, x_bins, pd.DataFrame(res_single)], axis=1)
            test_single.columns = ['Cumative_hist_sing', 'fit_sing', 'x_bins_sing', 'residuals_sing']
            test_single['treatment'] = treatment
            test_single['transition_type'] = transition
            data.append(test_single)
            print(f"MLE (CDF) fitting for {treatment} {transition}")
            # MLE part
            dwell_times = df["CumulativeTime(s)"].values
            dwell_times = dwell_times[dwell_times > 0]
            
            # --- Definitions for MLE ---
            def CDF_mixture(x, k1, k2, frac):
                return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))
                
            def MLE_cdf(params, x):
                k1, k2, frac = params
                if not (0.01 <= frac <= 0.99): # Corrected range constraint here
                    return np.inf
                if k1 <= 1e-4 or k2 <= 1e-4: # Corrected lower bound here
                    return np.inf
                pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
                pdf_vals = np.clip(pdf_vals, 1e-12, None)
                return -np.sum(np.log(pdf_vals))
            # --- End Definitions ---

            initial_guess = [1/20, 1/300, 0.5]
            bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
            result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
            k1, k2, frac = result.x
            
            # --- ADDITION 2: Enforce k1 is the FAST rate (k1 > k2) ---
            if k1 < k2:
                k1, k2 = k2, k1
                frac = 1 - frac

            print(f"Fitted: k1={k1:.4f}, k2={k2:.4f}, frac={frac:.4f}")
            # ----------- BOOTSTRAP TO ESTIMATE STANDARD ERRORS -----------
            n_bootstrap = 500
            bootstrap_frac = []
            bootstrap_k1 = []
            bootstrap_k2 = []

            for i in range(n_bootstrap):
                boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
                result_boot = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(boot_sample,), bounds=bounds)
                if result_boot.success:
                    k1_boot_raw, k2_boot_raw, frac_boot_raw = result_boot.x
                    
                    # --- ADDITION 2 continued: Enforce k1 is the FAST rate on bootstrap results ---
                    if k1_boot_raw < k2_boot_raw:
                        k1_boot, k2_boot = k2_boot_raw, k1_boot_raw
                        frac_boot = 1 - frac_boot_raw
                    else:
                        k1_boot, k2_boot, frac_boot = k1_boot_raw, k2_boot_raw, frac_boot_raw
                        
                    bootstrap_frac.append(frac_boot)
                    bootstrap_k1.append(k1_boot)
                    bootstrap_k2.append(k2_boot)
                    
                    # --- ADDITION 1 continued: Store the individual run result ---
                    all_bootstrap_data.append({
                        'treatment': treatment,
                        'transition_type': transition,
                        'k1_boot': k1_boot,
                        'k2_boot': k2_boot,
                        'frac_boot': frac_boot,
                        'bootstrap_run': i
                    })
                        
            bootstrap_frac = np.array(bootstrap_frac)
            bootstrap_k1 = np.array(bootstrap_k1)
            bootstrap_k2 = np.array(bootstrap_k2)
            # Calculate bootstrap statistics
            def bootstrap_summary(values, name):
                mean = np.mean(values)
                std_error = np.std(values, ddof=1)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                print(f"Bootstrap {name}: mean={mean:.4f}, std error={std_error:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
                return mean, std_error, ci_lower, ci_upper
            frac_mean_bootstrap, frac_std_error, frac_ci_lower, frac_ci_upper = bootstrap_summary(bootstrap_frac, 'frac')
            k1_mean_bootstrap, k1_std_error, k1_ci_lower, k1_ci_upper = bootstrap_summary(bootstrap_k1, 'k1')
            k2_mean_bootstrap, k2_std_error, k2_ci_lower, k2_ci_upper = bootstrap_summary(bootstrap_k2, 'k2')
            # ----------- END BOOTSTRAP -----------
            bin_edges = np.arange(0, xlim + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2
            hist, _ = np.histogram(dwell_times, bins=bin_edges)
            cum_counts = np.cumsum(hist)
            cum_norm = cum_counts / cum_counts[-1]
            fitted_cdf = CDF_mixture(bin_centers, k1, k2, frac)
            # R² of double fit
            ss_res = np.sum((cum_norm - fitted_cdf) ** 2)
            ss_tot = np.sum((cum_norm - np.mean(cum_norm)) ** 2)
            r_squared_double = 1 - ss_res / ss_tot
            print(f'{treatment} R2 is {r_squared_double}')
            # Residuals plot
            res = cum_norm - fitted_cdf
            fig = plt.figure(figsize=(6, 2))
            plt.scatter(bin_centers, res_single, label='single_fit', s=8, c='#f05423')
            plt.scatter(bin_centers, res, label='double_fit', s=8, c='#00aeef')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"Residuals for {treatment} {transition} ")
            plt.xlabel("Resident time (s)")
            plt.ylabel("Residual")
            plt.xlim(0, 100)
            plt.ylim(-0.25, 0.25)
            plt.legend()
            plt.show()
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=300)
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=600)
            # Half-times (k1 is fast, k2 is slow)
            half_time_fast = np.log(2) / k1
            half_time_slow = np.log(2) / k2
            fit_df = pd.DataFrame({
                'x_bins': bin_centers,
                'fit': fitted_cdf,
                'Cumative_hist': cum_norm,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(fit_df)
            col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K, rSquared,
                                     half_time_fast, half_time_slow, k1, k2, frac, frac_std_error, r_squared_double,
                                     k1_std_error, k2_std_error,bootstrap_frac.mean(), bootstrap_k1.mean(), bootstrap_k2.mean()]],
                            columns=['mean', 'sem', 'n', 'treatment', 'transition', 'K', 'r_squared',
                                     'half_time_fast', 'half_time_slow', 'k1', 'k2', 'frac_fast', 'frac_std_error', 'R2',
                                     'k1_std_error', 'k2_std_error', 'bootstrap_frac', 'bootstrap_k1', 'bootstrap_k2'])
            summary.append(col_df)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            # Bootstrap plot for frac
            sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[0], color="#939990FF", edgecolor='black')
            axs[0].axvline(frac, color='black', linestyle='--', label='Original Fit')
            axs[0].set_title('Bootstrap Distribution of frac')
            axs[0].set_xlabel('frac')
            axs[0].legend()
            # Bootstrap plot for k1
            sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="#386B20FF", edgecolor='black')
            axs[1].axvline(k1, color='black', linestyle='--', label='Original Fit')
            axs[1].set_title('Bootstrap Distribution of k1 (FAST)')
            axs[1].set_xlabel('k1 (1/s)')
            axs[1].legend()
            # Bootstrap plot for k2
            sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="#81b868df", edgecolor='black')
            axs[2].axvline(k2, color='black', linestyle='--', label='Original Fit')
            axs[2].set_title('Bootstrap Distribution of k2 (SLOW)')
            axs[2].set_xlabel('k2 (1/s)')
            axs[2].legend()
            plt.tight_layout()
            plt.show()
            # Optional: save the figure
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=300) 
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=600) 
        else:
            bin_width = bin_width
            bin_edges = range(0, xlim+1, bin_width)

            # Bin the 'CumulativeTime(s)' column
            bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False) 

            # Count the number of values in each bin
            bin_counts = bins.value_counts().sort_index()

            # Calculate cumulative count
            cumulative_counts = bin_counts.cumsum()

            bin_edges_array = bin_edges[:-1]
            # Exclude the last edge to match the length of bin_counts
            cumulative_counts_array = cumulative_counts.values
            cumulative_counts_array = cumulative_counts_array/cumulative_counts_array.max()
            # plt.show()

            # perform the fit
            p0 = (20, 100, 0.1) # start with values near those we expect
            params, cv = scipy.optimize.curve_fit(func, bin_edges_array, cumulative_counts_array, p0)
            Y0, Plateau, K = params
            tauSec = (1 / K)
            half_time = np.log(2)/K

            # Calculate standard errors from the covariance matrix
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2

            # Calculate confidence interval for the half-time (assuming normal distribution)
            alpha = 0.05 
            z_score = scipy.stats.norm.ppf(1 - alpha / 2) 
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            n_value = cumulative_counts.max()


            print("Estimated Half-Time:", half_time)
            print("Standard Error of Half-Time:", se_half_time)
            print("95% Confidence Interval of Half-Time:", ci_half_time)

            # determine quality of the fit
            squaredDiffs = np.square(cumulative_counts_array - func(bin_edges_array, Y0, Plateau, K))
            squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

            # inspect the parameters
            print(f"R² = {rSquared}")
            print(f"Y = {Y0}")
            print(f'Plateau = {Plateau}')
            print(f'K = {K}')
            print(f'half-time = {half_time} s')
            print(f"Tau = {tauSec} s")

            fitted_data = func(bin_edges_array, Y0, Plateau, K)
            fitted_data_df = pd.DataFrame(fitted_data)
            x_bins = pd.DataFrame(bin_edges_array)
            test = pd.DataFrame(cumulative_counts_array)
            test = pd.concat([test, fitted_data_df, x_bins],axis=1)
            test.columns = ['Cumative_hist', 'fit', 'x_bins']
            test['treatment'] = treatment
            test['transition_type'] = transition
            data.append(test)


            col = [half_time, se_half_time, n_value, rSquared, treatment, transition]
            col_halftime_df = pd.DataFrame([col], columns=['mean', 'sem', 'n', 'r_squared', 'treatment', 'transition'])
            summary.append(col_halftime_df)
            bootstrap_df = pd.DataFrame()  # Empty DataFrame when not using bootstrap

    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.concat(summary, ignore_index=True)
    
    # --- ADDITION 1: Create and return the bootstrap results dataframe ---
    bootstrap_df = pd.DataFrame(all_bootstrap_data)

    return fits_df, halftime_summary, bootstrap_df

#new chat slop

def cumulative_residence_fitting_singlefail(dfs, output_folder, bin_width, xlim, func='both', n_bootstrap=500):
    """
    Fits cumulative dwell time histograms with single or double exponential
    using binned inverse CDFs. Performs bootstrap and calculates AIC/BIC/LLR.
    Produces plots for residuals, bootstrap distributions, inverse and positive CDFs.
    """
    all_data = []
    all_summary = []
    all_bootstrap_data = []

    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        dwell_times = df['CumulativeTime(s)'].values
        dwell_times = dwell_times[dwell_times > 0]

        # --- Bin the data ---
        bin_edges = np.arange(0, xlim + bin_width, bin_width)
        hist, _ = np.histogram(dwell_times, bins=bin_edges)
        cum_counts = np.cumsum(hist)
        cum_norm = cum_counts / cum_counts[-1]  # normalized CDF
        inv_cdf = 1.0 - cum_norm
        bin_centers = bin_edges[:-1] + bin_width / 2

        # --- Single exponential fit ---
        def single_exp(t, k):
            return np.exp(-k * t)

        p0_single = [1/np.mean(dwell_times)]
        bounds_single = [(1e-6, 10)]

        try:
            popt_single, _ = scipy.optimize.curve_fit(
                single_exp, bin_centers, inv_cdf, p0=p0_single, bounds=bounds_single, maxfev=10000
            )
            singleK = popt_single[0]
            fitted_single = single_exp(bin_centers, singleK)
        except Exception as e:
            print(f"[WARN] Single exponential fit FAILED for {treatment}-{transition}: {e}")
            singleK = 0
            fitted_single = np.zeros_like(bin_centers)

        residuals_single = inv_cdf - fitted_single

        # --- Double exponential fit via MLE ---
        def neg_log_likelihood(params, data):
            k1, k2, frac = params
            if not (0.01 <= frac <= 0.99) or k1 <= 1e-6 or k2 <= 1e-6:
                return np.inf
            pdf_vals = frac * k1 * np.exp(-k1 * data) + (1 - frac) * k2 * np.exp(-k2 * data)
            pdf_vals = np.clip(pdf_vals, 1e-12, None)
            return -np.sum(np.log(pdf_vals))

        initial_guess = [1/20, 1/300, 0.5]
        bounds = [(1e-6, 10), (1e-6, 10), (0.01, 0.99)]

        try:
            result = scipy.optimize.minimize(
                neg_log_likelihood, initial_guess, args=(dwell_times,), bounds=bounds
            )
            if not result.success:
                raise RuntimeError(result.message)

            k1, k2, frac = result.x

            if k1 < k2:
                k1, k2 = k2, k1
                frac = 1 - frac

            fitted_double = frac * np.exp(-k1 * bin_centers) + (1 - frac) * np.exp(-k2 * bin_centers)

        except Exception as e:
            print(f"[WARN] Double exponential MLE FAILED for {treatment}-{transition}: {e}")
            k1 = k2 = frac = 0
            fitted_double = np.zeros_like(bin_centers)

        residuals_double = inv_cdf - fitted_double

        # --- Bootstrap ---
        boot_singleK, boot_k1, boot_k2, boot_frac = [], [], [], []

        for i in range(n_bootstrap):
            sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)

            # Single exponential bootstrap
            try:
                inv_cdf_sample = 1.0 - np.cumsum(np.histogram(sample, bins=bin_edges)[0]) / len(sample)
                popt_s, _ = scipy.optimize.curve_fit(
                    single_exp, bin_centers, inv_cdf_sample, p0=p0_single,
                    bounds=bounds_single, maxfev=10000
                )
                boot_singleK.append(popt_s[0])
            except:
                boot_singleK.append(0)

            # Double exponential bootstrap
            try:
                result_boot = scipy.optimize.minimize(neg_log_likelihood, initial_guess, args=(sample,), bounds=bounds)
                if not result_boot.success:
                    raise RuntimeError()
                k1b, k2b, fracb = result_boot.x
                if k1b < k2b:
                    k1b, k2b = k2b, k1b
                    fracb = 1 - fracb
            except:
                k1b = k2b = fracb = 0

            boot_k1.append(k1b)
            boot_k2.append(k2b)
            boot_frac.append(fracb)

            all_bootstrap_data.append({
                'treatment': treatment,
                'transition_type': transition,
                'singleK_boot': boot_singleK[-1],
                'k1_boot': k1b,
                'k2_boot': k2b,
                'frac_boot': fracb,
                'bootstrap_run': i
            })

        # --- AIC/BIC/LLR ---
        ss_single = np.sum((inv_cdf - fitted_single)**2)
        ss_double = np.sum((inv_cdf - fitted_double)**2)
        n_points = len(bin_centers)

        try:
            aic_single = n_points * np.log(ss_single/n_points) + 2*1
            bic_single = n_points * np.log(ss_single/n_points) + np.log(n_points)*1
        except:
            aic_single = bic_single = 0

        try:
            aic_double = n_points * np.log(ss_double/n_points) + 2*3
            bic_double = n_points * np.log(ss_double/n_points) + np.log(n_points)*3
        except:
            aic_double = bic_double = 0

        try:
            llr = 2*( -0.5*ss_single + -0.5*ss_double )
        except:
            llr = 0

        # --- Dataframe for plotting fits and residuals ---
        fit_df = pd.DataFrame({
            'x_bins': bin_centers,
            'single_fit': fitted_single,
            'double_fit': fitted_double,
            'inv_cdf': inv_cdf,
            'residual_single': residuals_single,
            'residual_double': residuals_double,
            'treatment': treatment,
            'transition_type': transition
        })
        all_data.append(fit_df)

        # --- Summary ---
        summary_df = pd.DataFrame({
            'treatment': [treatment],
            'transition_type': [transition],
            'singleK': [singleK],
            'k1': [k1],
            'k2': [k2],
            'frac': [frac],
            'AIC_single': [aic_single],
            'BIC_single': [bic_single],
            'AIC_double': [aic_double],
            'BIC_double': [bic_double],
            'LLR': [llr],
            'mean_boot_singleK': [np.mean(boot_singleK)],
            'mean_boot_k1': [np.mean(boot_k1)],
            'mean_boot_k2': [np.mean(boot_k2)],
            'mean_boot_frac': [np.mean(boot_frac)]
        })
        all_summary.append(summary_df)

        ###########################################################
        # YOUR ORIGINAL PLOTTING CODE — LEFT UNCHANGED
        ###########################################################

        # Residuals
        fig, ax = plt.subplots(1,1, figsize=(6,3))
        ax.scatter(bin_centers, residuals_single, label='single residual', color='orange')
        ax.scatter(bin_centers, residuals_double, label='double residual', color='blue')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(f'Residuals {treatment} {transition}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Residual')
        ax.legend()
        plt.show()

        # Inverse CDF fits
        fig, ax = plt.subplots(figsize=(6,3))
        ax.scatter(bin_centers, inv_cdf, label='Inverse CDF', color='black', s=8)
        ax.plot(bin_centers, fitted_single, '--', label='Single Exp Fit', color='orange')
        ax.plot(bin_centers, fitted_double, '-', label='Double Exp Fit', color='blue')
        ax.set_title(f'Inverse CDF Fits {treatment} {transition}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('1 - CDF')
        ax.legend()
        plt.show()

        # Positive CDF fits
        fig, ax = plt.subplots(figsize=(6,3))
        ax.scatter(bin_centers, cum_norm, label='CDF', color='black', s=8)
        ax.plot(bin_centers, 1 - fitted_single, '--', label='Single Exp Fit', color='orange')
        ax.plot(bin_centers, 1 - fitted_double, '-', label='Double Exp Fit', color='blue')
        ax.set_title(f'Positive CDF Fits {treatment} {transition}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('CDF')
        ax.legend()
        plt.show()

        # Bootstrap distributions
        fig, axs = plt.subplots(1,4, figsize=(16,3))
        sns.histplot(boot_singleK, bins=30, kde=True, ax=axs[0], color='orange')
        axs[0].set_title('Bootstrap singleK')
        sns.histplot(boot_k1, bins=30, kde=True, ax=axs[1], color='blue')
        axs[1].set_title('Bootstrap k1')
        sns.histplot(boot_k2, bins=30, kde=True, ax=axs[2], color='green')
        axs[2].set_title('Bootstrap k2')
        sns.histplot(boot_frac, bins=30, kde=True, ax=axs[3], color='purple')
        axs[3].set_title('Bootstrap frac')
        plt.tight_layout()
        plt.show()

    fits_df = pd.concat(all_data, ignore_index=True)
    halftime_summary = pd.concat(all_summary, ignore_index=True)
    bootstrap_df = pd.DataFrame(all_bootstrap_data)

    return fits_df, halftime_summary, bootstrap_df

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
def cumulative_residence_fitting_OGcolours(dfs, output_folder, bin_width, xlim, func='both', n_bootstrap=500):
    """
    Fit binned dwell time data using single or double exponentials on inverse CDF.
    Generates fits, residuals, bootstrap distributions, and AIC/BIC/LLR for model selection.
    """
    import scipy.optimize
    import scipy.stats
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import os as os 
    import warnings
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'

    # Ensure SVG text remains editable


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = []
    summary_rows = []
    all_bootstrap_data = []

    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        dwell_times = df['CumulativeTime(s)'].values
        dwell_times = dwell_times[dwell_times > 0]

        bin_edges = np.arange(0, xlim + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        # Build complementary CDF
        hist, _ = np.histogram(dwell_times, bins=bin_edges)
        cum_counts = np.cumsum(hist)
        if cum_counts[-1] == 0:
            continue
        comp_cdf = 1.0 - cum_counts / cum_counts[-1]

        # ----- SINGLE EXPONENTIAL FIT -----
        def single_exp(t, k):
            return np.exp(-k * t)

        p0_single = [1/np.mean(dwell_times)]
        bounds_single = ([1e-4], [10])

        try:
            popt_s, pcov_s = scipy.optimize.curve_fit(
                single_exp, bin_centers, comp_cdf,
                p0=p0_single, bounds=bounds_single, maxfev=10000
            )
            k_single = popt_s[0]
        except Exception:
            k_single = np.nan

        fit_single_vals = single_exp(bin_centers, k_single)
        res_single = comp_cdf - fit_single_vals

        # ----- DOUBLE EXPONENTIAL FIT (MLE) -----
        def MLE_double(params, x):
            k1, k2, frac = params
            if not (0.01 <= frac <= 0.99):
                return np.inf
            if k1 <= 1e-4 or k2 <= 1e-4:
                return np.inf
            pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
            pdf_vals = np.clip(pdf_vals, 1e-12, None)
            return -np.sum(np.log(pdf_vals))

        initial_guess = [1/20, 1/300, 0.5]
        bounds_double = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
        result = scipy.optimize.minimize(MLE_double, initial_guess,
                                         args=(dwell_times,), bounds=bounds_double)
        k1, k2, frac = result.x
        if k1 < k2:
            k1, k2 = k2, k1
            frac = 1 - frac

        fit_double_vals = frac * np.exp(-k1 * bin_centers) + (1 - frac) * np.exp(-k2 * bin_centers)
        res_double = comp_cdf - fit_double_vals

        # ----- BOOTSTRAP FOR SINGLE AND DOUBLE FIT -----
        bootstrap_k_single = []
        bootstrap_k1 = []
        bootstrap_k2 = []
        bootstrap_frac = []

        for i in range(n_bootstrap):
            boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
            hist_b, _ = np.histogram(boot_sample, bins=bin_edges)
            cum_b = np.cumsum(hist_b)
            if cum_b[-1] == 0:
                continue
            comp_cdf_b = 1.0 - cum_b / cum_b[-1]

            # Single exponential bootstrap
            try:
                popt_s_b, _ = scipy.optimize.curve_fit(
                    single_exp, bin_centers, comp_cdf_b,
                    p0=p0_single, bounds=bounds_single, maxfev=10000
                )
                bootstrap_k_single.append(popt_s_b[0])
            except Exception:
                continue

            # Double exponential bootstrap
            result_b = scipy.optimize.minimize(MLE_double, initial_guess,
                                               args=(boot_sample,), bounds=bounds_double)
            if result_b.success:
                k1_b, k2_b, frac_b = result_b.x
                if k1_b < k2_b:
                    k1_b, k2_b = k2_b, k1_b
                    frac_b = 1 - frac_b
                bootstrap_k1.append(k1_b)
                bootstrap_k2.append(k2_b)
                bootstrap_frac.append(frac_b)

                all_bootstrap_data.append({
                    'treatment': treatment,
                    'transition_type': transition,
                    'k_single': popt_s_b[0],
                    'k1_boot': k1_b,
                    'k2_boot': k2_b,
                    'frac_boot': frac_b,
                    'bootstrap_run': i
                })

        # ----- ADD THE BOOTSTRAP STATISTICS -----
        summary_rows.append({
            'treatment': treatment,
            'transition_type': transition,
            'k_single_mean': np.mean(bootstrap_k_single) if bootstrap_k_single else np.nan,
            'k_single_std' : np.std(bootstrap_k_single, ddof=1) if bootstrap_k_single else np.nan,
            'k1_mean': np.mean(bootstrap_k1) if bootstrap_k1 else np.nan,
            'k1_std' : np.std(bootstrap_k1, ddof=1) if bootstrap_k1 else np.nan,
            'k2_mean': np.mean(bootstrap_k2) if bootstrap_k2 else np.nan,
            'k2_std' : np.std(bootstrap_k2, ddof=1) if bootstrap_k2 else np.nan,
            'frac_mean': np.mean(bootstrap_frac) if bootstrap_frac else np.nan,
            'frac_std' : np.std(bootstrap_frac, ddof=1) if bootstrap_frac else np.nan
        })

        # ----- FIX MISSING FIT COLUMN -----
        fit_column_vals = fit_single_vals   # required by your seaborn code

        # ----- STORE FITS AND RESIDUALS -----
        fit_df = pd.DataFrame({
            'x_bins': bin_centers,
            'Cumative_hist': cum_counts / cum_counts[-1],
            'Cumative_hist_inv': comp_cdf,

            # Inverse CDF fit (your original fit)
            'fit': fit_single_vals,
            'fit_single': fit_single_vals,
            'fit_double': fit_double_vals,

            # NEW: Positive CDF fits → required by your plotting function
            'fit_pos_single': 1 - fit_single_vals,
            'fit_pos_double': 1 - fit_double_vals,

            'residual_single': res_single,
            'residual_double': res_double,
            'treatment': treatment,
            'transition_type': transition,
            'k_single': k_single,
            'k1': k1,
            'k2': k2,
            'frac': frac
        })

        data.append(fit_df)

        # ============================================================
        # ============== YOUR PLOTTING CODE WITH PATCHES ============
        # ============================================================

        # ----- BOOTSTRAP DISTRIBUTION PLOTS (NOW 4 PLOTS) -----
        fig, axs = plt.subplots(1,4,figsize=(20,4))
        sns.histplot(bootstrap_k_single, bins=30, kde=True, ax=axs[0], color="orange")
        axs[0].set_title('Bootstrap k_single')

        sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="green")
        axs[1].set_title('Bootstrap k1 (fast)')

        sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="blue")
        axs[2].set_title('Bootstrap k2 (slow)')

        # *** NEW PLOT ***
        sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[3], color="purple")
        axs[3].set_title('Bootstrap frac')

        plt.suptitle(f"Bootstrap Distributions: {treatment} {transition}")
        fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=300, format='svg', bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # ----- FIT PLOTS (NOW BOTH SHOW INVERSE CDF) -----
        fig, axs = plt.subplots(1,2,figsize=(12,4))

        # Panel 1: inverse CDF
        axs[0].scatter(bin_centers, comp_cdf, label='Data (1-CDF)', color='black')
        axs[0].plot(bin_centers, fit_single_vals, label='Single Exp Fit', c='orange')
        axs[0].plot(bin_centers, fit_double_vals, label='Double Exp Fit', c='blue')
        axs[0].set_title(f"Inverse CDF Fit {treatment} {transition}")
        axs[0].legend()
        # Panel 2: POSITIVE CDF + POSITIVE FITS
        axs[1].scatter(
            bin_centers,
            cum_counts / cum_counts[-1],          # positive CDF data
            label='Data (CDF)',
            color='black'
        )

        axs[1].plot(
            bin_centers,
            1 - fit_single_vals,                  # POSITIVE single-exponential fit
            label='Single Exp Fit (pos)',
            c='orange'
        )

        axs[1].plot(
            bin_centers,
            1 - fit_double_vals,                  # POSITIVE double-exponential fit
            label='Double Exp Fit (pos)',
            c='blue'
        )

        axs[1].set_title(f"Positive CDF Fit {treatment} {transition}")
        axs[1].legend()
        fig.savefig(f"{output_folder}/cdf_fits_{treatment}_{transition}.svg", dpi=300, format='svg', bbox_inches='tight')

        plt.show()
        plt.close(fig)

        # ----- RESIDUALS -----
        fig, axs = plt.subplots(1,1,figsize=(6,3))
        axs.scatter(bin_centers, res_single, label='Residual Single', s=10, c='orange')
        axs.scatter(bin_centers, res_double, label='Residual Double', s=10, c='blue')
        axs.axhline(0, color='gray', linestyle='--')
        axs.set_title(f"Residuals {treatment} {transition}")
        axs.legend()
        fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=300, format='svg', bbox_inches='tight')
        plt.show()
        plt.close(fig)

    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.DataFrame(summary_rows)
    bootstrap_df = pd.DataFrame(all_bootstrap_data)

    return fits_df, halftime_summary, bootstrap_df



def cumulative_residence_fitting(dfs, output_folder, bin_width, xlim, func='both', n_bootstrap=500):
    import scipy.optimize
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import os
    import matplotlib.pyplot as plt

    # ------------------ GLOBAL STYLE ------------------
    plt.rcParams.update({
        'svg.fonttype': 'none',
        'font.family': 'Arial',
        'font.size': 12
    })

    sns.set_theme(style="whitegrid")

    # ----------- PALETTES -----------
    crest = sns.color_palette("crest", 4)
    magma = sns.color_palette("magma", 4)

    col_single_magma = magma[2]
    col_fast_crest = crest[2]
    col_slow_crest = crest[3]
    col_frac_crest = crest[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = []
    summary_rows = []
    all_bootstrap_data = []

    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        dwell_times = df['CumulativeTime(s)'].values
        dwell_times = dwell_times[dwell_times > 0]

        bin_edges = np.arange(0, xlim + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        # ------------------ CDF ------------------
        hist, _ = np.histogram(dwell_times, bins=bin_edges)
        cum_counts = np.cumsum(hist)
        if cum_counts[-1] == 0:
            continue
        comp_cdf = 1.0 - cum_counts / cum_counts[-1]

        # ------------------ SINGLE EXP ------------------
        def single_exp(t, k):
            return np.exp(-k * t)

        p0_single = [1/np.mean(dwell_times)]
        bounds_single = ([1e-4], [10])

        try:
            popt_s, _ = scipy.optimize.curve_fit(
                single_exp, bin_centers, comp_cdf,
                p0=p0_single, bounds=bounds_single, maxfev=10000
            )
            k_single = popt_s[0]
        except Exception:
            k_single = np.nan

        fit_single_vals = single_exp(bin_centers, k_single)
        res_single = comp_cdf - fit_single_vals

        # ------------------ DOUBLE EXP (MLE) ------------------
        def MLE_double(params, x):
            k1, k2, frac = params
            if not (0.01 <= frac <= 0.99):
                return np.inf
            if k1 <= 1e-4 or k2 <= 1e-4:
                return np.inf
            pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
            pdf_vals = np.clip(pdf_vals, 1e-12, None)
            return -np.sum(np.log(pdf_vals))

        initial_guess = [1/20, 1/300, 0.5]
        bounds_double = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]

        result = scipy.optimize.minimize(
            MLE_double, initial_guess,
            args=(dwell_times,), bounds=bounds_double
        )

        k1, k2, frac = result.x
        if k1 < k2:
            k1, k2 = k2, k1
            frac = 1 - frac

        fit_double_vals = frac * np.exp(-k1 * bin_centers) + (1 - frac) * np.exp(-k2 * bin_centers)
        res_double = comp_cdf - fit_double_vals

        # ------------------ BOOTSTRAP ------------------
        bootstrap_k_single, bootstrap_k1, bootstrap_k2, bootstrap_frac = [], [], [], []

        for i in range(n_bootstrap):
            boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)

            hist_b, _ = np.histogram(boot_sample, bins=bin_edges)
            cum_b = np.cumsum(hist_b)
            if cum_b[-1] == 0:
                continue
            comp_cdf_b = 1.0 - cum_b / cum_b[-1]

            try:
                popt_s_b, _ = scipy.optimize.curve_fit(
                    single_exp, bin_centers, comp_cdf_b,
                    p0=p0_single, bounds=bounds_single, maxfev=10000
                )
                bootstrap_k_single.append(popt_s_b[0])
            except Exception:
                continue

            result_b = scipy.optimize.minimize(
                MLE_double, initial_guess,
                args=(boot_sample,), bounds=bounds_double
            )

            if result_b.success:
                k1_b, k2_b, frac_b = result_b.x
                if k1_b < k2_b:
                    k1_b, k2_b = k2_b, k1_b
                    frac_b = 1 - frac_b

                bootstrap_k1.append(k1_b)
                bootstrap_k2.append(k2_b)
                bootstrap_frac.append(frac_b)

                all_bootstrap_data.append({
                    'treatment': treatment,
                    'transition_type': transition,
                    'k_single': popt_s_b[0],
                    'k1_boot': k1_b,
                    'k2_boot': k2_b,
                    'frac_boot': frac_b,
                    'bootstrap_run': i
                })

        # ------------------ SUMMARY ------------------
        summary_rows.append({
            'treatment': treatment,
            'transition_type': transition,
            'k_single_mean': np.mean(bootstrap_k_single) if bootstrap_k_single else np.nan,
            'k_single_std': np.std(bootstrap_k_single, ddof=1) if bootstrap_k_single else np.nan,
            'k1_mean': np.mean(bootstrap_k1) if bootstrap_k1 else np.nan,
            'k1_std': np.std(bootstrap_k1, ddof=1) if bootstrap_k1 else np.nan,
            'k2_mean': np.mean(bootstrap_k2) if bootstrap_k2 else np.nan,
            'k2_std': np.std(bootstrap_k2, ddof=1) if bootstrap_k2 else np.nan,
            'frac_mean': np.mean(bootstrap_frac) if bootstrap_frac else np.nan,
            'frac_std': np.std(bootstrap_frac, ddof=1) if bootstrap_frac else np.nan
        })

        # ------------------ STORE DATA ------------------
        fit_df = pd.DataFrame({
            'x_bins': bin_centers,
            'Cumative_hist': cum_counts / cum_counts[-1],
            'Cumative_hist_inv': comp_cdf,
            'fit_single': fit_single_vals,
            'fit_double': fit_double_vals,
            'fit_pos_single': 1 - fit_single_vals,
            'fit_pos_double': 1 - fit_double_vals,
            'residual_single': res_single,
            'residual_double': res_double,
            'treatment': treatment,
            'transition_type': transition
        })

        data.append(fit_df)

        # ------------------ PLOTTING ------------------

        # Bootstrap distributions
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))

        sns.histplot(bootstrap_k_single, bins=30, kde=True, ax=axs[0], color=col_single_magma)
        sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color=col_fast_crest)
        sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color=col_slow_crest)
        sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[3], color=col_frac_crest)

        axs[0].set_title('k_single')
        axs[1].set_title('k1 (fast)')
        axs[2].set_title('k2 (slow)')
        axs[3].set_title('fraction')

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)

        plt.suptitle(f"{treatment} | {transition}")
        fig.savefig(f"{output_folder}/bootstrap_{treatment}_{transition}.svg", bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # Fit plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].scatter(bin_centers, comp_cdf, color='black', s=15)
        axs[0].plot(bin_centers, fit_single_vals, color=col_single_magma, label='Single')
        axs[0].plot(bin_centers, fit_double_vals, color=col_fast_crest, label='Double')
        axs[0].legend()
        axs[0].set_title("Inverse CDF")

        axs[1].scatter(bin_centers, cum_counts / cum_counts[-1], color='black', s=15)
        axs[1].plot(bin_centers, 1 - fit_single_vals, color=col_single_magma, label='Single')
        axs[1].plot(bin_centers, 1 - fit_double_vals, color=col_fast_crest, label='Double')
        axs[1].legend()
        axs[1].set_title("CDF")

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)

        fig.savefig(f"{output_folder}/fits_{treatment}_{transition}.svg", bbox_inches='tight')
        plt.show()
        plt.close(fig)

        # Residuals
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(bin_centers, res_single, s=12, color=col_single_magma, label='Single')
        ax.scatter(bin_centers, res_double, s=12, color=col_fast_crest, label='Double')
        ax.axhline(0, color='gray', linestyle='--')

        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title("Residuals")

        fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", bbox_inches='tight')
        plt.show()
        plt.close(fig)


        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # ------------------ CDF + INSET RESIDUAL ------------------
        fig, ax = plt.subplots(figsize=(10, 4))  # wide rectangular

        # Main CDF plot
        ax.scatter(
            bin_centers,
            cum_counts / cum_counts[-1],
            color='black',
            s=15,
            label='Data'
        )

        ax.plot(
            bin_centers,
            1 - fit_single_vals,
            color=col_single_magma,
            label='Single'
        )

        ax.plot(
            bin_centers,
            1 - fit_double_vals,
            color=col_fast_crest,
            label='Double'
        )

        ax.set_title(f"CDF – {treatment} {transition}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CDF")
        ax.legend(frameon=False)

        # Clean axes
        ax.spines[['top', 'right']].set_visible(False)

        # ------------------ INSET (Residuals) ------------------
        ax_inset = inset_axes(
            ax,
            width="35%",   # size relative to main plot
            height="45%",
            loc='lower right',
            borderpad=1.5
        )

        ax_inset.scatter(
            bin_centers,
            res_single,
            s=8,
            color=col_single_magma
        )

        ax_inset.scatter(
            bin_centers,
            res_double,
            s=8,
            color=col_fast_crest
        )

        ax_inset.axhline(0, color='gray', linestyle='--', linewidth=1)

        # Minimal inset styling
        ax_inset.set_title("Residuals", fontsize=9)
        ax_inset.tick_params(axis='both', labelsize=8)
        ax_inset.spines[['top', 'right']].set_visible(False)

        # Save + show
        fig.savefig(
            f"{output_folder}/cdf_with_residual_inset_{treatment}_{transition}.svg",
            bbox_inches='tight'
        )

        plt.show()
        plt.close(fig)

    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.DataFrame(summary_rows)
    bootstrap_df = pd.DataFrame(all_bootstrap_data)

    return fits_df, halftime_summary, bootstrap_df
# NOTE: The helper functions (one_phase_association, CDF_mixture, MLE_cdf) 
# must be defined and available globally for this code to run correctly, 
# as they are not included in the definition below, but the logic relies on them.

    
# Add the Single MLE function here:
def MLE_single(params, x):
    """Negative Log Likelihood for Single Exponential (PDF)."""
    K = params[0]
    # Enforce K > 0 constraint
    if K <= 1e-5:
        return np.inf 
        
    pdf_vals = K * np.exp(-K * x)
    pdf_vals = np.clip(pdf_vals, 1e-12, None)
    return -np.sum(np.log(pdf_vals))

# ... (End of Global Definitions) ...

# NOTE: The MLE_single, one_phase_association, and func (defaulting to one_phase_association)
# must be defined or imported outside this function, but the core logic is now 
# defined within or assumed to be available. 
# CDF_mixture and MLE_cdf are defined inside the 'both' branch for scope clarity.

def cumulative_residence_fitting_deletemayube(dfs, output_folder, bin_width, xlim, func='one_phase_association'):
    """
    Fits cumulative histogram data with one-phase association (Least Squares/MLE) or bi-exponential (MLE).
    Includes Model Selection (AIC, BIC, LRT) and comprehensive bootstrap analysis for the bi-exponential case.
    
    Args:
        dfs (df): dataframe containing raw data for fitting.
        output_folder (str): where to save data and plots.
        bin_width (float): bin_width used to calculate the CDF for plotting/R2.
        xlim (float): max time value for binning and plotting.
        func (str/callable): 'both' for single-MLE and double-MLE fits/comparison, 
                             or a callable function (like one_phase_association) for LS fit.
    Returns:
        df: returns the fits, summary data, and raw bootstrap results.
    """
    data = []
    summary = []
    all_bootstrap_data = [] 
    n_bootstrap = 500
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- NLL Definition for MLE_single (Assumed to exist, defined here for MLE consistency) ---
    def MLE_single(k, x):
        if k <= 1e-4:
            return np.inf
        # PDF for single exponential decay: f(t) = k * exp(-k*t)
        pdf = k * np.exp(-k * x)
        pdf = np.clip(pdf, 1e-12, None)
        return -np.sum(np.log(pdf))

    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        print(f"Processing: {treatment} - {transition}")
        
        dwell_times_raw = df["CumulativeTime(s)"].values
        dwell_times = dwell_times_raw[dwell_times_raw > 0] # Filter zeros for MLE (t > 0)
        n_value = len(dwell_times)

        if n_value < 10:
             print(f"  -> WARNING: Insufficient data (N={n_value}) in group: {treatment} {transition}")
             continue

        # Binning for CDF visualization / R2 calculation
        bin_edges = np.arange(0, xlim + 1, bin_width)
        hist, _ = np.histogram(dwell_times_raw, bins=bin_edges)
        cumulative_counts = np.cumsum(hist)
        
        if cumulative_counts.max() == 0:
            print(f"  -> WARNING: No data in this group: {treatment} {transition}")
            continue
            
        cumulative_counts_array = cumulative_counts.values / cumulative_counts.max()
        bin_edges_fit = bin_edges[:-1]
        bin_centers = bin_edges_fit + bin_width / 2

        # Initialize LL and fit parameters for both models
        LL1, LL2 = np.nan, np.nan
        K_single_MLE = np.nan # K rate from single MLE
        
        # Initialize single fit reporting (used for both LS and MLE)
        half_time, se_half_time, rSquared = np.nan, np.nan, np.nan
        res_single = np.full_like(cumulative_counts_array, np.nan)
        Y0, Plateau = 0, 1 # Assume fixed for one-phase association (CDF)

        # Initialize double fit reporting
        k1, k2, frac = np.nan, np.nan, np.nan
        frac_std_error, k1_std_error, k2_std_error = np.nan, np.nan, np.nan
        r_squared_double = np.nan
        p_boot_lrt = np.nan # LRT p-value

        # ---------------------------------------------------------------------
        # 2. FITTING LOGIC (MLE for 'both' or LS for single fit)
        # ---------------------------------------------------------------------
        
        if func == 'both':
            # --- Definitions for MLE (Bi-Exponential) ---
            def CDF_mixture(x, k1, k2, frac):
                return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))
                
            def MLE_cdf(params, x):
                k1, k2, frac = params
                if not (0.01 <= frac <= 0.99): 
                    return np.inf
                if k1 <= 1e-4 or k2 <= 1e-4: 
                    return np.inf
                # PDF for bi-exponential: f(t) = frac*k1*exp(-k1*t) + (1-frac)*k2*exp(-k2*t)
                pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
                pdf_vals = np.clip(pdf_vals, 1e-12, None)
                return -np.sum(np.log(pdf_vals))
            # --- End Definitions ---

            # --- 2a. SINGLE-EXPONENTIAL FIT (MLE) ---
            initial_guess_s = [1/100] 
            bounds_s = [(1e-5, 10)]
            
            try:
                # 1. Initial MLE Fit (for K and LL1)
                result_single_mle = scipy.optimize.minimize(
                    MLE_single, initial_guess_s, args=(dwell_times,), bounds=bounds_s
                )
                K_single_MLE = result_single_mle.x[0]
                LL1 = -result_single_mle.fun # <--- CAPTURE LL1
                half_time = np.log(2) / K_single_MLE
                
                # 2. R2 Calculation (for CDF comparison, assuming one_phase_association is available)
                # Assumed form: one_phase_association(x, Y0, Plateau, K)
                # We use K_single_MLE as K, Y0=0, Plateau=1
                fitted_data = one_phase_association(bin_edges_fit, Y0, Plateau, K_single_MLE)
                ss_res_s = np.sum((cumulative_counts_array - fitted_data) ** 2)
                ss_tot_s = np.sum((cumulative_counts_array - np.mean(cumulative_counts_array)) ** 2)
                rSquared = 1 - (ss_res_s / ss_tot_s)
                
                # 3. Store Data and Residuals
                res_single = cumulative_counts_array - fitted_data
                test_single = pd.DataFrame({
                    'Cumative_hist_sing': cumulative_counts_array,
                    'fit_sing': fitted_data,
                    'x_bins_sing': bin_edges_fit,
                    'residuals_sing': res_single,
                    'treatment': treatment,
                    'transition_type': transition
                })
                data.append(test_single)
            except Exception as e:
                print(f"  -> Single MLE fit failed: {e}")
                LL1 = np.nan
            
            # --- 2b. BI-EXPONENTIAL FIT (MLE) ---
            initial_guess = [1/20, 1/300, 0.5]
            bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
            
            result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
            
            k1_raw, k2_raw, frac_raw = result.x
            LL2 = -result.fun # <--- CAPTURE LL2
            
            # Enforce k1 is the FAST rate (k1 > k2)
            if k1_raw < k2_raw:
                k1, k2 = k2_raw, k1_raw
                frac = 1 - frac_raw
            else:
                k1, k2, frac = k1_raw, k2_raw, frac_raw

            print(f"Fitted: k1={k1:.4f}, k2={k2:.4f}, frac={frac:.4f}")
            
            # --- 2c. BOOTSTRAP (SEM & LRT) ---
            bootstrap_frac, bootstrap_k1, bootstrap_k2, LR_null = [], [], [], []
            
            for i in range(n_bootstrap):
                boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
                
                # Double Fit Bootstrap (for SEM)
                result_boot = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(boot_sample,), bounds=bounds)
                if result_boot.success:
                    k1_boot_raw, k2_boot_raw, frac_boot_raw = result_boot.x
                    
                    # Enforce k1 is the FAST rate on bootstrap results
                    if k1_boot_raw < k2_boot_raw:
                        k1_boot, k2_boot = k2_boot_raw, k1_boot_raw
                        frac_boot = 1 - frac_boot_raw
                    else:
                        k1_boot, k2_boot, frac_boot = k1_boot_raw, k2_boot_raw, frac_boot_raw
                            
                    bootstrap_frac.append(frac_boot)
                    bootstrap_k1.append(k1_boot)
                    bootstrap_k2.append(k2_boot)
                    
                    # Store raw bootstrap data
                    all_bootstrap_data.append({
                        'treatment': treatment, 'transition_type': transition, 'k1_boot': k1_boot,
                        'k2_boot': k2_boot, 'frac_boot': frac_boot, 'bootstrap_run': i
                    })
                    
                # LRT Bootstrap (Simulate from fitted single exp - Null Hypothesis)
                if not np.isnan(K_single_MLE):
                    sim = np.random.exponential(scale=1/K_single_MLE, size=n_value)
                    
                    rs_sim = scipy.optimize.minimize(MLE_single, initial_guess_s, args=(sim,), bounds=bounds_s)
                    LL1_sim = -rs_sim.fun if rs_sim.success else np.nan
                    
                    rd_sim = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(sim,), bounds=bounds)
                    LL2_sim = -rd_sim.fun if rd_sim.success else np.nan
                    
                    if rs_sim.success and rd_sim.success and not np.isnan(LL1_sim) and not np.isnan(LL2_sim):
                        LR_null.append(2*(LL2_sim - LL1_sim))

            bootstrap_frac = np.array(bootstrap_frac)
            bootstrap_k1 = np.array(bootstrap_k1)
            bootstrap_k2 = np.array(bootstrap_k2)
            
            # Calculate SEM and CI
            def bootstrap_summary(values):
                mean = np.mean(values)
                std_error = np.std(values, ddof=1)
                return mean, std_error

            # Note: We calculate means/SEs but use the original MLE fit (k1, k2, frac) for the final summary table
            frac_mean_bootstrap, frac_std_error = bootstrap_summary(bootstrap_frac)
            k1_mean_bootstrap, k1_std_error = bootstrap_summary(bootstrap_k1)
            k2_mean_bootstrap, k2_std_error = bootstrap_summary(bootstrap_k2)
            
            # Calculate LRT p-value
            if len(LR_null) > 0 and not np.isnan(LL1) and not np.isnan(LL2):
                LR_obs = 2*(LL2 - LL1)
                p_boot_lrt = np.mean(np.array(LR_null)[np.array(LR_null) > 0] >= LR_obs)
            
            # R² of double fit (using binned CDF)
            fitted_cdf = CDF_mixture(bin_centers, k1, k2, frac)
            ss_res = np.sum((cumulative_counts_array - fitted_cdf) ** 2)
            ss_tot = np.sum((cumulative_counts_array - np.mean(cumulative_counts_array)) ** 2)
            r_squared_double = 1 - ss_res / ss_tot
            
            # Half-times
            half_time_fast = np.log(2) / k1
            half_time_slow = np.log(2) / k2
            
            # Store double fit curve data
            fit_df = pd.DataFrame({
                'x_bins': bin_centers,
                'fit': fitted_cdf,
                'Cumative_hist': cumulative_counts_array,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(fit_df)

            # --- PLOTS (Residuals and Bootstrap Diagnostics) ---
            # 1. Residuals Plot
            res = cumulative_counts_array - fitted_cdf
            fig = plt.figure(figsize=(6, 2))
            plt.scatter(bin_centers, res_single, label='1-exp (MLE)', s=8, c='#f05423')
            plt.scatter(bin_centers, res, label='2-exp (MLE)', s=8, c='#00aeef')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"Residuals for {treatment} {transition} ")
            plt.xlabel("Dwell time (s)")
            plt.ylabel("Residual (CDF)")
            plt.xlim(0, 100)
            plt.ylim(-0.25, 0.25)
            plt.legend()
            plt.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=300)
            plt.close(fig)
            
            # 2. Bootstrap Distributions (k1, k2, frac)
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[0], color="#939990FF", edgecolor='black')
            axs[0].axvline(frac, color='black', linestyle='--', label='Final Fit')
            axs[0].set_title(f'Bootstrap Distribution of frac ({treatment} {transition})')
            axs[0].set_xlabel('frac')
            axs[0].legend()
            sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="#386B20FF", edgecolor='black')
            axs[1].axvline(k1, color='black', linestyle='--', label='Final Fit')
            axs[1].set_title('Bootstrap Distribution of k1 (FAST)')
            axs[1].set_xlabel('k1 (1/s)')
            axs[1].legend()
            sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="#81b868df", edgecolor='black')
            axs[2].axvline(k2, color='black', linestyle='--', label='Final Fit')
            axs[2].set_title('Bootstrap Distribution of k2 (SLOW)')
            axs[2].set_xlabel('k2 (1/s)')
            axs[2].legend()
            plt.tight_layout()
            plt.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=300) 
            plt.close(fig)
            
            # 3. k1 vs k2 Scatter Plot
            df_boot_group = pd.DataFrame({'k1': bootstrap_k1, 'k2': bootstrap_k2})
            fig = plt.figure(figsize=(6, 5))
            plt.scatter(df_boot_group['k2'], df_boot_group['k1'], alpha=0.5, s=15, color='#00aeef')
            plt.scatter(k2, k1, color='red', s=50, marker='x', label='Final Fit')
            plt.title(f'k1 (FAST) vs k2 (SLOW) Bootstrap ({treatment} {transition})')
            plt.xlabel('k2 (SLOW) [1/s]')
            plt.ylabel('k1 (FAST) [1/s]')
            plt.legend()
            plt.savefig(f"{output_folder}/k1_vs_k2_scatter_{treatment}_{transition}.svg", dpi=300)
            plt.close(fig)

        else: # Original Least Squares (LS) fit logic (for backward compatibility if func is a callable)
            # This branch uses the original LS method, so no MLE/AIC/BIC/LRT calculated here.
            
            # perform the fit using the passed function (func)
            p0 = (0.1, 1, 0.1) # Adjusted P0 for one-phase_association function form
            try:
                params, cv = scipy.optimize.curve_fit(func, bin_edges_fit, cumulative_counts_array, p0)
                Y0, Plateau, K_ls = params
            except RuntimeError:
                 print(f"  -> LS fit failed for {treatment} {transition}")
                 continue # Skip to next group if fit fails
                 
            tauSec = (1 / K_ls)
            half_time = np.log(2)/K_ls

            # Calculate standard errors from the covariance matrix
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K_ls**2
            alpha = 0.05 
            z_score = scipy.stats.norm.ppf(1 - alpha / 2) 
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            
            # determine quality of the fit
            rSquared = 1 - np.sum(np.square(cumulative_counts_array - func(bin_edges_fit, Y0, Plateau, K_ls))) / np.sum(np.square(cumulative_counts_array - np.mean(cumulative_counts_array)))

            fitted_data = func(bin_edges_fit, Y0, Plateau, K_ls)
            test = pd.DataFrame({
                'Cumative_hist': cumulative_counts_array,
                'fit': fitted_data,
                'x_bins': bin_edges_fit,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(test)
            K_single_MLE = K_ls # Use LS K for the summary placeholder

        # ---------------------------------------------------------------------
        # 3. MODEL SELECTION & SUMMARY (Executed only if func='both')
        # ---------------------------------------------------------------------
        
        AIC1, AIC2, BIC1, BIC2, LR_obs = np.nan, np.nan, np.nan, np.nan, np.nan
        prefer_AIC, prefer_BIC = np.nan, np.nan
        
        if func == 'both' and not np.isnan(LL1) and not np.isnan(LL2):
            k_single_param = 1 
            k_double_param = 3
            
            AIC1 = 2*k_single_param - 2*LL1
            AIC2 = 2*k_double_param - 2*LL2
            BIC1 = k_single_param*np.log(n_value) - 2*LL1
            BIC2 = k_double_param*np.log(n_value) - 2*LL2
            
            LR_obs = 2*(LL2 - LL1)
            
            prefer_AIC = "2-exp" if AIC2 < AIC1 else "1-exp"
            prefer_BIC = "2-exp" if BIC2 < BIC1 else "1-exp"

        # --- Summary DataFrame Creation (Updated Columns) ---
        col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K_single_MLE, rSquared,
                                k1, k2, frac, frac_std_error, r_squared_double,
                                k1_std_error, k2_std_error, LL1, LL2, AIC1, AIC2, BIC1, BIC2, LR_obs, p_boot_lrt,
                                prefer_AIC, prefer_BIC]],
            columns=['mean_halftime_single', 'sem_halftime_single', 'n', 'treatment', 'transition', 'K_single', 'R2_single_CDF',
                     'k1_fast', 'k2_slow', 'frac_fast', 'frac_std_error', 'R2_double_CDF',
                     'k1_std_error', 'k2_std_error', 'LL1', 'LL2', 'AIC1', 'AIC2', 'BIC1', 'BIC2', 'LR_obs', 'p_boot_LRT',
                     'prefer_AIC', 'prefer_BIC'])
        summary.append(col_df)

    fits_df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    halftime_summary = pd.concat(summary, ignore_index=True) if summary else pd.DataFrame()
    bootstrap_df = pd.DataFrame(all_bootstrap_data)

    warnings.filterwarnings('default', category=RuntimeWarning)
    
    return fits_df, halftime_summary, bootstrap_df

def cumulative_residence_fitting_oldchatslop_geminiwithscatter(dfs, output_folder, bin_width, xlim, func='both'):
    """
    Fits cumulative histogram data with one-phase association (MLE) or bi-exponential (MLE).
    All fits now use MLE on raw dwell times with bootstrapping for Standard Error estimation.
    """
    
    # Ensure imports are defined globally: import os, warnings, pandas, numpy, scipy, matplotlib, seaborn
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = []
    summary = []
    all_bootstrap_data = [] 
    

    # Suppress warnings that often occur during bootstrap fitting convergence
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        print(f"Processing: {treatment} - {transition}")
        
        # --- 1. Data Preparation (Raw and Binned) ---
        dwell_times_raw = df["CumulativeTime(s)"].values
        # Filter zeros for MLE (MLE must use P(t|t>0))
        dwell_times = dwell_times_raw[dwell_times_raw > 0]
        
        # Calculate N_VALUE (Total number of events, used in summary)
        n_value = len(dwell_times)
        
        # Binning for CDF visualization / R2 calculation
        bin_edges = np.arange(0, xlim + 1, bin_width)
        hist, _ = np.histogram(dwell_times_raw, bins=bin_edges)
        cumulative_counts = np.cumsum(hist)
        
        # Normalize (Safety check: avoid divide by zero if empty)
        if cumulative_counts.max() > 0:
            cumulative_counts_norm = cumulative_counts / cumulative_counts.max()
        else:
            print(f"  -> WARNING: No data in this group: {treatment} {transition}")
            continue
            
        bin_edges_fit = bin_edges[:-1]
        bin_centers = bin_edges_fit + bin_width / 2

        # ---------------------------------------------------------
        # 2. Single-Exponential Fit (MLE on Raw Data with Bootstrap)
        # ---------------------------------------------------------
        print('Single-fit (MLE) first')
        
        # Initial guess and bounds for K (rate constant)
        initial_guess_s = [1/100] 
        bounds_s = [(1e-5, 10)]

        # Placeholder initial values in case of failure
        K, half_time, se_half_time, rSquared = np.nan, np.nan, np.nan, np.nan
        res_single = np.full_like(cumulative_counts_norm, np.nan)
        Y0, Plateau = 0, 1 # Fixed for normalized CDF R2 calculation

        try:
            # --- 2a. Initial MLE Fit (for the central K value) ---
            result_single_mle = scipy.optimize.minimize(
                MLE_single, initial_guess_s, args=(dwell_times,), bounds=bounds_s
            )
            K = result_single_mle.x[0]
            half_time = np.log(2) / K
            
            # --- 2b. Single Fit BOOTSTRAP (for Standard Error) ---
            n_bootstrap = 500
            bootstrap_K = []
            bootstrap_half_time = []
            
            for i in range(n_bootstrap):
                boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
                result_boot = scipy.optimize.minimize(
                    MLE_single, initial_guess_s, args=(boot_sample,), bounds=bounds_s
                )
                if result_boot.success:
                    K_boot = result_boot.x[0]
                    bootstrap_K.append(K_boot)
                    bootstrap_half_time.append(np.log(2) / K_boot)

            bootstrap_half_time = np.array(bootstrap_half_time)
            
            # Calculate SEM (Standard Error of the Mean) from the bootstrapped half-times
            se_half_time = np.std(bootstrap_half_time, ddof=1)
            print(f"  -> Single Fit Bootstrap SE Half-time: {se_half_time:.4f}")
            
            # --- 2c. R2 Calculation & Storing ---
            fitted_data = one_phase_association(bin_edges_fit, Y0, Plateau, K)
            ss_res_s = np.sum((cumulative_counts_norm - fitted_data) ** 2)
            ss_tot_s = np.sum((cumulative_counts_norm - np.mean(cumulative_counts_norm)) ** 2)
            rSquared = 1 - (ss_res_s / ss_tot_s)
            

            # --- AIC, BIC for Single-Exponential ---
            logL_single = -MLE_single([K], dwell_times)
            k_params_single = 1  # only K
            AIC_single = 2*k_params_single - 2*logL_single
            BIC_single = k_params_single * np.log(len(dwell_times)) - 2*logL_single


            # Store Single Fit Data
            res_single = cumulative_counts_norm - fitted_data
            test_single = pd.DataFrame({
                'Cumative_hist_sing': cumulative_counts_norm,
                'fit_sing': fitted_data,
                'x_bins_sing': bin_edges_fit,
                'residuals_sing': res_single,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(test_single)

        except Exception as e:
            print(f"  -> Single MLE fit failed for {treatment} {transition}: {e}")

        if func != 'both':
            continue

        # ---------------------------------------------------------
        # 3. Bi-Exponential Fit (MLE on Raw Data with Bootstrap)
        # ---------------------------------------------------------
        print(f"MLE fitting for {treatment} {transition}")
        
        initial_guess = [1/20, 1/300, 0.5]
        bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
        
        result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
        k1, k2, frac = result.x
        
        # Enforce k1 is the FAST rate (k1 > k2)
        if k1 < k2:
            k1, k2 = k2, k1
            frac = 1 - frac

        print(f"Fitted: k1={k1:.4f}, k2={k2:.4f}, frac={frac:.4f}")
        # ----------- BOOTSTRAP TO ESTIMATE STANDARD ERRORS -----------
        n_bootstrap = 500
        bootstrap_frac = []
        bootstrap_k1 = []
        bootstrap_k2 = []

        for i in range(n_bootstrap):
            boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
            result_boot = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(boot_sample,), bounds=bounds)
            if result_boot.success:
                k1_boot_raw, k2_boot_raw, frac_boot_raw = result_boot.x
                
                # Enforce k1 is the FAST rate on bootstrap results
                if k1_boot_raw < k2_boot_raw:
                    k1_boot, k2_boot = k2_boot_raw, k1_boot_raw
                    frac_boot = 1 - frac_boot_raw
                else:
                    k1_boot, k2_boot, frac_boot = k1_boot_raw, k2_boot_raw, frac_boot_raw
                        
                bootstrap_frac.append(frac_boot)
                bootstrap_k1.append(k1_boot)
                bootstrap_k2.append(k2_boot)
                
                all_bootstrap_data.append({
                    'treatment': treatment,
                    'transition_type': transition,
                    'k1_boot': k1_boot,
                    'k2_boot': k2_boot,
                    'frac_boot': frac_boot,
                    'bootstrap_run': i
                })
                        
        bootstrap_frac = np.array(bootstrap_frac)
        bootstrap_k1 = np.array(bootstrap_k1)
        bootstrap_k2 = np.array(bootstrap_k2)
        
        # Calculate bootstrap statistics (Mean, SEM, CI)
        def bootstrap_summary(values, name):
            mean = np.mean(values)
            std_error = np.std(values, ddof=1)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            print(f"Bootstrap {name}: mean={mean:.4f}, std error={std_error:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
            return mean, std_error, ci_lower, ci_upper
            
        frac_mean_bootstrap, frac_std_error, frac_ci_lower, frac_ci_upper = bootstrap_summary(bootstrap_frac, 'frac')
        k1_mean_bootstrap, k1_std_error, k1_ci_lower, k1_ci_upper = bootstrap_summary(bootstrap_k1, 'k1')
        k2_mean_bootstrap, k2_std_error, k2_ci_lower, k2_ci_upper = bootstrap_summary(bootstrap_k2, 'k2')

        # --- Calculate Double Fit R2 (using binned CDF for comparison) ---
        fitted_cdf = CDF_mixture(bin_edges_fit, k1, k2, frac)
        ss_res = np.sum((cumulative_counts_norm - fitted_cdf) ** 2)
        ss_tot = np.sum((cumulative_counts_norm - np.mean(cumulative_counts_norm)) ** 2)
        r_squared_double = 1 - ss_res / ss_tot
        print(f'{treatment} Double Fit R2 is {r_squared_double}')
        # --- AIC, BIC for Double-Exponential ---
        logL_double = -MLE_cdf([k1, k2, frac], dwell_times)
        k_params_double = 3
        AIC_double = 2*k_params_double - 2*logL_double
        BIC_double = k_params_double * np.log(len(dwell_times)) - 2*logL_double

        # --- LLR Test (Single vs Double) ---
        LLR = 2 * (logL_double - logL_single)
        df_llr = k_params_double - k_params_single  # = 2
        p_value_llr = 1 - scipy.stats.chi2.cdf(LLR, df_llr)

        # --- Residuals and Plotting ---
        res = cumulative_counts_norm - fitted_cdf
        fig = plt.figure(figsize=(6, 2))
        plt.scatter(bin_centers, res_single, label='single_fit (MLE K)', s=8, c='#f05423')
        plt.scatter(bin_centers, res, label='double_fit (MLE)', s=8, c='#00aeef')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Residuals for {treatment} {transition} ")
        plt.xlabel("Resident time (s)")
        plt.ylabel("Residual")
        plt.xlim(0, xlim)
        plt.ylim(-0.25, 0.25)
        plt.legend()
        plt.show()
        fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=300)
        plt.close(fig)

        # Half-times (k1 is fast, k2 is slow)
        half_time_fast = np.log(2) / k1
        half_time_slow = np.log(2) / k2
        
        # Store fit curve data
        fit_df = pd.DataFrame({
            'x_bins': bin_centers,
            'fit': fitted_cdf,
            'Cumative_hist': cumulative_counts_norm,
            'treatment': treatment,
            'transition_type': transition
        })
        data.append(fit_df)
        
        # --- Summary DataFrame Creation (20 Columns) ---
        col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K, rSquared,
                                 half_time_fast, half_time_slow, k1, k2, frac, frac_std_error, r_squared_double,
                                 k1_std_error, k2_std_error,bootstrap_frac.mean(), bootstrap_k1.mean(), bootstrap_k2.mean(),AIC_single, BIC_single, AIC_double, BIC_double, LLR, p_value_llr]],
                     columns=['mean_halftime_single', 'sem_halftime_single', 'n', 'treatment', 'transition', 'K_single_MLE', 'R2_single_CDF',
                             'half_time_fast', 'half_time_slow', 'k1_fast', 'k2_slow', 'frac_fast', 'frac_std_error', 'R2_double_CDF',
                             'k1_std_error', 'k2_std_error', 'bootstrap_frac_mean', 'bootstrap_k1_mean', 'bootstrap_k2_mean','AIC_single', 'BIC_single', 'AIC_double', 'BIC_double', 'LLR_test', 'LLR_pvalue'])
        summary.append(col_df)

        # --- Bootstrap Histograms Plot ---
        fig_hist, axs = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[0], color="#939990FF", edgecolor='black').axvline(frac, color='black', linestyle='--', label='Original Fit')
        axs[0].set_title('Bootstrap Distribution of frac'); axs[0].set_xlabel('frac'); axs[0].legend()
        sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="#386B20FF", edgecolor='black').axvline(k1, color='black', linestyle='--', label='Original Fit')
        axs[1].set_title('Bootstrap Distribution of k1 (FAST)'); axs[1].set_xlabel('k1 (1/s)'); axs[1].legend()
        sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="#81b868df", edgecolor='black').axvline(k2, color='black', linestyle='--', label='Original Fit')
        axs[2].set_title('Bootstrap Distribution of k2 (SLOW)'); axs[2].set_xlabel('k2 (1/s)'); axs[2].legend()
        plt.tight_layout()
        fig_hist.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=300) 
        plt.show(fig_hist)
        
        # --- Diagnostic Scatter Plot (k1 vs k2) ---
        fig_scatter, ax_scatter = plt.subplots(figsize=(5, 5))
        
        ax_scatter.scatter(bootstrap_k2, bootstrap_k1, alpha=0.5, s=15, color='#00a18f')
        
        max_k = max(bootstrap_k1.max(), bootstrap_k2.max()) if bootstrap_k1.size > 0 else 1
        min_k = min(bootstrap_k1.min(), bootstrap_k2.min()) if bootstrap_k1.size > 0 else 0
        ax_scatter.plot([min_k, max_k], [min_k, max_k], 'k--', alpha=0.7, label='$k_1 = k_2$')

        ax_scatter.set_xlabel('k2 (SLOW, 1/s)')
        ax_scatter.set_ylabel('k1 (FAST, 1/s)')
        ax_scatter.set_title(f'Bootstrap $k_1$ vs $k_2$ ({treatment} {transition})')
        ax_scatter.legend()
        plt.tight_layout()
        fig_scatter.savefig(f"{output_folder}/bootstrap_scatter_{treatment}_{transition}.svg", dpi=300)
        plt.show(fig_scatter)
        # --- End Diagnostic Scatter Plot ---


    fits_df = pd.concat(data, ignore_index=True) if data else pd.DataFrame()
    halftime_summary = pd.concat(summary, ignore_index=True) if summary else pd.DataFrame()
    bootstrap_df = pd.DataFrame(all_bootstrap_data)

    # Re-enable warnings
    warnings.filterwarnings('default', category=RuntimeWarning)
    
    return fits_df, halftime_summary, bootstrap_df

##### actual code

def cumulative_residence_fitting3(dfs, output_folder, bin_width, xlim, func='both'):
    """
    Fits cumulative histogram data.
    
    Changes implemented:
    1. Returns 'bootstrap_df' with individual results from every bootstrap run.
    2. Enforces the convention: k1 is the FAST rate (k1 > k2).
    """
    data = []
    summary = []
    # NEW: List to collect all individual bootstrap results
    all_bootstrap_data = [] 

    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        print(treatment)
        print(transition)
        
        # --- Common Binning ---
        bin_edges = range(0, xlim + 1, bin_width)
        bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False)
        bin_counts = bins.value_counts().sort_index()
        cumulative_counts = bin_counts.cumsum()
        bin_edges_array = bin_edges[:-1]
        cumulative_counts_array = cumulative_counts.values
        cumulative_counts_array = cumulative_counts_array / cumulative_counts_array.max()

        if func == 'both':
            print('single-fit first')
            
            # --- Single-Exponential Fit (via curve_fit on CDF) ---
            try:
                # Assuming one_phase_association takes (x, Y0, Plateau, K)
                p0 = (0, 1, 0.1) # Adjusted p0 for normalized CDF
                params, cv = scipy.optimize.curve_fit(one_phase_association, bin_edges_array, cumulative_counts_array, p0)
                Y0, Plateau, K = params
                tauSec = (1 / K)
                half_time = np.log(2) / K
                se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2
                alpha = 0.05
                z_score = scipy.stats.norm.ppf(1 - alpha / 2)
                ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
                n_value = cumulative_counts.max()
                
                # R² of single fit
                squaredDiffs = np.square(cumulative_counts_array - one_phase_association(bin_edges_array, Y0, Plateau, K))
                squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
                rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

                fitted_data = one_phase_association(bin_edges_array, Y0, Plateau, K)
                fitted_data_df = pd.DataFrame(fitted_data)
                res_single = cumulative_counts_array - fitted_data
                x_bins = pd.DataFrame(bin_edges_array)
                test_single = pd.DataFrame(cumulative_counts_array)
                test_single = pd.concat([test_single, fitted_data_df, x_bins, pd.DataFrame(res_single)], axis=1)
                test_single.columns = ['Cumative_hist_sing', 'fit_sing', 'x_bins_sing', 'residuals_sing']
                test_single['treatment'] = treatment
                test_single['transition_type'] = transition
                data.append(test_single)
            except RuntimeError:
                K, half_time, se_half_time, rSquared = np.nan, np.nan, np.nan, np.nan
                n_value = cumulative_counts.max()
            
            print(f"MLE (CDF) fitting for {treatment} {transition}")
            
            # --- Bi-Exponential MLE Fit ---
            dwell_times = df["CumulativeTime(s)"].values
            dwell_times = dwell_times[dwell_times > 0]
            
            # Definitions of CDF_mixture and MLE_cdf (REQUIRED to fix NameError)
            def CDF_mixture(x, k1, k2, frac):
                return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))
                
            def MLE_cdf(params, x):
                k1, k2, frac = params
                if not (0.01 <= frac <= 0.99): 
                    return np.inf
                if k1 <= 1e-4 or k2 <= 1e-4: 
                    return np.inf
                pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
                pdf_vals = np.clip(pdf_vals, 1e-12, None)
                return -np.sum(np.log(pdf_vals))

            initial_guess = [1/20, 1/300, 0.5]
            bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
            
            result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
            k1, k2, frac = result.x
            
            if k1 < k2:
                k1, k2 = k2, k1
                frac = 1 - frac
                
            print(f"Fitted: k1={k1:.4f}, k2={k2:.4f}, frac={frac:.4f}")
            
            # ----------- BOOTSTRAP TO ESTIMATE STANDARD ERRORS -----------
            n_bootstrap = 500
            bootstrap_frac = []
            bootstrap_k1 = []
            bootstrap_k2 = []
            
            for i in range(n_bootstrap):
                boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
                result_boot = scipy.optimize.minimize(MLE_cdf, [k1, k2, frac], args=(boot_sample,), bounds=bounds) 
                
                if result_boot.success:
                    k1_boot_raw, k2_boot_raw, frac_boot_raw = result_boot.x
                    
                    # Enforce k1 > k2 on bootstrap results
                    if k1_boot_raw < k2_boot_raw:
                        k1_boot, k2_boot = k2_boot_raw, k1_boot_raw
                        frac_boot = 1 - frac_boot_raw
                    else:
                        k1_boot, k2_boot, frac_boot = k1_boot_raw, k2_boot_raw, frac_boot_raw
                        
                    bootstrap_frac.append(frac_boot)
                    bootstrap_k1.append(k1_boot)
                    bootstrap_k2.append(k2_boot)
                    
                    all_bootstrap_data.append({
                        'treatment': treatment,
                        'transition_type': transition,
                        'k1_boot': k1_boot,
                        'k2_boot': k2_boot,
                        'frac_boot': frac_boot,
                        'bootstrap_run': i
                    })

            # Calculate bootstrap statistics
            def bootstrap_summary(values, name):
                mean = np.mean(values)
                std_error = np.std(values, ddof=1)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                return mean, std_error, ci_lower, ci_upper
                
            frac_mean_bootstrap, frac_std_error, frac_ci_lower, frac_ci_upper = bootstrap_summary(bootstrap_frac, 'frac')
            k1_mean_bootstrap, k1_std_error, k1_ci_lower, k1_ci_upper = bootstrap_summary(bootstrap_k1, 'k1')
            k2_mean_bootstrap, k2_std_error, k2_ci_lower, k2_ci_upper = bootstrap_summary(bootstrap_k2, 'k2')
            
            # ----------- END BOOTSTRAP -----------
            
            # --- Double-Exponential CDF R2 and Plotting Data ---
            bin_centers = bin_edges_array 
            hist, _ = np.histogram(dwell_times, bins=bin_edges)
            cum_counts = np.cumsum(hist)
            cum_norm = cum_counts / cum_counts[-1]
            fitted_cdf = CDF_mixture(bin_centers, k1, k2, frac)
            
            # R² of double fit
            ss_res = np.sum((cum_norm - fitted_cdf) ** 2)
            ss_tot = np.sum((cum_norm - np.mean(cum_norm)) ** 2)
            r_squared_double = 1 - ss_res / ss_tot
            
            # Half-times (k1 is fast, k2 is slow)
            half_time_fast = np.log(2) / k1 
            half_time_slow = np.log(2) / k2 
            
            # Store double-fit data (for plotting/residuals)
            fit_df = pd.DataFrame({
                'x_bins': bin_centers,
                'fit': fitted_cdf,
                'Cumative_hist': cum_norm,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(fit_df)
            
            # Store summary row (MLE results + Bootstrap SEMs)
            col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K, rSquared,
                                    half_time_fast, half_time_slow, k1, k2, frac, frac_std_error, r_squared_double,
                                    k1_std_error, k2_std_error]],
                        columns=['mean', 'sem', 'n', 'treatment', 'transition', 'K', 'r_squared',
                                 'half_time_fast', 'half_time_slow', 'k1', 'k2', 'frac_fast', 'frac_std_error', 'R2',
                                 'k1_std_error', 'k2_std_error'])
            summary.append(col_df)
            
            # Residuals plot
            res = cum_norm - fitted_cdf
            fig = plt.figure(figsize=(6, 2))
            plt.scatter(bin_centers, res_single, label='single_fit', s=8, c='#f05423')
            plt.scatter(bin_centers, res, label='double_fit', s=8, c='#00aeef')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"Residuals for {treatment} {transition} ")
            plt.xlabel("Resident time (s)")
            plt.ylabel("Residual")
            plt.xlim(0, 100)
            plt.ylim(-0.25, 0.25)
            plt.legend()
            plt.show()
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=300)
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=600)
            # Half-times
            half_time_fast = np.log(2) / k1
            half_time_slow = np.log(2) / k2
            fit_df = pd.DataFrame({
                'x_bins': bin_centers,
                'fit': fitted_cdf,
                'Cumative_hist': cum_norm,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(fit_df)
            col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K, rSquared,
                        half_time_fast, half_time_slow, k1, k2, frac, frac_std_error, r_squared_double,
                        k1_std_error, k2_std_error,bootstrap_frac.mean(), bootstrap_k1.mean(), bootstrap_k2.mean()]],
                    columns=['mean', 'sem', 'n', 'treatment', 'transition', 'K', 'r_squared',
                            'half_time_fast', 'half_time_slow', 'k1', 'k2', 'frac_fast', 'frac_std_error', 'R2',
                            'k1_std_error', 'k2_std_error', 'bootstrap_frac', 'bootstrap_k1', 'bootstrap_k2'])
            summary.append(col_df)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            # Bootstrap plot for frac
            sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[0], color="#939990FF", edgecolor='black')
            axs[0].axvline(frac, color='black', linestyle='--', label='Original Fit')
            axs[0].set_title('Bootstrap Distribution of frac')
            axs[0].set_xlabel('frac')
            axs[0].legend()
            # Bootstrap plot for k1
            sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="#386B20FF", edgecolor='black')
            axs[1].axvline(k1, color='black', linestyle='--', label='Original Fit')
            axs[1].set_title('Bootstrap Distribution of k1')
            axs[1].set_xlabel('k1 (1/s)')
            axs[1].legend()
            # Bootstrap plot for k2
            sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="#81b868df", edgecolor='black')
            axs[2].axvline(k2, color='black', linestyle='--', label='Original Fit')
            axs[2].set_title('Bootstrap Distribution of k2')
            axs[2].set_xlabel('k2 (1/s)')
            axs[2].legend()
            plt.tight_layout()
            
            # Optional: save the figure
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=300)   
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=600)  
            plt.show()
        else:
            # --- Single-Exponential Fit (via curve_fit on CDF) ---
            # ... (Your existing single fit logic for when func != 'both') ...
            bin_width = bin_width
            bin_edges = range(0, xlim+1, bin_width)

            # Bin the 'CumulativeTime(s)' column
            bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False)  # Set right=False to include the rightmost edge

            # Count the number of values in each bin
            bin_counts = bins.value_counts().sort_index()

            # Calculate cumulative count
            cumulative_counts = bin_counts.cumsum()

            bin_edges_array = bin_edges[:-1] 
            # Exclude the last edge to match the length of bin_counts
            cumulative_counts_array = cumulative_counts.values
            cumulative_counts_array = cumulative_counts_array/cumulative_counts_array.max()
            # plt.show()

            # perform the fit
            p0 = (20, 100, 0.1) # start with values near those we expect
            params, cv = scipy.optimize.curve_fit(func, bin_edges_array, cumulative_counts_array, p0)
            Y0, Plateau, K = params
            tauSec = (1 / K) 
            half_time = np.log(2)/K

            # Calculate standard errors from the covariance matrix
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2

            # Calculate confidence interval for the half-time (assuming normal distribution)
            alpha = 0.05  # significance level
            z_score = scipy.stats.norm.ppf(1 - alpha / 2)  # two-tailed z-score
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            n_value = cumulative_counts.max()


            print("Estimated Half-Time:", half_time)
            print("Standard Error of Half-Time:", se_half_time)
            print("95% Confidence Interval of Half-Time:", ci_half_time)

            # determine quality of the fit
            squaredDiffs = np.square(cumulative_counts_array - func(bin_edges_array, Y0, Plateau, K))
            squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

            # inspect the parameters
            print(f"R² = {rSquared}")
            print(f"Y = {Y0}")
            print(f'Plateau = {Plateau}')
            print(f'K = {K}')
            print(f'half-time = {half_time} s')
            print(f"Tau = {tauSec} s")

            fitted_data = func(bin_edges_array, Y0, Plateau, K)
            fitted_data_df = pd.DataFrame(fitted_data)
            x_bins = pd.DataFrame(bin_edges_array)
            test = pd.DataFrame(cumulative_counts_array)
            test = pd.concat([test, fitted_data_df, x_bins],axis=1)
            test.columns = ['Cumative_hist', 'fit', 'x_bins']
            test['treatment'] = treatment
            test['transition_type'] = transition
            data.append(test)


            col = [half_time, se_half_time, n_value, rSquared, treatment, transition]
            col_halftime_df = pd.DataFrame([col], columns=['mean', 'sem', 'n', 'r_squared', 'treatment', 'transition'])
            summary.append(col_halftime_df)
            pass
            
    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.concat(summary, ignore_index=True)
    
    bootstrap_df = pd.DataFrame(all_bootstrap_data)
    
    # Return three DataFrames now
    return fits_df, halftime_summary, bootstrap_df

def cumulative_residence_fitting2(dfs, output_folder, bin_width, xlim, func=one_phase_association):
    """Function is used to fit cumulative histogram data with a one-phase association curve. The script will create bins from the raw data and create a cumulative histogram, which is then
    used to fit the curve to the data. Will return the fit (with half time, plateua, etc) and a an Rsquared value to provide a measure of goodness of fit.
    Args:
        dfs (df): dataframe containing raw data to be used for fitting.
        output_folder (str): where to save data.
        bin_width (float): bin_width used to calculate the fit. Recommended to use smaller bin_widths (especially if data is tightly distributed at low values), but note
                            smaller bin_widths will reduce the number of datapoints in each bin.
        xlim (float): value used to determine how far the fit will extend to. Recommended to extend to max possible bin value.
        func (float, optional): decide here what fit to use. Defaults to one_phase_association. Can call another fit as long as it has been previously defined in a function.
    Returns:
        df: returns the fits and also the summary data (containing the half-time for each treatment and residence time state).
    """
    data = []
    summary = []
    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        print(treatment)
        print(transition)
        bin_width = bin_width
        bin_edges = range(0, xlim+1, bin_width)
        # Bin the 'CumulativeTime(s)' column
        bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False)  # Set right=False to include the rightmost edge
        # Count the number of values in each bin
        bin_counts = bins.value_counts().sort_index()
        # Calculate cumulative count
        cumulative_counts = bin_counts.cumsum()
        bin_edges_array = bin_edges[:-1]
        # Exclude the last edge to match the length of bin_counts
        cumulative_counts_array = cumulative_counts.values
        cumulative_counts_array = cumulative_counts_array/cumulative_counts_array.max()
        # plt.show()
        fits_output = []
        if func == 'both':
            print('single-fit first')
            p0 = (20, 100, 0.1)  # initial guess
            params, cv = scipy.optimize.curve_fit(one_phase_association, bin_edges_array, cumulative_counts_array, p0)
            Y0, Plateau, K = params
            tauSec = (1 / K)
            half_time = np.log(2) / K
            # Calculate standard error of half-time
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2
            # 95% confidence interval
            alpha = 0.05
            z_score = scipy.stats.norm.ppf(1 - alpha / 2)
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            n_value = cumulative_counts.max()
            print(treatment)
            print(transition)
            print('bruh')
            print("Estimated Half-Time:", half_time)
            print("Standard Error of Half-Time:", se_half_time)
            print("95% Confidence Interval of Half-Time:", ci_half_time)
            # R² of single fit
            squaredDiffs = np.square(cumulative_counts_array - one_phase_association(bin_edges_array, Y0, Plateau, K))
            squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

            fitted_data = one_phase_association(bin_edges_array, Y0, Plateau, K)
            fitted_data_df = pd.DataFrame(fitted_data)
            bin_edges = np.arange(0, xlim + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2
            res_single = cumulative_counts_array - fitted_data
            x_bins = pd.DataFrame(bin_edges_array)
            test_single = pd.DataFrame(cumulative_counts_array)
            test_single = pd.concat([test_single, fitted_data_df, x_bins, pd.DataFrame(res_single)], axis=1)
            test_single.columns = ['Cumative_hist_sing', 'fit_sing', 'x_bins_sing', 'residuals_sing']
            test_single['treatment'] = treatment
            test_single['transition_type'] = transition
            data.append(test_single)
            print(f"MLE (CDF) fitting for {treatment} {transition}")
            # MLE part
            dwell_times = df["CumulativeTime(s)"].values
            dwell_times = dwell_times[dwell_times > 0]
            def CDF_mixture(x, k1, k2, frac):
                return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))
            def MLE_cdf(params, x):
                k1, k2, frac = params
                if not (0 < frac < 1):
                    return np.inf
                if k1 <= 0 or k2 <= 0:
                    return np.inf
                pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
                pdf_vals = np.clip(pdf_vals, 1e-12, None)
                return -np.sum(np.log(pdf_vals))
            initial_guess = [1/20, 1/300, 0.5]
            bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
            result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
            k1, k2, frac = result.x
            print(f"Fitted: k1={k1:.4f}, k2={k2:.4f}, frac={frac:.4f}")
                        # ----------- BOOTSTRAP TO ESTIMATE STANDARD ERRORS -----------
            n_bootstrap = 500
            bootstrap_frac = []
            bootstrap_k1 = []
            bootstrap_k2 = []

            for i in range(n_bootstrap):
                boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
                result_boot = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(boot_sample,), bounds=bounds)
                if result_boot.success:
                    k1_boot, k2_boot, frac_boot = result_boot.x
                    bootstrap_frac.append(frac_boot)
                    bootstrap_k1.append(k1_boot)
                    bootstrap_k2.append(k2_boot)

            bootstrap_frac = np.array(bootstrap_frac)
            bootstrap_k1 = np.array(bootstrap_k1)
            bootstrap_k2 = np.array(bootstrap_k2)
            # Calculate bootstrap statistics
            def bootstrap_summary(values, name):
                mean = np.mean(values)
                std_error = np.std(values, ddof=1)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                print(f"Bootstrap {name}: mean={mean:.4f}, std error={std_error:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
                return mean, std_error, ci_lower, ci_upper
            frac_mean_bootstrap, frac_std_error, frac_ci_lower, frac_ci_upper = bootstrap_summary(bootstrap_frac, 'frac')
            k1_mean_bootstrap, k1_std_error, k1_ci_lower, k1_ci_upper = bootstrap_summary(bootstrap_k1, 'k1')
            k2_mean_bootstrap, k2_std_error, k2_ci_lower, k2_ci_upper = bootstrap_summary(bootstrap_k2, 'k2')
            # ----------- END BOOTSTRAP -----------
            bin_edges = np.arange(0, xlim + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2
            hist, _ = np.histogram(dwell_times, bins=bin_edges)
            cum_counts = np.cumsum(hist)
            cum_norm = cum_counts / cum_counts[-1]
            fitted_cdf = CDF_mixture(bin_centers, k1, k2, frac)
            # R² of double fit
            ss_res = np.sum((cum_norm - fitted_cdf) ** 2)
            ss_tot = np.sum((cum_norm - np.mean(cum_norm)) ** 2)
            r_squared_double = 1 - ss_res / ss_tot
            print(f'{treatment} R2 is {r_squared_double}')
            # Residuals plot
            res = cum_norm - fitted_cdf
            fig = plt.figure(figsize=(6, 2))
            plt.scatter(bin_centers, res_single, label='single_fit', s=8, c='#f05423')
            plt.scatter(bin_centers, res, label='double_fit', s=8, c='#00aeef')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"Residuals for {treatment} {transition} ")
            plt.xlabel("Resident time (s)")
            plt.ylabel("Residual")
            plt.xlim(0, 100)
            plt.ylim(-0.25, 0.25)
            plt.legend()
            plt.show()
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=300)
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=600)
            # Half-times
            half_time_fast = np.log(2) / k1
            half_time_slow = np.log(2) / k2
            fit_df = pd.DataFrame({
                'x_bins': bin_centers,
                'fit': fitted_cdf,
                'Cumative_hist': cum_norm,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(fit_df)
            col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K, rSquared,
                        half_time_fast, half_time_slow, k1, k2, frac, frac_std_error, r_squared_double,
                        k1_std_error, k2_std_error,bootstrap_frac.mean(), bootstrap_k1.mean(), bootstrap_k2.mean()]],
                    columns=['mean', 'sem', 'n', 'treatment', 'transition', 'K', 'r_squared',
                            'half_time_fast', 'half_time_slow', 'k1', 'k2', 'frac_fast', 'frac_std_error', 'R2',
                            'k1_std_error', 'k2_std_error', 'bootstrap_frac', 'bootstrap_k1', 'bootstrap_k2'])
            summary.append(col_df)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            # Bootstrap plot for frac
            sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[0], color="#939990FF", edgecolor='black')
            axs[0].axvline(frac, color='black', linestyle='--', label='Original Fit')
            axs[0].set_title('Bootstrap Distribution of frac')
            axs[0].set_xlabel('frac')
            axs[0].legend()
            # Bootstrap plot for k1
            sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="#386B20FF", edgecolor='black')
            axs[1].axvline(k1, color='black', linestyle='--', label='Original Fit')
            axs[1].set_title('Bootstrap Distribution of k1')
            axs[1].set_xlabel('k1 (1/s)')
            axs[1].legend()
            # Bootstrap plot for k2
            sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="#81b868df", edgecolor='black')
            axs[2].axvline(k2, color='black', linestyle='--', label='Original Fit')
            axs[2].set_title('Bootstrap Distribution of k2')
            axs[2].set_xlabel('k2 (1/s)')
            axs[2].legend()
            plt.tight_layout()
            plt.show()
            # Optional: save the figure
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=300)   
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=600)   
        else:
            bin_width = bin_width
            bin_edges = range(0, xlim+1, bin_width)

            # Bin the 'CumulativeTime(s)' column
            bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False)  # Set right=False to include the rightmost edge

            # Count the number of values in each bin
            bin_counts = bins.value_counts().sort_index()

            # Calculate cumulative count
            cumulative_counts = bin_counts.cumsum()

            bin_edges_array = bin_edges[:-1] 
            # Exclude the last edge to match the length of bin_counts
            cumulative_counts_array = cumulative_counts.values
            cumulative_counts_array = cumulative_counts_array/cumulative_counts_array.max()
            # plt.show()

            # perform the fit
            p0 = (20, 100, 0.1) # start with values near those we expect
            params, cv = scipy.optimize.curve_fit(func, bin_edges_array, cumulative_counts_array, p0)
            Y0, Plateau, K = params
            tauSec = (1 / K) 
            half_time = np.log(2)/K

            # Calculate standard errors from the covariance matrix
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2

            # Calculate confidence interval for the half-time (assuming normal distribution)
            alpha = 0.05  # significance level
            z_score = scipy.stats.norm.ppf(1 - alpha / 2)  # two-tailed z-score
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            n_value = cumulative_counts.max()


            print("Estimated Half-Time:", half_time)
            print("Standard Error of Half-Time:", se_half_time)
            print("95% Confidence Interval of Half-Time:", ci_half_time)

            # determine quality of the fit
            squaredDiffs = np.square(cumulative_counts_array - func(bin_edges_array, Y0, Plateau, K))
            squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

            # inspect the parameters
            print(f"R² = {rSquared}")
            print(f"Y = {Y0}")
            print(f'Plateau = {Plateau}')
            print(f'K = {K}')
            print(f'half-time = {half_time} s')
            print(f"Tau = {tauSec} s")

            fitted_data = func(bin_edges_array, Y0, Plateau, K)
            fitted_data_df = pd.DataFrame(fitted_data)
            x_bins = pd.DataFrame(bin_edges_array)
            test = pd.DataFrame(cumulative_counts_array)
            test = pd.concat([test, fitted_data_df, x_bins],axis=1)
            test.columns = ['Cumative_hist', 'fit', 'x_bins']
            test['treatment'] = treatment
            test['transition_type'] = transition
            data.append(test)


            col = [half_time, se_half_time, n_value, rSquared, treatment, transition]
            col_halftime_df = pd.DataFrame([col], columns=['mean', 'sem', 'n', 'r_squared', 'treatment', 'transition'])
            summary.append(col_halftime_df)

    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.concat(summary, ignore_index=True)
    return fits_df, halftime_summary

