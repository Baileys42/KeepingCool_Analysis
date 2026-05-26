
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os as os


font = {'weight' : 'normal', 'size'   : 12 }
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

def simulate_state_transitions(num_states=4, rate_up=0.9, rate_down=0.3, simulation_time=100, 
                                time_step=0.5, seed=None):
    """
    Simulates state transitions with rates dictating the likelihood of transitions,
    with probabilities adjusted based on the number of available states.

    Parameters:
        num_states (int): Number of states in the system.
        rate_up (float): Base rate constant for transitioning up (average time = 1/rate_up).
        rate_down (float): Base rate constant for transitioning down (average time = 1/rate_down).
        simulation_time (float): Total time to simulate.
        time_step (float): Time step for each iteration.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing time points and state transitions.
    """
    if seed is not None:
        np.random.seed(seed)
    # Initialize state probabilities with a random starting state
    current_state = np.random.randint(0, num_states)
    # Track state transitions over time
    state_transitions = []
    time_points = []
    # Simulation loop
    current_time = 0
    while current_time < simulation_time:
        state_transitions.append(current_state)
        time_points.append(current_time)
        # Adjust transition probabilities based on the number of available states
        transition_up_prob = rate_up * time_step * (num_states - current_state - 1) / (num_states - 1) if current_state < num_states - 1 else 0
        transition_down_prob = rate_down * time_step * current_state / (num_states - 1) if current_state > 0 else 0
        # Determine if a transition occurs
        rand_val = np.random.rand()
        if rand_val < transition_up_prob:
            current_state += 1  # Transition up
        elif rand_val < transition_up_prob + transition_down_prob:
            current_state -= 1  # Transition down
        # Increment time
        current_time += time_step
    # Prepare data for seaborn
    data = pd.DataFrame({'Time': time_points, 'State': state_transitions})
    return data

def plot_binary_state(data, plot_export, threshold_state=1, noise_level=0.05, identifier=None):
    """
    Plots a binary state transition over time based on a threshold state, with optional noise added.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Time' and 'State' columns.
        threshold_state (int): The state threshold for binary classification. States <= threshold_state are 1, others are 0.
        noise_level (float): Standard deviation of Gaussian noise to add to the binary state.
    """
    # Add a binary column based on the threshold state
    data['Binary State'] = data['State'].apply(lambda x: 1 if x <= threshold_state else 0)
    # Add Gaussian noise to the binary state if noise_level > 0
    if noise_level > 0:
        data['Binary State'] = data['Binary State'] + np.random.normal(0, noise_level, len(data))
    # Plot the binary state transition over time
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x='Time', y='Binary State')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Binary State (1: State <= {threshold_state}, 0: State > {threshold_state})')
    plt.title('Binary State Transition Over Time (with Noise)')
    fig.savefig(f'{plot_export}/binary_states_{identifier}.svg', dpi=300)
    fig.savefig(f'{plot_export}/binary_states_{identifier}.png', dpi=300)
    plt.show()
    return data


def analyze_dwell_times(plot_export, threshold_state=0, num_simulations=300, num_states=4, rate_up=1.5, rate_down=0.3, 
                        simulation_time=360, time_step=1, identifier=None, xlim=None):
    """
    Analyzes dwell times for a specified binary threshold state.

    Parameters:
        threshold_state (int): The state threshold for binary classification. States <= threshold_state are 1, others are 0.
        num_simulations (int): Number of simulations to run.
        num_states (int): Number of states in the system.
        rate_up (float): Rate constant for transitioning up.
        rate_down (float): Rate constant for transitioning down.
        simulation_time (float): Total time to simulate.
        time_step (float): Time step for each iteration.

    Returns:
        tuple: Mean dwell time, concatenated simulation data, and a DataFrame containing fit parameters.
    """
    all_dwell_times = []
    all_simulation_data = []  # List to store all simulation data

    for sim in range(num_simulations):
        data = simulate_state_transitions(num_states=num_states, rate_up=rate_up, rate_down=rate_down, 
                                          simulation_time=simulation_time, time_step=time_step, 
                                          seed=sim)
        # Append the simulation data with an identifier for the simulation
        data['Simulation'] = sim
        all_simulation_data.append(data)
        # Check if the threshold state or below is present in the simulation
        if not any(data['State'] <= threshold_state):
            continue  # Skip this simulation if no relevant states are present
        data['Binary State'] = data['State'].apply(lambda x: 1 if x <= threshold_state else 0)
        binary_state_1_durations = []
        current_duration = 0
        for i in range(len(data) - 1):
            if data['Binary State'].iloc[i] == 0:  # Binary State 0 corresponds to states above the threshold
                current_duration += data['Time'].iloc[i + 1] - data['Time'].iloc[i]
            else:
                if current_duration > 0:
                    binary_state_1_durations.append(current_duration)
                    current_duration = 0
        # Exclude the final dwell time if it ends at state 0
        if current_duration > 0 and data['Binary State'].iloc[-1] != 0:
            binary_state_1_durations.append(current_duration)
        all_dwell_times.extend(binary_state_1_durations)
    # Concatenate all simulation data into a single DataFrame
    concatenated_data = pd.concat(all_simulation_data, ignore_index=True)
    # Plot cumulative dwell times
    sorted_dwell_times = np.sort(all_dwell_times)
    cumulative_prob = np.arange(1, len(sorted_dwell_times) + 1) / len(sorted_dwell_times)
    # Fit to single and double exponential models
    def single_exponential(x, a, b):
        return a * (1 - np.exp(-b * x))

    def double_exponential(x, a1, b1, a2, b2):
        return a1 * (1 - np.exp(-b1 * x)) + a2 * (1 - np.exp(-b2 * x))
    # Fit single exponential
    popt_single, _ = curve_fit(single_exponential, sorted_dwell_times, cumulative_prob, p0=[1, 0.1])
    # Fit double exponential
    popt_double, _ = curve_fit(double_exponential, sorted_dwell_times, cumulative_prob, 
                           p0=[0.5, 0.1, 0.5, 0.01], maxfev=5000)
    # Plot fits
    x_fit = np.linspace(0, max(sorted_dwell_times), 1000)
    y_single = single_exponential(x_fit, *popt_single)
    y_double = double_exponential(x_fit, *popt_double)

    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.step(sorted_dwell_times, cumulative_prob, where='post', label='Data', color='black')
    plt.plot(x_fit, y_single, label=f'Single Exp Fit (a={popt_single[0]:.2f}, b={popt_single[1]:.2f})', linestyle='--')
    plt.plot(x_fit, y_double, label=f'Double Exp Fit (a1={popt_double[0]:.2f}, b1={popt_double[1]:.2f}, a2={popt_double[2]:.2f}, b2={popt_double[3]:.2f})', linestyle=':', color='red')
    plt.xlabel('Dwell Time (s)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.xlim(-5, xlim if xlim is not None else max(sorted_dwell_times) * 1.1)
    # fig.savefig(f'{plot_export}/cumulative_dwell_times_{identifier}.svg', dpi=600)
    # fig.savefig(f'{plot_export}/cumulative_dwell_times_{identifier}.png', dpi=600)
    plt.show()

    # Print fit parameters
    print("Single Exponential Fit Parameters:")
    print(f"a = {popt_single[0]:.4f}, b = {popt_single[1]:.4f}")

    print("\nDouble Exponential Fit Parameters:")
    print(f"a1 = {popt_double[0]:.4f}, b1 = {popt_double[1]:.4f}, a2 = {popt_double[2]:.4f}, b2 = {popt_double[3]:.4f}")
    # Calculate mean dwell time and standard error using bootstrapping
    mean_dwell_time = np.mean(all_dwell_times)
    std_error = np.std(all_dwell_times, ddof=1) / np.sqrt(len(all_dwell_times)) if len(all_dwell_times) > 1 else 0
    n_dwell = len(all_dwell_times)

    # Bootstrapping for standard error of mean dwell time, k1, and k2
    n_bootstrap = 500
    if len(all_dwell_times) > 1:
        boot_means = []
        boot_k1 = []
        boot_k2 = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(all_dwell_times, size=len(all_dwell_times), replace=True)
            boot_means.append(np.mean(boot_sample))
            # Fit double exponential to bootstrap sample
            try:
                # Only fit if all dwell times are positive and greater than zero
                if np.any(np.array(boot_sample) <= 0):
                    boot_k1.append(np.nan)
                    boot_k2.append(np.nan)
                    continue
                popt_double_boot, _ = curve_fit(double_exponential, np.sort(boot_sample),
                                                np.arange(1, len(boot_sample) + 1) / len(boot_sample),
                                                p0=[0.5, 0.1, 0.5, 0.01], 
                                                bounds=([0, 0, 0, 0], [1, np.inf, 1, np.inf]), 
                                                maxfev=5000)
                # Only accept positive rates
                if popt_double_boot[1] > 0 and popt_double_boot[3] > 0:
                    boot_k1.append(popt_double_boot[1])
                    boot_k2.append(popt_double_boot[3])
                else:
                    boot_k1.append(np.nan)
                    boot_k2.append(np.nan)
            except Exception:
                # If fit fails, append nan
                boot_k1.append(np.nan)
                boot_k2.append(np.nan)
        boot_means = np.array(boot_means)
        boot_k1 = np.array(boot_k1)
        boot_k2 = np.array(boot_k2)
        std_error = np.std(boot_means, ddof=1)
        k1_std_error = np.nanstd(boot_k1, ddof=1)
        k2_std_error = np.nanstd(boot_k2, ddof=1)
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        print(f"Bootstrap Mean Dwell Time: mean={mean_dwell_time:.4f}, std error={std_error:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
        print(f"Bootstrap k1: mean={np.nanmean(boot_k1):.4f}, std error={k1_std_error:.4f}")
        print(f"Bootstrap k2: mean={np.nanmean(boot_k2):.4f}, std error={k2_std_error:.4f}")
    else:
        ci_lower = ci_upper = mean_dwell_time  
        k1_std_error = np.nan
        k2_std_error = np.nan
    print(f"\nMean Dwell Time: {mean_dwell_time:.4f} seconds")
    print(f"Standard Error: {std_error:.4f}")
    print(f"Number of Dwell Times (n): {n_dwell}")

    # Create a DataFrame to store fit parameters and dwell statistics, including k1/k2 std errors
    fit_parameters_df = pd.DataFrame({
        'Model': ['Single Exponential', 'Double Exponential'],
        'a': [popt_single[0], popt_double[0]],
        'b': [popt_single[1], popt_double[1]],
        'a2': [None, popt_double[2]],
        'b2': [None, popt_double[3]],
        'mean_dwell_time': [mean_dwell_time, mean_dwell_time],
        'std_error': [std_error, std_error],
        'n': [n_dwell, n_dwell],
        'k1_std_error': [None, k1_std_error],
        'k2_std_error': [None, k2_std_error]
    })

    return mean_dwell_time, concatenated_data, fit_parameters_df

def calculate_average_state_per_simulation(concatenated_data, plot_export):
    """
    Calculates the average state for each simulation.

    Parameters:
        concatenated_data (pd.DataFrame): DataFrame containing 'State' and 'Simulation' columns.

    Returns:
        pd.DataFrame: DataFrame containing 'Simulation' and 'Average State' columns.
    """
    # Group by simulation and calculate the average state
    # Calculate the average state for each simulation
    average_state_df = concatenated_data.groupby('Simulation')['State'].mean().reset_index()
    average_state_df.rename(columns={'State': 'Average State'}, inplace=True)

    # Calculate the occurrence of each state for each simulation
    state_occurrences = concatenated_data.groupby(['Simulation', 'State']).size().unstack(fill_value=0).reset_index()

    # Merge the average state and state occurrences into a single DataFrame
    result_df = pd.merge(average_state_df, state_occurrences, on='Simulation')

    # Calculate the proportion of time each state is occupied across all simulations
    total_time_per_state = concatenated_data['State'].value_counts(normalize=True).sort_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=total_time_per_state.index, y=total_time_per_state.values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('State')
    plt.ylabel('Proportion of Time Occupied')
    plt.title('Proportion of Time Each State is Occupied Across All Simulations')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plot_export}/average_state_per_simulation.svg', dpi=300)
    plt.show()

    return result_df

def calculate_average_state_per_simulation2(concatenated_data, plot_export, palette='BuPu'):
    """
    Calculates the average state for each simulation and plots the proportion of time each state is occupied
    across all simulations, grouped by num_states, with each num_states on a separate subplot.

    Parameters:
        concatenated_data (pd.DataFrame): DataFrame containing 'State', 'Simulation', and 'num_states' columns.

    Returns:
        pd.DataFrame: DataFrame containing 'Simulation', 'Average State', and state occurrences.
    """
    # Group by simulation and calculate the average state
    average_state_df = concatenated_data.groupby(['Simulation', 'num_states'])['State'].mean().reset_index()
    average_state_df.rename(columns={'State': 'Average State'}, inplace=True)
    # Calculate the occurrence of each state for each simulation
    state_occurrences = concatenated_data.groupby(['Simulation', 'num_states', 'State']).size().unstack(fill_value=0).reset_index()
    # Merge the average state and state occurrences into a single DataFrame
    result_df = pd.merge(average_state_df, state_occurrences, on=['Simulation', 'num_states'])
    # Calculate the proportion of time each state is occupied across all simulations, grouped by num_states
    total_time_per_state = concatenated_data.groupby(['num_states', 'State']).size().groupby(level=0).apply(
        lambda x: x / x.sum()).reset_index(name='Proportion')
    # Get unique num_states for subplots
    unique_num_states = total_time_per_state['num_states'].unique()
    num_subplots = len(unique_num_states)
    # Create subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True, sharey=True)
    # Generate a palette with a number of colors equal to the maximum number of unique num_states
    max_num_states = len(unique_num_states)
    color_palette = sns.color_palette(palette, max_num_states)
    # Sort unique_num_states to ensure the darkest color corresponds to the highest num_states
    sorted_num_states = sorted(unique_num_states)
    for i, num_states in enumerate(sorted_num_states):
        subset = total_time_per_state[total_time_per_state['num_states'] == num_states]
        sns.barplot(
            data=subset, 
            x='State', 
            y='Proportion', 
            ax=axes[i], 
            palette=[color_palette[i]],  # Assign a specific color for each num_states
            edgecolor='black', 
            alpha=0.7)
        axes[i].tick_params(axis='x')  # Rotate x-axis labels for better readability
        axes[i].set_xlabel('')  # Remove x-axis label for individual subplots
        axes[i].set_ylabel('')  # Remove y-axis label for individual subplots
        subset = total_time_per_state[total_time_per_state['num_states'] == num_states]
        sns.barplot(
            data=subset, 
            x='State', 
            y='Proportion', 
            ax=axes[i], 
            palette=sns.color_palette(palette, len(unique_num_states)), 
            edgecolor='black', 
            alpha=0.7)
        axes[i].tick_params(axis='x')  # Rotate x-axis labels for better readability
        axes[i].set_xlabel('')  # R
        axes[i].set_ylabel('')  # R

    # Add a single, centered ylabel
    fig.text(0.04, 0.5, 'Proportion of Time Occupied', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout to accommodate the ylabel
    plt.savefig(f'{plot_export}/average_state_per_simulation2.svg', dpi=300)
    plt.show()
    return result_df

def simulate_and_analyze_states(num_states_range, rate_up_list, rate_down, simulation_time, time_step, threshold_state, noise_level, num_simulations, identifier_prefix, plot_export):
    """
    Simulates state transitions for a range of num_states, analyzes dwell times, and combines results into DataFrames.

    Parameters:
        num_states_range (range): Range of num_states to simulate (e.g., range(3, 7)).
        rate_up (float): Rate constant for transitioning up.
        rate_down (float): Rate constant for transitioning down.
        simulation_time (float): Total time to simulate.
        time_step (float): Time step for each iteration.
        threshold_state (int): The state threshold for binary classification.
        noise_level (float): Standard deviation of Gaussian noise to add to the binary state.
        num_simulations (int): Number of simulations to run for dwell time analysis.
        identifier_prefix (str): Prefix for plot identifiers.
        plot_export (str): Directory to save plots.

    Returns:
        dict: Dictionary containing concatenated data and average state DataFrames for each num_states.
    """
    results = {}

    for rate_up in rate_up_list:  # Loop through rate_up values
        for num_states in num_states_range:  # Loop through num_states
            print(f"Simulating and analyzing for num_states={num_states}, rate_up={rate_up}")
            # Simulate state transitions
            data = simulate_state_transitions(num_states=num_states, rate_up=rate_up, rate_down=rate_down, 
                                              simulation_time=simulation_time, time_step=time_step, seed=1)
            data['num_states'] = num_states  # Add num_states column to the DataFrame
            data['rate_up'] = rate_up  # Add rate_up column to the DataFrame
            # Plot results using seaborn
            sns.set_style('ticks')
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x='Time', y='State')
            plt.xlabel('Time (s)')
            plt.ylabel('')
            plt.title(f'num_states={num_states}, rate_up={rate_up}')
            plt.yticks(range(num_states), [f'State {i}' for i in range(num_states)])
            plt.savefig(f'{plot_export}/simulated_states_num_states_{num_states}_rate_up_{rate_up}.svg', dpi=300)
            plt.show()

            # Plot binary state
            data = plot_binary_state(data, plot_export, threshold_state=threshold_state, noise_level=noise_level, identifier=f"{identifier_prefix}_{num_states}_rate_up_{rate_up}")
            # Analyze dwell times
            mean_dwell_time, concatenated_data, fits = analyze_dwell_times(plot_export, threshold_state=threshold_state, num_simulations=num_simulations, 
                                                                     num_states=num_states, rate_up=rate_up, rate_down=rate_down, 
                                                                     simulation_time=simulation_time, time_step=time_step, 
                                                                     identifier=f"{identifier_prefix}_{num_states}_rate_up_{rate_up}", xlim=simulation_time)
            concatenated_data['num_states'] = num_states  # Add num_states column to concatenated_data
            concatenated_data['rate_up'] = rate_up  # Add rate_up column to concatenated_data

            # Calculate average state per simulation
            average_state_df = calculate_average_state_per_simulation(concatenated_data, plot_export)

            # Plot the average state per simulation
            sns.set_style('ticks')
            plt.figure(figsize=(10, 6))
            sns.histplot(data=average_state_df, x='Average State', color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel('Average State')
            plt.ylabel('Frequency')
            plt.title(f'Average State Per Simulation (num_states={num_states}, rate_up={rate_up})')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'{plot_export}/average_state_histogram_num_states_{num_states}_rate_up_{rate_up}.svg', dpi=300)
            plt.show()

            # Store results in dictionary
            results[(num_states, rate_up)] = {
                'concatenated_data': concatenated_data,
                'average_state_df': average_state_df, 
                'fits': fits,
            }
    return results

def df_processing_for_plots(results, state_x=2, state_y=1, plot_export=None):
    """
    Processes simulation results for plotting and data export.
    This function aggregates and analyzes simulation results stored in the `results` dictionary,
    producing several summary DataFrames useful for downstream plotting and analysis. Optionally,
    it exports these DataFrames as CSV files if a `plot_export` directory is provided.
    Parameters
    ----------
    results : dict
        Dictionary where keys are tuples (num_states, rate_up) and values are dicts containing:
            - 'average_state_df': DataFrame with average state data.
            - 'concatenated_data': DataFrame with concatenated simulation data.
            - 'fits': DataFrame with fit results, including dwell times and rate constants.
    state_x : int, optional
        State to count as the origin of transitions (default is 2).
    state_y : int, optional
        State to count as the destination of transitions (default is 1).
    plot_export : str or None, optional
        Directory path to export resulting DataFrames as CSV files. If None, no files are exported.
    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
            - combined_average_state_df: Concatenated average state DataFrame across all results.
            - combined_concat: Concatenated simulation data across all results.
            - proportion_time_combined: Proportion of time spent in states 0 or 1 for each num_states and rate_up.
            - transitions_x_to_y_rate: Average rate of transitions from `state_x` to `state_y` per num_states and rate_up.
            - mean_dwell_times_by_num_states_rate_up: Mean dwell times grouped by num_states and rate_up.
            - fits_data_melted: Melted DataFrame of mean rate constants (k1 and k2) for plotting.
    Notes
    -----
    - The function expects the input `results` to be structured as described above.
    - The function is designed for use with simulation data from smFRET or similar state-based models.
    - Exported CSVs are saved in the specified `plot_export` directory if provided.
    """
    combined_average_state_df = pd.concat([result['average_state_df']
        .assign(num_states=key[0], rate_up_down=key[1])
        for key, result in results.items()], ignore_index=True)


    combined_concat = pd.concat(
        [result['concatenated_data'].assign(num_states=key[0], rate_up_down=key[1]) for key, result in results.items()],
        ignore_index=True)

    # Calculate proportion of time for each num_states spent in state 0 or 1 combined
    proportion_time_combined = combined_concat[combined_concat['State'].isin([0, 1])].groupby(['num_states', 'rate_up']).size() / combined_concat.groupby(['num_states', 'rate_up']).size()
    proportion_time_combined = proportion_time_combined.reset_index()

    # Calculate the average rate of transitions from state 2 to state 1 for each num_states and rate_up (events per unit time)
    transitions_x_to_y_rate = combined_concat.groupby(['num_states', 'rate_up', 'Simulation']).apply(
        lambda df: ((df['State'].shift(1) == state_x) & (df['State'] == state_y)).sum() / df['Time'].max()).groupby(['num_states', 'rate_up']).mean()
    transitions_x_to_y_rate = transitions_x_to_y_rate.reset_index()

    # Extract fits data and analyze dwell times for each unique num_states and rate_up
    fits_data = pd.concat([result['fits'].assign(num_states=num_states, rate_up=rate_up) for (num_states, rate_up), result in results.items()], ignore_index=True)

    # Group by num_states and rate_up, and calculate the mean dwell time, std_error, and n for each
    mean_dwell_times_by_num_states_rate_up = fits_data.groupby(['num_states', 'rate_up']).agg({
        'mean_dwell_time': 'mean',
        'std_error': 'mean',
        'n': 'mean'
    }).reset_index()

    # Plot the mean values of k1 and k2 as a scatterplot with categorical x-axis (up and down layout)
    # Also keep the k1_std_error and k2_std_error
    fits_data_double = fits_data[fits_data['Model'] == 'Double Exponential']
    fits_data_mean = fits_data_double.groupby(['num_states', 'rate_up']).agg(
        k1_mean=('b', 'mean'),
        k2_mean=('b2', 'mean'),
        k1_std_error=('k1_std_error', 'mean'),
        k2_std_error=('k2_std_error', 'mean')
    ).reset_index()

    # Melt for plotting, keeping std errors
    fits_data_melted = pd.melt(
        fits_data_mean,
        id_vars=['num_states', 'rate_up', 'k1_std_error', 'k2_std_error'],
        value_vars=['k1_mean', 'k2_mean'],
        var_name='Rate Type',
        value_name='Mean Rate Constant'
    )
    # Add std error column for each row
    fits_data_melted['Rate Std Error'] = fits_data_melted.apply(
        lambda row: row['k1_std_error'] if row['Rate Type'] == 'k1_mean' else row['k2_std_error'], axis=1
    )
    # Clean up Rate Type for plotting
    fits_data_melted['Rate Type'] = fits_data_melted['Rate Type'].map({'k1_mean': 'k1', 'k2_mean': 'k2'})

    
    # Save datasets if plot_export is provided
    if plot_export is not None:
        combined_average_state_df.to_csv(f"{plot_export}/combined_average_state_df.csv", index=False)
        combined_concat.to_csv(f"{plot_export}/combined_concat.csv", index=False)
        proportion_time_combined.to_csv(f"{plot_export}/proportion_time_combined.csv", index=False)
        transitions_x_to_y_rate.to_csv(f"{plot_export}/transitions_x_to_y_rate.csv", index=False)
        mean_dwell_times_by_num_states_rate_up.to_csv(f"{plot_export}/mean_dwell_times_by_num_states_rate_up.csv", index=False)
        fits_data_melted.to_csv(f"{plot_export}/fits_data_melted.csv", index=False)

    return (combined_average_state_df, 
            combined_concat, 
            proportion_time_combined, 
            transitions_x_to_y_rate, 
            mean_dwell_times_by_num_states_rate_up, 
            fits_data_melted)

def kinetic_sequence_space_analysis(plot_export, 
                                    time_step=1,
                                    rate_up_list=[0.2, 0.3, 0.05], 
                                    rate_down_list=[0.025, 0.05, 0.1], 
                                    nstates=[3, 6], 
                                    threshold_state=1,
                                    simulation_time=360):
                                    
    """
    Performs a parameter sweep over a range of kinetic rates and state numbers to analyze dwell times in a simulated kinetic sequence space.
    For each combination of number of states, rate_up, and rate_down, this function simulates state transitions, analyzes dwell times, 
    fits the dwell time distributions, and extracts kinetic parameters. The results are saved to a CSV file and returned as a DataFrame.
    Parameters
    ----------
    plot_export : str
        Directory path where plots and results will be exported.
    rate_up_list : list of float, optional
        List containing [start, stop, step] values for the upward transition rate sweep (default is [0.2, 0.3, 0.05]).
    rate_down_list : list of float, optional
        List containing [start, stop, step] values for the downward transition rate sweep (default is [0.025, 0.05, 0.1]).
    nstates : list of int, optional
        List containing [min_states, max_states] for the number of states to analyze (default is [3, 6]).
    threshold_state : int, optional
        State threshold used in dwell time analysis (default is 1).
    simulation_time : int or float, optional
        Total simulation time for each run (default is 360).
    time_step : int or float, optional
        Time step for the simulation (default is 1).
    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame containing the results for each parameter combination, including:
            - num_states
            - rate_up
            - rate_down
            - mean_dwell_time
            - k1 (fast rate from double exponential fit)
            - k2 (slow rate from double exponential fit)
            - prop_fast (proportion of fast component)
    Notes
    -----
    - The function expects an `analyze_dwell_times` function to be defined elsewhere, which performs the simulation and fitting.
    - Results are saved as 'mean_dwell_time_results.csv' in the specified `plot_export` directory.
    """
                                    
    results_data = []
    for num_states in range(nstates[0], nstates[1]+1):  
        for rate_up in np.arange(rate_up_list[0], rate_up_list[1], rate_up_list[2]):  # Rate up from 0.05 to 0.4
            for rate_down in np.arange(rate_down_list[0], rate_down_list[1], rate_down_list[2]):  # Rate down from 0.05 to 0.4
                print(f"Analyzing: num_states={num_states}, rate_up={rate_up}, rate_down={rate_down}")
                try:
                # Analyze dwell times and capture the mean dwell time
                    mean_dwell_time, concatenated_data, fit_param = analyze_dwell_times(plot_export=plot_export,
                    threshold_state=threshold_state, num_simulations=300, 
                    num_states=num_states, rate_up=rate_up, 
                    rate_down=rate_down, simulation_time=simulation_time, 
                    time_step=time_step
                )

                # Extract k1 and k2 rates from the fit parameters
                    k1 = fit_param.loc[fit_param['Model'] == 'Double Exponential', 'b'].values[0]
                    k2 = fit_param.loc[fit_param['Model'] == 'Double Exponential', 'b2'].values[0]
                    prop_fast = fit_param.loc[fit_param['Model'] == 'Double Exponential', 'a'].values[0]

                # Save the results
                    results_data.append({
                    'num_states': num_states,
                    'rate_up': rate_up,
                    'rate_down': rate_down,
                    'mean_dwell_time': mean_dwell_time,
                    'k1': k1,
                    'k2': k2, 
                    'prop_fast': prop_fast, 
                })
                except Exception as e:
                    print(f"Error analyzing parameters: {e}")
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results_data)
    results_df['mean_dwell_time'] = results_df['mean_dwell_time'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
    results_df.to_csv(f'{plot_export}/mean_dwell_time_results.csv', index=False)
    return results_df



