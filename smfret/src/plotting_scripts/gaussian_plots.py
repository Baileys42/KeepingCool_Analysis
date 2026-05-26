import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from lmfit import models, Parameters
import smfret.src.Utilities.Data_analysis as uda


def fit_3gauss_dif_constrained_nativespont(df, treatment, save_loc, mu_1, sigma_1, amplitude_1, gamma_1, mu_2, sigma_2, amplitude_2, mu_3, sigma_3, amplitude_3, gamma_3):
    filt_df = df[df['treatment_name'] == treatment]
    bins = np.arange(-0.21, 1.1, 0.025) 
    inds = np.digitize(filt_df['FRET'].astype(float), bins)
    xdata, ydata = np.unique(inds, return_counts=True)
    ydata = ydata[1:-1] #### trim off outside range bins at the end
    xdata = [np.mean(bins[x : x + 2]) for x in range(len(bins)- 1)]  ##### convert bin edges to bin centres, therefore end up with one less bin
    sns.lineplot(xdata, ydata)

    model_1 = models.SkewedGaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')
    model_3 = models.SkewedGaussianModel(prefix='m3_')
    model = model_1 + model_2 + model_3 
   
    model_1.set_param_hint('m1_center', vary=False)

    model_2.set_param_hint('m2_sigma', vary=False)

    model_2.set_param_hint('m2_center', vary=False)
    model_3.set_param_hint('m3_gamma', vary=False)
    model_3.set_param_hint('m3_sigma', vary=False)
    model_3.set_param_hint('m3_center', vary=False)


    params_1 = model_1.make_params(center=mu_1, sigma=sigma_1, amplitude=amplitude_1, gamma=gamma_1, min=0)
    params_2 = model_2.make_params(center=mu_2, sigma=sigma_2, amplitude=amplitude_2, min=0)
    params_3 = model_3.make_params(center=mu_3, sigma=sigma_3, amplitude=amplitude_3, gamma=gamma_3, min=0)
    params = params_1.update(params_2)
    params = params.update(params_3)

    output = model.fit((ydata/np.max(ydata)), params, x=xdata)
    fig = output.plot(data_kws={'markersize': 3})

    paramaters = {name:output.params[name].value for name in output.params.keys()}
    fitx = np.arange(-0.2, 1.2, 0.025)

    fit1 = model_1.eval(x=fitx, center=paramaters['m1_center'], amplitude=abs(paramaters['m1_amplitude']), sigma=paramaters['m1_sigma'], gamma=paramaters['m1_gamma'])
    fit2 = model_2.eval(x=fitx, center=paramaters['m2_center'], amplitude=abs(paramaters['m2_amplitude']), sigma=paramaters['m2_sigma'], fwhm=paramaters['m2_fwhm'])
    fit3 = model_3.eval(x=fitx, center=paramaters['m3_center'], amplitude=abs(paramaters['m3_amplitude']), sigma=paramaters['m3_sigma'], gamma=paramaters['m3_gamma'])

    sns.lineplot(fitx, fit1)
    sns.lineplot(fitx, fit2)
    sns.lineplot(fitx, fit3)
    plt.show()

    # Calculate area under the curve for each gaussian
    aoc_m1 = paramaters['m1_amplitude']
    aoc_m2 = paramaters['m2_amplitude']
    aoc_m3 = paramaters['m3_amplitude']
    sum_aoc = aoc_m1 + aoc_m2 + aoc_m3 

    aoc_m1_percent_of_total = (aoc_m1/sum_aoc)*100
    aoc_m2_percent_of_total = (aoc_m2/sum_aoc)*100
    aoc_m3_percent_of_total = (aoc_m3/sum_aoc)*100
    list_of_gaus_proportion = [aoc_m1_percent_of_total, aoc_m2_percent_of_total, aoc_m3_percent_of_total]
    labels_of_gaus_proportion = ['m1', 'm2', 'm3']
    proportion_df = pd.DataFrame([labels_of_gaus_proportion, list_of_gaus_proportion])
    proportion_df.columns = proportion_df.iloc[0]
    proportion_df = proportion_df.drop(0)
    proportion_df['treatment'] = treatment
    proportion_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv')
    return proportion_df


def fit_2gauss_dif_constrained_nativespont(df, treatment, save_loc, mu_1, sigma_1, amplitude_1, gamma_1, mu_2, sigma_2, amplitude_2):
    filt_df = df[df['treatment_name'] == treatment]
    bins = np.arange(-0.21, 1.1, 0.025) 
    inds = np.digitize(filt_df['FRET'].astype(float), bins)
    xdata, ydata = np.unique(inds, return_counts=True)
    ydata = ydata[1:-1] #### trim off outside range bins at the end
    xdata = [np.mean(bins[x : x + 2]) for x in range(len(bins)- 1)]  ##### convert bin edges to bin centres, therefore end up with one less bin
    sns.lineplot(xdata, ydata)

    model_1 = models.SkewedGaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')

    model = model_1 + model_2
   
    model_1.set_param_hint('m1_center', vary=False)
    model_2.set_param_hint('m2_sigma', vary=False)
    model_2.set_param_hint('m2_center', vary=False)



    params_1 = model_1.make_params(center=mu_1, sigma=sigma_1, amplitude=amplitude_1, gamma=gamma_1, min=0)
    params_2 = model_2.make_params(center=mu_2, sigma=sigma_2, amplitude=amplitude_2, min=0)
    params = params_1.update(params_2)

    output = model.fit((ydata/np.max(ydata)), params, x=xdata)
    fig = output.plot(data_kws={'markersize': 3})

    paramaters = {name:output.params[name].value for name in output.params.keys()}
    fitx = np.arange(-0.2, 1.2, 0.025)

    fit1 = model_1.eval(x=fitx, center=paramaters['m1_center'], amplitude=abs(paramaters['m1_amplitude']), sigma=paramaters['m1_sigma'], gamma=paramaters['m1_gamma'])
    fit2 = model_2.eval(x=fitx, center=paramaters['m2_center'], amplitude=abs(paramaters['m2_amplitude']), sigma=paramaters['m2_sigma'], fwhm=paramaters['m2_fwhm'])

    sns.lineplot(fitx, fit1)
    sns.lineplot(fitx, fit2)
    plt.show()

    # Calculate area under the curve for each gaussian
    aoc_m1 = paramaters['m1_amplitude']
    aoc_m2 = paramaters['m2_amplitude']


    sum_aoc = aoc_m1 + aoc_m2 

    aoc_m1_percent_of_total = (aoc_m1/sum_aoc)*100
    aoc_m2_percent_of_total = (aoc_m2/sum_aoc)*100
    list_of_gaus_proportion = [aoc_m1_percent_of_total, aoc_m2_percent_of_total]
    labels_of_gaus_proportion = ['m1', 'm2']
    proportion_df = pd.DataFrame([labels_of_gaus_proportion, list_of_gaus_proportion])
    proportion_df.columns = proportion_df.iloc[0]
    proportion_df = proportion_df.drop(0)
    proportion_df['treatment'] = treatment
    proportion_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv')
    return proportion_df

def fit_3gauss_dif_constrained_nativespont2(df, treatment, save_loc,
                                            mu_1, sigma_1, amplitude_1,
                                            mu_2, sigma_2, amplitude_2,
                                            mu_3, sigma_3, amplitude_3, gamma_3):
    filt_df = df[df['treatment_name'] == treatment]
    print(treatment)

    bins = np.arange(-0.025, 1.1, 0.025)
    ydata, bin_edges = np.histogram(filt_df['FRET'].astype(float), bins=bins)
    xdata = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(len(xdata), len(ydata))

    sns.lineplot(x=xdata, y=ydata)
    plt.xlabel('FRET')
    plt.ylabel('Counts')
    plt.title(f'{treatment} FRET Histogram')
    plt.show()

    model_1 = models.GaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')
    model_3 = models.SkewedGaussianModel(prefix='m3_')
    model = model_1 + model_2 + model_3

    model_1.set_param_hint('m1_center', vary=False)
    model_1.set_param_hint('m1_sigma', min=0.01, max=0.1)
    model_1.set_param_hint('m1_amplitude', min=0)

    model_2.set_param_hint('m2_center', vary=False)
    model_2.set_param_hint('m2_sigma', min=0.01, max=0.12)

    model_3.set_param_hint('m3_center', vary=False)
    model_3.set_param_hint('m3_sigma', vary=False)
    model_3.set_param_hint('m3_gamma', vary=False)

    params_1 = model_1.make_params(center=mu_1, sigma=sigma_1, amplitude=amplitude_1)
    params_2 = model_2.make_params(center=mu_2, sigma=sigma_2, amplitude=amplitude_2)
    params_3 = model_3.make_params(center=mu_3, sigma=sigma_3, amplitude=amplitude_3, gamma=gamma_3)

    params = params_1.update(params_2)
    params = params.update(params_3)

    output = model.fit((ydata/np.max(ydata)), params, x=xdata)
    fig = output.plot(data_kws={'markersize': 3})

    paramaters = {name: output.params[name].value for name in output.params.keys()}
    fitx = np.arange(-0.2, 1.2, 0.025)

    fit1 = model_1.eval(x=fitx, center=paramaters['m1_center'], amplitude=abs(paramaters['m1_amplitude']), sigma=paramaters['m1_sigma'])
    fit2 = model_2.eval(x=fitx, center=paramaters['m2_center'], amplitude=abs(paramaters['m2_amplitude']), sigma=paramaters['m2_sigma'], fwhm=paramaters['m2_fwhm'])
    fit3 = model_3.eval(x=fitx, center=paramaters['m3_center'], amplitude=abs(paramaters['m3_amplitude']), sigma=paramaters['m3_sigma'], gamma=paramaters['m3_gamma'])

    sns.lineplot(fitx, fit1)
    sns.lineplot(fitx, fit2)
    sns.lineplot(fitx, fit3)
    plt.show()

    aoc_m1 = paramaters['m1_amplitude']
    aoc_m2 = paramaters['m2_amplitude']
    aoc_m3 = paramaters['m3_amplitude']
    sum_aoc = aoc_m1 + aoc_m2 + aoc_m3

    proportion_df = pd.DataFrame({
        'm1': [(aoc_m1/sum_aoc)*100],
        'm2': [(aoc_m2/sum_aoc)*100],
        'm3': [(aoc_m3/sum_aoc)*100],
        'treatment': [treatment]
    })
    proportion_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv', index=False)

    return proportion_df


def fit_3gauss_dif_constrained_nativespont3(df, treatment, save_loc,
                                           mu_1, sigma_1, amplitude_1,
                                           mu_2, sigma_2, amplitude_2,
                                           mu_3, sigma_3, amplitude_3, gamma_3):
    filt_df = df[df['treatment_name'] == treatment]
    print(f"Fitting treatment: {treatment}")

    bins = np.arange(-0.025, 1.1, 0.025)
    ydata, bin_edges = np.histogram(filt_df['FRET'].astype(float), bins=bins)
    xdata = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(f"xdata/ydata lengths: {len(xdata)}/{len(ydata)}")

    sns.lineplot(x=xdata, y=ydata)
    plt.xlabel('FRET'); plt.ylabel('Counts'); plt.title(f'{treatment} FRET Histogram')
    plt.show()

    model_1 = models.GaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')
    model_3 = models.SkewedGaussianModel(prefix='m3_')
    model = model_1 + model_2 + model_3

    model_1.set_param_hint('m1_center', value=mu_1, vary=True)
    model_1.set_param_hint('m1_sigma', value=sigma_1, vary=True)
    model_1.set_param_hint('m1_amplitude', value=amplitude_1, min=0)

    model_2.set_param_hint('m2_center', value=mu_2, min=0.55, max=0.75)
    model_2.set_param_hint('m2_sigma', value=sigma_2, min=0.01, max=0.1)
    model_2.set_param_hint('m2_amplitude', value=amplitude_2, min=0)

    model_3.set_param_hint('m3_center', value=mu_3, min=0.85, max=0.95)
    model_3.set_param_hint('m3_sigma', value=sigma_3, min=0.02, max=0.15)
    model_3.set_param_hint('m3_amplitude', value=amplitude_3, min=0)
    model_3.set_param_hint('m3_gamma', value=gamma_3, min=0, max=25)

    params = model.make_params()
    params.update(model_1.make_params())
    params.update(model_2.make_params())
    params.update(model_3.make_params())

    output = model.fit((ydata/np.max(ydata)), params, x=xdata)
    output.plot(data_kws={'markersize': 3})

    paramaters = {name: output.params[name].value for name in output.params.keys()}
    fitx = np.arange(-0.2, 1.2, 0.025)

    fit1 = model_1.eval(x=fitx, **{k: paramaters[k] for k in paramaters if k.startswith('m1_')})
    fit2 = model_2.eval(x=fitx, **{k: paramaters[k] for k in paramaters if k.startswith('m2_')})
    fit3 = model_3.eval(x=fitx, **{k: paramaters[k] for k in paramaters if k.startswith('m3_')})

    sns.lineplot(fitx, fit1)
    sns.lineplot(fitx, fit2)
    sns.lineplot(fitx, fit3)
    plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(filt_df['FRET'].astype(float), bins=bins, stat='density', color='gray', alpha=0.4, label='Data')
    plt.plot(fitx, fit1/np.max([fit1.max(), fit2.max(), fit3.max()]), 'r--', lw=2, label='Low FRET')
    plt.plot(fitx, fit2/np.max([fit1.max(), fit2.max(), fit3.max()]), 'b--', lw=2, label='Mid FRET')
    plt.plot(fitx, fit3/np.max([fit1.max(), fit2.max(), fit3.max()]), 'g--', lw=2, label='High FRET (Skewed)')
    plt.plot(fitx, (fit1+fit2+fit3)/np.max([fit1.max(), fit2.max(), fit3.max()]), 'k-', lw=2, label='Total fit')
    plt.xlabel('FRET'); plt.ylabel('Density'); plt.title(f'{treatment} — Fitted Gaussians')
    plt.legend(); sns.despine()
    plt.show()

    aoc_m1 = paramaters['m1_amplitude']
    aoc_m2 = paramaters['m2_amplitude']
    aoc_m3 = paramaters['m3_amplitude']
    sum_aoc = aoc_m1 + aoc_m2 + aoc_m3

    proportion_df = pd.DataFrame({
        'm1': [(aoc_m1/sum_aoc)*100],
        'm2': [(aoc_m2/sum_aoc)*100],
        'm3': [(aoc_m3/sum_aoc)*100],
        'treatment': [treatment]
    })
    proportion_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv', index=False)

    return proportion_df


def fit_3gauss_dif_constrained_nativespont4(
        df, treatment, save_loc,
        mu_1, sigma_1, amplitude_1,
        mu_2, sigma_2, amplitude_2,
        mu_3, sigma_3, amplitude_3, gamma_3):
    filt_df = df[df['treatment_name'] == treatment].copy()
    print("Processing treatment:", treatment)

    bins = np.arange(-0.025, 1.1, 0.025)
    ydata, bin_edges = np.histogram(filt_df['FRET'].astype(float), bins=bins)
    xdata = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print("Bins:", len(xdata), "Counts:", len(ydata))

    sns.lineplot(x=xdata, y=ydata, marker='o')
    plt.xlabel('FRET')
    plt.ylabel('Counts')
    plt.title(f'{treatment} FRET Histogram')
    plt.show()

    model_1 = models.GaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')
    model_3 = models.SkewedGaussianModel(prefix='m3_')
    model = model_1 + model_2 + model_3

    params = Parameters()

    params.add('m1_center', value=mu_1, vary=False)
    params.add('m2_center', value=mu_2, vary=False)
    params.add('m3_center', value=mu_3, vary=False)

    params.add('m1_sigma', value=sigma_1, min=0.005, max=0.12, vary=True)
    params.add('m2_sigma', value=sigma_2, min=0.01, max=0.12, vary=True)
    params.add('m3_sigma', value=sigma_3, min=0.005, max=0.07, vary=True)

    params.add('m3_gamma', value=gamma_3, vary=True, min=-2.0, max=2.0)

    total_init = amplitude_1 + amplitude_2 + amplitude_3
    params.add('total_amp', value=float(total_init), min=0)
    params.add('f1', value=amplitude_1/total_init, min=0.0, max=1.0)
    params.add('f2', value=amplitude_2/total_init, min=0.0, max=1.0)

    params.add('m1_amplitude', expr='total_amp * f1')
    params.add('m2_amplitude', expr='total_amp * f2')
    params.add('m3_amplitude', expr='total_amp * (1 - f1 - f2)')

    weights = 1.0 / np.sqrt(ydata + 1.0)

    output = model.fit(ydata, params, x=xdata, weights=weights)
    print(output.fit_report())

    best = {name: output.params[name].value for name in output.params.keys()}
    fitx = np.linspace(min(xdata)-0.05, max(xdata)+0.05, 500)

    fit1 = model_1.eval(x=fitx,
                        center=best['m1_center'],
                        amplitude=abs(best['m1_amplitude']),
                        sigma=best['m1_sigma'])
    fit2 = model_2.eval(x=fitx,
                        center=best['m2_center'],
                        amplitude=abs(best['m2_amplitude']),
                        sigma=best['m2_sigma'])
    fit3 = model_3.eval(x=fitx,
                        center=best['m3_center'],
                        amplitude=abs(best['m3_amplitude']),
                        sigma=best['m3_sigma'],
                        gamma=best['m3_gamma'])

    plt.figure(figsize=(8,4))
    sns.lineplot(x=xdata, y=ydata, marker='o', label='data')
    sns.lineplot(x=fitx, y=fit1, label='m1: low-FRET')
    sns.lineplot(x=fitx, y=fit2, label='m2: mid-FRET')
    sns.lineplot(x=fitx, y=fit3, label='m3: high-FRET')
    plt.xlabel('FRET')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.show()

    amp1 = best['m1_amplitude']
    amp2 = best['m2_amplitude']
    amp3 = best['m3_amplitude']
    sum_amp = amp1 + amp2 + amp3
    prop_df = pd.DataFrame({
        'm1_%': [(amp1 / sum_amp) * 100],
        'm2_%': [(amp2 / sum_amp) * 100],
        'm3_%': [(amp3 / sum_amp) * 100],
        'treatment': [treatment]
    })

    prop_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv', index=False)

    return prop_df


# -------------------------------- MASTER FUNCTION -----------------------------------------

def fit_gauss_master(input_folder, gauss_num='two', timepoint=None, df=None):
    output_folder = f"{input_folder}/GaussianFits"  ### modify for each experiment
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if df is None:
        filename = f'{input_folder}/Cleaned_FRET_histogram_data.csv'
        compiled_df = pd.read_csv(filename, header="infer")
    else:
        compiled_df = df
    treatment_list = list(compiled_df['treatment_name'].unique())
    collated = []
    if gauss_num == 'three':
        for i, df in enumerate(treatment_list):
            treatment = fit_3gauss_dif_constrained_nativespont(compiled_df, df, output_folder, 0.05, .1, 1, 1, .63, .1, .05, .95, .22, .03, -2.7)
            collated.append(treatment)
        collated_df = pd.concat(collated)
    else:
        for i, df in enumerate(treatment_list):
            treatment = fit_2gauss_dif_constrained_nativespont(compiled_df, df, output_folder, 0.05, .1, 1, 1, .63, .1, .05)
            collated.append(treatment)
        collated_df = pd.concat(collated)
    # ----------- Set timepoints to plot based on your conditions --------------
    # ----------- Cleans up data and melts it for plotting ---------------------
    if timepoint:
        collated_df['timepoint'] = collated_df['treatment'].map(timepoint)
        collated_df.drop('treatment', axis=1, inplace=True)
        collated_df.columns = ['DnaK-bound', 'Native', 'Misfolded', 'timepoint']
        test = pd.melt(collated_df, id_vars='timepoint', value_name='pop_percent')
        test['pop_percent'] = test['pop_percent'].astype(float)
        test.to_csv(f'{output_folder}/collated_populations.csv')
        return test, collated_df, output_folder
    return collated_df, output_folder


def plot_gauss_timelapse(data_from_exp, output_folder):
    data_col = []
    for data_name, data_path in data_from_exp.items():
        data = uda.file_reader(data_path, 'other')
        data['treatment'] = data_name
        data_col.append(data)
    final = pd.concat(data_col).reset_index()
    final = final.iloc[:, 2:]


    for pop, df in final.groupby('variable'):
        fig, ax = plt.subplots()
        sns.set_style('ticks',{'font_scale': 1.5})
        sns.lineplot(data=df, x='timepoint', y='pop_percent', hue='treatment', palette='BuPu', marker='o')
        plt.legend(title='', loc='best')
        plt.ylim(0, 100)
        plt.xlabel('Time (min)')
        plt.ylabel(f'{pop} (% of total)')
        [x.set_linewidth(1.5) for x in ax.spines.values()]
        [x.set_color('black') for x in ax.spines.values()]
        fig.savefig(f'{output_folder}/Proportion_of_{pop}.svg', dpi=600)
        plt.show()


def plot_gmm_fits(compiled_df, output_folder, palette=None):
    """Fit a 3-component Gaussian Mixture Model to the FRET distribution for
    each treatment, overlay the component curves on the histogram, and return
    a summary of component weights.

    Args:
        compiled_df (pd.DataFrame): DataFrame containing ``'FRET'`` and
            ``'treatment_name'`` columns.
        output_folder (str): Directory in which to save output SVG figures
            (``GMM_{treatment}.svg`` per treatment).
        palette (dict, optional): Mapping from component index (0, 1, 2, ...)
            to colour string.  Defaults to a built-in six-colour palette.

    Returns:
        pd.DataFrame: One row per treatment with columns
            ``['treatment_name', 'chap_%', 'nat_%', 'mf_%']``.
    """
    from sklearn.mixture import GaussianMixture
    import scipy.stats as stats

    if palette is None:
        palette = {0: '#FB8B24', 1: '#820263', 2: '#234E69',
                   3: '#1AB0B0', 4: '#D90368', 5: '#EA4746'}

    percent_names, percent_chap, percent_Nat, percent_MF = [], [], [], []

    for treatment, filt in compiled_df.groupby('treatment_name'):
        peaks1 = filt['FRET'].values
        peaks2 = peaks1.reshape(-1, 1)

        # Quick histogram preview before fitting
        bins = np.linspace(0, 1, 400)
        counts = (pd.DataFrame(
                      pd.cut(peaks1, bins=bins, right=True, labels=bins[:-1])
                      .value_counts())
                  .reset_index()
                  .sort_index())
        sns.lineplot(data=counts, x='index', y=0)
        plt.show()

        # Fit 3-component GMM
        gauss_model = GaussianMixture(n_components=3, covariance_type='full',
                                      means_init=[[0.1], [0.6], [0.95]])
        gauss_model.fit(peaks2)

        means   = gauss_model.means_
        weights = gauss_model.weights_
        covars  = gauss_model.covariances_

        percent_names.append(treatment)
        percent_chap.append(weights[0] * 100)
        percent_Nat.append(weights[1] * 100)
        percent_MF.append(weights[2] * 100)

        # Build per-component fit curves and overlay on histogram
        fit_vals = []
        for model in range(len(means)):
            x_range = np.linspace(0, 1, 1000)
            y_vals = (weights[model]
                      * stats.norm.pdf(x_range, means[model],
                                       np.sqrt(covars[model])).ravel())
            vals = pd.DataFrame({'x_fit': x_range, 'y_fit': y_vals})
            vals['model'] = model
            vals['norm_val'] = vals['y_fit'] / vals['y_fit'].max()
            fit_vals.append(vals)
        fit_vals = pd.concat(fit_vals, ignore_index=True)

        sns.lineplot(data=fit_vals, x='x_fit', y='y_fit',
                     hue='model', palette=palette, linewidth=2)
        sns.distplot(peaks1, kde=False, norm_hist=True, color='black')
        plt.savefig(f"{output_folder}/GMM_{treatment}.svg", dpi=600)
        plt.show()

    percent_df = pd.DataFrame({
        'treatment_name': percent_names,
        'chap_%':  percent_chap,
        'nat_%':   percent_Nat,
        'mf_%':    percent_MF,
    })
    return percent_df
    return final
