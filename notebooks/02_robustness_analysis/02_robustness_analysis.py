import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import OptimizeWarning
import seaborn as sns
import warnings


import src.analyze
import src.plot

np.seterr(over="ignore")
np.seterr(invalid="ignore")
warnings.filterwarnings("ignore", category=OptimizeWarning)

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

chinchilla_fits_df, chinchilla_tokens_per_parameter_df = (
    src.analyze.compute_or_load_chinchilla_robustness_fit_dataframes(
        data_dir=data_dir,
        models_parameters_columns=[
            "Correct Eqn. Parameters",
        ],
    )
)
fit_parameters = ["E", "A", "alpha", "B", "beta"]


# Plot Multiplicative Constant parameter fits.
multiplicative_constant_columns = [
    col
    for col in chinchilla_fits_df.columns
    if col.startswith("Correct Eqn. Parameters_Multiplicative Constant_")
]
multiplicative_constant_fits_df = chinchilla_fits_df[
    multiplicative_constant_columns
].copy()
multiplicative_constant_fits_df.loc["Constant"] = [
    float(col.split("_")[2]) for col in multiplicative_constant_columns
]
plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(fit_parameters),
    figsize=(30, 9),
    sharex=True,
    sharey=False,
)
cmap = matplotlib.colormaps.get_cmap("rocket")
constants = multiplicative_constant_fits_df.T["Constant"].values
# slopes_to_colors_dict = {slope: cmap(slope) for slope in slopes}
slopes_to_colors_dict = {
    constant: cmap(i / len(constants)) for i, constant in enumerate(constants)
}
for ax_idx, (ax, fit_parameter) in enumerate(zip(axes, fit_parameters)):
    # Set the axes titles.
    if fit_parameter == "alpha":
        latex_title = r"$\hat{\alpha}$"
    elif fit_parameter == "beta":
        latex_title = r"$\hat{\beta}$"
    else:
        latex_title = rf"$\hat{{{fit_parameter}}}$"
    ax.set_title(latex_title)

    # Control the y limits.
    if fit_parameter == "E":
        ax.set_ylim(1.74, 1.86)
    elif fit_parameter == "A":
        ax.set_yscale("log")
        ax.set_ylim(1e2, 1e4)
    elif fit_parameter == "alpha":
        ax.set_ylim(0.30, 0.42)
    elif fit_parameter == "B":
        ax.set_ylim(-1000, 5000)
    elif fit_parameter == "beta":
        ax.set_ylim(0.30, 0.42)

    for col_idx, systematic_bias_column in enumerate(multiplicative_constant_columns):
        ax.errorbar(
            x=[multiplicative_constant_fits_df.loc["Constant", systematic_bias_column]],
            y=[
                multiplicative_constant_fits_df.loc[
                    fit_parameter + "_fit", systematic_bias_column
                ]
            ],
            yerr=[
                1.96
                * multiplicative_constant_fits_df.loc[
                    fit_parameter + "_se", systematic_bias_column
                ]
            ],
            color=slopes_to_colors_dict[
                multiplicative_constant_fits_df.loc["Constant", systematic_bias_column]
            ],
            marker="d",
            markersize=20,
            linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Constant ($c$)")
    ax.set_xlim(1e-3, 1e3)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="fit_parameters_multiplicative_constant"
)
plt.show()

# Plot Systematic Bias parameter fits.
systematic_bias_columns = [
    col
    for col in chinchilla_fits_df.columns
    if col.startswith("Correct Eqn. Parameters_Systematic Bias_")
]
systematic_bias_fits_df = chinchilla_fits_df[systematic_bias_columns].copy()
# Add a new row with the slope extracted from the column title.
systematic_bias_fits_df.loc["Slope"] = [
    float(col.split("_")[2]) for col in systematic_bias_columns
]

plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(fit_parameters),
    figsize=(30, 9),
    sharex=True,
    sharey=False,
)
cmap = matplotlib.colormaps.get_cmap("mako")
slopes = systematic_bias_fits_df.T["Slope"].values
norm = matplotlib.colors.LogNorm(
    vmin=slopes.min(),
    vmax=slopes.max(),
)
# slopes_to_colors_dict = {slope: cmap(slope) for slope in slopes}
slopes_to_colors_dict = {slope: cmap(i / len(slopes)) for i, slope in enumerate(slopes)}
for ax_idx, (ax, fit_parameter) in enumerate(zip(axes, fit_parameters)):
    # Set the axes titles.
    if fit_parameter == "alpha":
        latex_title = r"$\hat{\alpha}$"
    elif fit_parameter == "beta":
        latex_title = r"$\hat{\beta}$"
    else:
        latex_title = rf"$\hat{{{fit_parameter}}}$"
    ax.set_title(latex_title)

    # Control the y limits.
    if fit_parameter == "E":
        ax.set_ylim(1.74, 1.86)
    elif fit_parameter == "A":
        ax.set_yscale("log")
        ax.set_ylim(1e-0, 1e10)
    elif fit_parameter == "alpha":
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1.3e0)
    elif fit_parameter == "B":
        ax.set_ylim(-1000, 5000)
    elif fit_parameter == "beta":
        ax.set_ylim(0.30, 0.42)

    for col_idx, systematic_bias_column in enumerate(systematic_bias_columns):
        ax.errorbar(
            x=[systematic_bias_fits_df.loc["Slope", systematic_bias_column]],
            y=[
                systematic_bias_fits_df.loc[
                    fit_parameter + "_fit", systematic_bias_column
                ]
            ],
            yerr=[
                1.96
                * systematic_bias_fits_df.loc[
                    fit_parameter + "_se", systematic_bias_column
                ]
            ],
            color=slopes_to_colors_dict[
                systematic_bias_fits_df.loc["Slope", systematic_bias_column]
            ],
            marker="d",
            markersize=20,
            linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Score ($s$)")
    ax.set_xlim(1 / 4.0, 4.0)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="fit_parameters_systematic_bias"
)
plt.show()

print("Finished 02_robustness_analysis!")
