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


# Extract the Additive Constant parameter fits.
additive_constant_columns = [
    col
    for col in chinchilla_fits_df.columns
    if col.startswith("Correct Eqn. Parameters_Additive Constant_")
]
additive_constant_fits_df = chinchilla_fits_df[additive_constant_columns].copy()
additive_constant_fits_df.loc["Constant"] = [
    float(col.split("_")[2]) for col in additive_constant_columns
]


# Create the colormap for multiplicative constant perturbations.
cmap = matplotlib.colormaps.get_cmap("flare")
additive_constants = additive_constant_fits_df.T["Constant"].values
additive_constants_to_colors_dict = {
    additive_constant: cmap(i / len(additive_constants))
    for i, additive_constant in enumerate(additive_constants)
}

# Plot the compute-optimal tokens per parameter for additive constant perturbations.
plt.close()
fig = plt.figure(figsize=(12, 6))
ax = plt.gca()
training_flop = chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"]
for models_parameters_column in additive_constant_columns:
    low = chinchilla_tokens_per_parameter_df[models_parameters_column + "_Low"]
    median = chinchilla_tokens_per_parameter_df[models_parameters_column + "_Median"]
    high = chinchilla_tokens_per_parameter_df[models_parameters_column + "_High"]
    additive_constant = float(models_parameters_column.split("_")[2])
    ax.plot(
        training_flop,
        median,
        label=src.analyze.sci_notation_trimmed(additive_constant),
        color=additive_constants_to_colors_dict[additive_constant],
    )
    ax.fill_between(
        training_flop,
        low,
        high,
        color=additive_constants_to_colors_dict[additive_constant],
        alpha=0.2,
    )
ax.set_xlabel("Training Compute (FLOP)")
ax.set_ylabel("Compute-Optimal\nTokens per Parameter")
ax.set_title("Additive Constant")
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_ylim(1e-2, 1e4)
ax.axhline(y=20, color="black", linestyle="--")
ax.text(
    x=1e24,
    y=20,
    s=r"$D/N = 20$ rule of thumb",
    color="black",
    fontsize=20,
    verticalalignment="bottom",
)
ax.legend(title=r"Constant ($c$)", loc="center left", bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="compute_optimal_tokens_per_parameter_by_compute_additive_constant",
)
# plt.show()

# Plot the fit parameters for multiplicative constant perturbation.
plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(fit_parameters),
    figsize=(30, 9),
    sharex=True,
    sharey=False,
)
for ax_idx, (ax, fit_parameter) in enumerate(zip(axes, fit_parameters)):
    # Set the axes titles.
    if fit_parameter == "alpha":
        latex_title = r"$\hat{\alpha}$"
    elif fit_parameter == "beta":
        latex_title = r"$\hat{\beta}$"
    else:
        latex_title = rf"$\hat{{{fit_parameter}}}$"
    ax.set_title(latex_title)
    # # Control the y limits.
    if fit_parameter == "E":
        # ax.set_ylim(1.74, 1.86)
        pass
    elif fit_parameter == "A":
        # ax.set_ylim(1e1, 1e4)
        ax.set_yscale("log")
    elif fit_parameter == "alpha":
        # ax.set_ylim(0.30, 0.42)
        pass
    elif fit_parameter == "B":
        ax.set_ylim(-1000, 5000)
    elif fit_parameter == "beta":
        ax.set_ylim(0.30, 0.42)
    for col_idx, systematic_bias_column in enumerate(additive_constant_columns):
        ax.errorbar(
            x=[additive_constant_fits_df.loc["Constant", systematic_bias_column]],
            y=[
                additive_constant_fits_df.loc[
                    fit_parameter + "_fit", systematic_bias_column
                ]
            ],
            yerr=[
                1.96
                * additive_constant_fits_df.loc[
                    fit_parameter + "_se", systematic_bias_column
                ]
            ],
            color=additive_constants_to_colors_dict[
                additive_constant_fits_df.loc["Constant", systematic_bias_column]
            ],
            marker="d",
            markersize=20,
            linewidth=2,
        )
    ax.set_xlabel(r"Constant ($c$)")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="fit_parameters_additive_constant"
)
plt.show()


# Extract the Multiplicative Constant parameter fits.
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


# Create the colormap for multiplicative constant perturbations.
cmap = matplotlib.colormaps.get_cmap("rocket")
multiplicative_constants = multiplicative_constant_fits_df.T["Constant"].values
multiplicative_constants_to_colors_dict = {
    multiplicative_constant: cmap(i / len(multiplicative_constants))
    for i, multiplicative_constant in enumerate(multiplicative_constants)
}

# Plot the compute-optimal tokens per parameter for multiplicative constant perturbations.
plt.close()
fig = plt.figure(figsize=(12, 6))
ax = plt.gca()
training_flop = chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"]
for models_parameters_column in multiplicative_constant_columns:
    low = chinchilla_tokens_per_parameter_df[models_parameters_column + "_Low"]
    median = chinchilla_tokens_per_parameter_df[models_parameters_column + "_Median"]
    high = chinchilla_tokens_per_parameter_df[models_parameters_column + "_High"]
    multiplicative_constant = float(models_parameters_column.split("_")[2])
    ax.plot(
        training_flop,
        median,
        label=np.round(multiplicative_constant, 3),
        color=multiplicative_constants_to_colors_dict[multiplicative_constant],
    )
    ax.fill_between(
        training_flop,
        low,
        high,
        color=multiplicative_constants_to_colors_dict[multiplicative_constant],
        alpha=0.2,
    )
ax.set_xlabel("Training Compute (FLOP)")
ax.set_ylabel("Compute-Optimal\nTokens per Parameter")
ax.set_title("Multiplicative Constant")
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_ylim(1e-2, 1e4)
ax.axhline(y=20, color="black", linestyle="--")
ax.text(
    x=1e24,
    y=20,
    s=r"$D/N = 20$ rule of thumb",
    color="black",
    fontsize=20,
    verticalalignment="bottom",
)
ax.legend(title=r"Constant ($c$)", loc="center left", bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="compute_optimal_tokens_per_parameter_by_compute_multiplicative_constant",
)
# plt.show()

# Plot the fit parameters for multiplicative constant perturbation.
plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(fit_parameters),
    figsize=(30, 9),
    sharex=True,
    sharey=False,
)
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
        ax.set_ylim(1e1, 1e4)
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
            color=multiplicative_constants_to_colors_dict[
                multiplicative_constant_fits_df.loc["Constant", systematic_bias_column]
            ],
            marker="d",
            markersize=20,
            linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Constant ($c$)")
    ax.set_xlim(1.0 / 3e3, 3e3)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="fit_parameters_multiplicative_constant"
)
# plt.show()

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

cmap = matplotlib.colormaps.get_cmap("mako")
slopes = systematic_bias_fits_df.T["Slope"].values
slopes_to_colors_dict = {slope: cmap(i / len(slopes)) for i, slope in enumerate(slopes)}

# Plot the compute-optimal tokens per parameter for systematic bias perturbations.
plt.close()
fig = plt.figure(figsize=(12, 6))
ax = plt.gca()
training_flop = chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"]
for systematic_bias_column in systematic_bias_columns:
    low = chinchilla_tokens_per_parameter_df[systematic_bias_column + "_Low"]
    median = chinchilla_tokens_per_parameter_df[systematic_bias_column + "_Median"]
    high = chinchilla_tokens_per_parameter_df[systematic_bias_column + "_High"]
    slope = float(systematic_bias_column.split("_")[2])
    ax.plot(
        training_flop,
        median,
        label=np.round(slope, 3),
        color=slopes_to_colors_dict[slope],
    )
    ax.fill_between(
        training_flop,
        low,
        high,
        color=slopes_to_colors_dict[slope],
        alpha=0.2,
    )
ax.set_xlabel("Training Compute (FLOP)")
ax.set_ylabel("Compute-Optimal\nTokens per Parameter")
ax.set_title("Systematic Bias")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-1, 1e4)
ax.axhline(y=20, color="black", linestyle="--")
ax.text(
    x=1e24,
    y=20,
    s=r"$D/N = 20$ rule of thumb",
    color="black",
    fontsize=20,
    verticalalignment="bottom",
)
ax.legend(title=r"Slope ($s$)", loc="center left", bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="compute_optimal_tokens_per_parameter_by_compute_systematic_bias",
)
plt.show()

# Plot the fit parameters for systematic bias perturbation.
plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(fit_parameters),
    figsize=(30, 9),
    sharex=True,
    sharey=False,
)
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
