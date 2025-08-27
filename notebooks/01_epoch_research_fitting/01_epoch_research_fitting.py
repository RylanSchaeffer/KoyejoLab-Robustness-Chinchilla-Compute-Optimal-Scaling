import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import OptimizeWarning
from scipy.stats import linregress
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
    src.analyze.compute_or_load_chinchilla_fit_dataframes(
        data_dir=data_dir,
        models_parameters_columns=[
            "Model Size",
            "Reported Parameters",
            "Best Fit Formula Parameters",
            "Standard Formula Parameters",
        ],
        refresh=False,
        bootstraps=4000,
    )
)

# Modernize the column names.
chinchilla_fits_df.columns = [
    "Model Size",
    "Reported Parameters",
    "Best Fit Formula Parameters",
    "Correct Formula Parameters",
]
chinchilla_tokens_per_parameter_df_replacement_columns = []
for column in chinchilla_tokens_per_parameter_df.columns:
    if column.startswith("Best Fit Formula Parameters"):
        chinchilla_tokens_per_parameter_df_replacement_columns.append(
            column.replace("Best Fit Formula Parameters", "Best Fit Formula Parameters")
        )
    elif column.startswith("Standard Formula Parameters"):
        chinchilla_tokens_per_parameter_df_replacement_columns.append(
            column.replace("Standard Formula Parameters", "Correct Formula Parameters")
        )
    else:
        chinchilla_tokens_per_parameter_df_replacement_columns.append(column)

chinchilla_tokens_per_parameter_df.columns = (
    chinchilla_tokens_per_parameter_df_replacement_columns
)

models_parameters_columns = [
    # "Model Size",
    "Correct Formula Parameters",
    "Best Fit Formula Parameters",
    "Reported Parameters",
]
models_parameters_columns_colors = {
    models_parameters_column: plt.cm.viridis((2 - i) / len(models_parameters_columns))
    for i, models_parameters_column in enumerate(models_parameters_columns)
}
models_parameters_columns_markers = {
    "Reported Parameters": "o",
    "Best Fit Formula Parameters": "s",
    "Correct Formula Parameters": "d",
}
fit_parameters = ["E", "A", "alpha", "B", "beta"]


plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(fit_parameters),
    figsize=(30, 9),
    sharex=True,
    sharey=False,
)
axes[0].set_ylabel("Value")
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
        ax.set_ylim(0, 1000)
    elif fit_parameter == "alpha":
        ax.set_ylim(0.30, 0.42)
    elif fit_parameter == "B":
        ax.set_ylim(-1000, 5000)
    elif fit_parameter == "beta":
        ax.set_ylim(0.30, 0.42)

    for col_idx, models_parameters_column in enumerate(models_parameters_columns):
        ax.errorbar(
            x=[1 + col_idx],
            y=[
                chinchilla_fits_df.loc[fit_parameter + "_fit", models_parameters_column]
            ],
            yerr=[
                1.96
                * chinchilla_fits_df.loc[
                    fit_parameter + "_se", models_parameters_column
                ]
            ],
            color=models_parameters_columns_colors[models_parameters_column],
            marker=models_parameters_columns_markers[models_parameters_column],
            markersize=20,
            linewidth=2,
        )
    ax.set_xlim((0.5, 3.5))
    ax.set_xticks([1, 2, 3])  # Set the positions of the ticks
    ax.set_xticklabels(
        [col.replace(" Parameters", "") for col in models_parameters_columns],
        rotation=45,
        ha="right",
    )  # Set the labels and rotate them
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="fit_parameters"
)
# plt.show()


plt.close()
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(models_parameters_columns),
    figsize=(20, 6),
    sharey=True,
    sharex=True,
)
training_flop = chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"]
for ax_idx, (ax, models_parameters_column) in enumerate(
    zip(axes.flat, models_parameters_columns)
):
    low = chinchilla_tokens_per_parameter_df[models_parameters_column + " Low"]
    median = chinchilla_tokens_per_parameter_df[models_parameters_column + " Median"]
    high = chinchilla_tokens_per_parameter_df[models_parameters_column + " High"]

    slope, intercept, r, p, se = linregress(np.log10(training_flop), median)

    print("Model parameters:", models_parameters_column)
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("R:", r)
    print("P:", p)
    print("SE:", se)
    print("\n\n\n")

    ax.plot(
        training_flop,
        median,
        # label=models_parameters_column,
        color=models_parameters_columns_colors[models_parameters_column],
        # label=f"{slope:.3e}",
    )
    ax.fill_between(
        training_flop,
        low,
        high,
        color=models_parameters_columns_colors[models_parameters_column],
        alpha=0.2,
    )
    ax.set_xlabel("Training Compute (FLOP)")
    if ax_idx == 0:
        ax.set_ylabel("Compute-Optimal\nTokens per Parameter")
    ax.set_title(models_parameters_column.replace(" Parameters", ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.axhline(y=20, color="black", linestyle="--")
    ax.text(
        x=1e24,
        y=20,
        s=r"$D/N = 20$ rule of thumb",
        color="black",
        fontsize=20,
        verticalalignment="bottom",
    )
    # ax.legend()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="compute_optimal_tokens_per_parameter_by_compute",
)
plt.show()
print("Finished 01_epoch_research_fitting!")
