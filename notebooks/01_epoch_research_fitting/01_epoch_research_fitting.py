# Adapted from https://github.com/epoch-research/analyzing-chinchilla/blob/main/data_analysis.ipynb
# import matplotlib.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    src.analyze.compute_or_load_chinchilla_fit_dataframes(
        data_dir=data_dir,
    )
)

models_parameters_columns = [
    # "Model Size",
    "Reported Parameters",
    "Incorrect Eqn. Parameters",
    "Correct Eqn. Parameters",
]
models_parameters_columns_colors = [
    # "#FFDDDD",
    "#DDFFDD",
    "#DDDDFF",
    "#FFFFDD",
]  # Light Red, Green, Blue, Yellow


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

    plot_color = plt.cm.viridis(
        ax_idx / len(models_parameters_columns)
    )  # Get a color from a colormap
    ax.plot(training_flop, median, label=models_parameters_column, color=plot_color)
    ax.fill_between(training_flop, low, high, color=plot_color, alpha=0.2)
    ax.set_xlabel("Training Compute (FLOP)")
    if ax_idx == 0:
        ax.set_ylabel("Compute-Optimal\nTokens per Parameter")
    ax.set_title(models_parameters_column)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.axhline(y=20, color="gray", linestyle="--")
    ax.text(
        x=7e24,
        y=20,
        s=r"$D/N = 20$ rule of thumb",
        color="gray",
        fontsize=14,
        verticalalignment="bottom",
    )

src.plot.save_plot_with_multiple_extensions(
    results_dir,
    plot_filename="compute_optimal_tokens_per_parameter_by_models_parameters",
)
plt.show()
print("Finished 01_epoch_research_fitting!")
