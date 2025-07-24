import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

import numpy as np
import pandas as pd

import src.analyze
import src.plot


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

chinchilla_parameters_df = src.analyze.load_chinchilla_model_parameters_csv()

correct_eqn_parameters = chinchilla_parameters_df["Correct Eqn. Parameters"].values
min_lim_val = correct_eqn_parameters.min()
max_lim_val = correct_eqn_parameters.max()

sigmas = [0.0] + np.logspace(-0.5, 0.5, num=10).tolist()
multiplicative_constants = np.logspace(-3, 3, num=11)
slopes = np.logspace(-0.5, 0.5, 11)

# Create the colormap for log normal noise.
cmap = matplotlib.colormaps.get_cmap("crest")
sigmas_to_colors_dict = {sigma: cmap(i / len(sigmas)) for i, sigma in enumerate(sigmas)}
# Plot the effect of the log normal noise.
plt.close()
fig = plt.figure(figsize=(10, 8))
for sigma in sigmas:
    plt.plot(
        correct_eqn_parameters,
        (
            src.analyze.parameter_transformation_log_normal_noise(
                parameters=correct_eqn_parameters,
                sigma=sigma,
                repeat_idx=0,
            )
            if sigma != 0.0
            else correct_eqn_parameters
        ),
        label=np.round(sigma, 3),
        color=sigmas_to_colors_dict[sigma],
    )
# plt.plot(
#     [min_lim_val, max_lim_val], [min_lim_val, max_lim_val], linestyle="--", color="k"
# )
plt.legend(title=r"Noise ($\sigma$)", loc="center left", bbox_to_anchor=(1, 0.5))
plt.xscale("log")
plt.xlabel("Correct Eqn. Model Parameters")
plt.yscale("log")
plt.ylabel("Perturbed Model Parameters")
plt.title("Log Normal Noise")
fig.subplots_adjust(right=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=perturbed-params_x=params_log_normal_noise",
)
plt.show()


# Create the colormap for multiplicative constant perturbations.
cmap = matplotlib.colormaps.get_cmap("rocket")
multiplicative_constants_to_colors_dict = {
    constant: cmap(i / len(multiplicative_constants))
    for i, constant in enumerate(multiplicative_constants)
}
# Plot the effect of the multiplicative constant perturbations.
plt.close()
fig = plt.figure(figsize=(10, 8))
for multiplicative_constant in multiplicative_constants:
    plt.plot(
        correct_eqn_parameters,
        src.analyze.parameter_transformation_multiplicative_constant(
            parameters=correct_eqn_parameters,
            c=multiplicative_constant,
        ),
        label=np.round(multiplicative_constant, 3),
        color=multiplicative_constants_to_colors_dict[multiplicative_constant],
    )
# plt.plot(
#     [min_lim_val, max_lim_val], [min_lim_val, max_lim_val], linestyle="--", color="k"
# )
plt.legend(title=r"Constant ($c$)", loc="center left", bbox_to_anchor=(1, 0.5))
plt.xscale("log")
plt.xlabel("Correct Eqn. Model Parameters")
plt.yscale("log")
plt.ylabel("Perturbed Model Parameters")
plt.title("Multiplicative Constant")
fig.subplots_adjust(right=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=perturbed-params_x=params_multiplicative_constant",
)
plt.show()


cmap = matplotlib.colormaps.get_cmap("mako")
slopes_to_colors_dict = {slope: cmap(i / len(slopes)) for i, slope in enumerate(slopes)}

plt.close()
fig = plt.figure(figsize=(10, 8))
for slope in slopes:
    plt.plot(
        correct_eqn_parameters,
        src.analyze.parameter_transformation_systematic_bias(
            parameters=correct_eqn_parameters,
            slope=slope,
        ),
        label=np.round(slope, 3),
        color=slopes_to_colors_dict[slope],
    )
plt.legend(title=r"Slope ($s$)", loc="center left", bbox_to_anchor=(1, 0.5))
plt.xscale("log")
plt.xlabel("Correct Eqn. Model Parameters")
plt.yscale("log")
plt.ylabel("Perturbed Model Parameters")
plt.title("Systematic Bias")
fig.subplots_adjust(right=0.8)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=perturbed-params_x=params_systematic_bias",
)
plt.show()

print("Finished 03_robustness_schematic!")
