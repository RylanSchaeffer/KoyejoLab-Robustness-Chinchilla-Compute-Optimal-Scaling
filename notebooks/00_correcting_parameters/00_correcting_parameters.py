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
chinchilla_parameters_df["Relative Error Reported Parameters"] = 0.0

chinchilla_parameters_relative_error_tall_df = chinchilla_parameters_df[
    [
        "Reported Parameters",
        "Relative Error Reported Parameters",
        "Relative Error Correct Eqn.",
        "Relative Error Incorrect Eqn.",
    ]
].melt(
    id_vars="Reported Parameters",
    var_name="Equation",
    value_name="Relative Error (\%)",
)
chinchilla_parameters_relative_error_tall_df[
    "Equation"
] = chinchilla_parameters_relative_error_tall_df["Equation"].map(
    {
        "Relative Error Reported Parameters": "Reported",
        "Relative Error Correct Eqn.": "Correct Eqn.",
        "Relative Error Incorrect Eqn.": "Incorrect Eqn.",
    }
)
chinchilla_parameters_relative_error_tall_df["Relative Error Is Zero"] = (
    chinchilla_parameters_relative_error_tall_df["Relative Error (\%)"] == 0
)

print(
    chinchilla_parameters_relative_error_tall_df.groupby(["Equation"])[
        "Relative Error (\%)"
    ].max()
)

fraction_relative_error_zero_under_incorrect_eqn = np.mean(
    chinchilla_parameters_relative_error_tall_df[
        chinchilla_parameters_relative_error_tall_df["Equation"] == "Incorrect Eqn."
    ]["Relative Error (\%)"]
    == 0.0,
)

# Incorrect Eqn. reduces 0.58 to 0% relative error.
print(
    f"Incorrect Eqn. reduces {fraction_relative_error_zero_under_incorrect_eqn} to 0% relative error."
)

models_parameters_columns_colors = {
    "Reported": plt.cm.viridis(0.0 / 3.0),
    "Incorrect Eqn.": plt.cm.viridis(1.0 / 3.0),
    "Correct Eqn.": plt.cm.viridis(2.0 / 3.0),
}

plt.close()
g = sns.relplot(
    data=chinchilla_parameters_relative_error_tall_df,
    x="Reported Parameters",
    y="Relative Error (\%)",
    hue="Equation",
    palette=models_parameters_columns_colors,
    col="Equation",
    col_order=["Correct Eqn.", "Incorrect Eqn.", "Reported"],
    style="Relative Error Is Zero",
    size="Relative Error Is Zero",
    sizes={False: 50, True: 25},
    legend=False,
    linewidth=0,
)
g.set(xscale="log", xlabel="Reported Model Parameters")
g.set_titles(col_template="{col_name}")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=relative_error_x=reported_parameters_hue=equation",
)
# plt.show()

plt.close()
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=chinchilla_parameters_df,
    x="Reported Parameters",
    y="Correct Eqn. Parameters",
    label="Correct Eqn.",
)
g = sns.scatterplot(
    data=chinchilla_parameters_df,
    x="Reported Parameters",
    y="Incorrect Eqn. Parameters",
    label="Incorrect Eqn.",
)
plt.plot(
    [3.16e7, 3.16e10],
    [3.16e7, 3.16e10],
    color="black",
    linestyle="--",
)
g.set(
    xlim=(3.16e7, 3.16e10),
    ylim=(3.16e7, 3.16e10),
    xlabel="Reported Parameters",
    ylabel="Calculated Parameters",
    xscale="log",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=calculated_parameters_x=reported_parameters_hue=equation",
)
# plt.show()


print("Finished 00_correcting_parameters!")
