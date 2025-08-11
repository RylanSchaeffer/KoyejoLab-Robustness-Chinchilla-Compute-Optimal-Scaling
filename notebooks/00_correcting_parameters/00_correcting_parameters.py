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
        "Relative Error Correct Eqn.": "Correct Formula",
        "Relative Error Incorrect Eqn.": "Best Fit Formula",
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
        chinchilla_parameters_relative_error_tall_df["Equation"] == "Best Fit Formula"
    ]["Relative Error (\%)"]
    == 0.0,
)

# Best Fit Formula reduces 0.58 to 0% relative error.
print(
    f"Best Fit Formula reduces {fraction_relative_error_zero_under_incorrect_eqn} to 0% relative error."
)

for equation, subset_df in chinchilla_parameters_relative_error_tall_df.groupby(
    ["Equation"]
):
    print("Equation: ", equation[0])
    print(subset_df["Relative Error (\%)"].describe())
    print("\n\n")

models_parameters_columns_colors = {
    "Reported": plt.cm.viridis(0.0 / 3.0),
    "Best Fit Formula": plt.cm.viridis(1.0 / 3.0),
    "Correct Formula": plt.cm.viridis(2.0 / 3.0),
}
plt.close()
g = sns.relplot(
    data=chinchilla_parameters_relative_error_tall_df,
    kind="scatter",
    x="Reported Parameters",
    y="Relative Error (\%)",
    hue="Equation",
    palette=models_parameters_columns_colors,
    col="Equation",
    col_order=["Correct Formula", "Best Fit Formula"],
    # style="Relative Error Is Zero",
    # size="Relative Error Is Zero",
    # sizes={False: 50, True: 25},
    legend=False,
    linewidth=0,
)
g.set(xscale="log", xlabel="Reported Model Parameters", ylim=(-5, 20))
g.set_titles(col_template="{col_name}")
for ax in g.axes.flat:
    ax.axhline(0, color="k", linestyle="--")
# correct_attn_eq = r"$\text{Attn Params} = n\_layers \cdot (4 \cdot d\_model \cdot kv\_size \cdot n\_heads)$"
# New string with bold title and variables
correct_attn_eq = r"$\text{\textbf{Attn Params}} = \mathbf{n_{layers}} \cdot (\mathbf{4} \cdot \mathbf{d_{model}} \cdot \mathbf{kv_{size}} \cdot \mathbf{n_{heads}})$"
ax = g.axes.flat[0]
ax.text(
    0.05,
    0.9,
    correct_attn_eq,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="left",
)
# For some reason, \textcolor{red} doesn't work. I have to split the equation apart.
ax = g.axes.flat[1]
ax.text(
    0.05,
    0.9,
    r"$\text{\textbf{Attn Params}} = \mathbf{n_{layers}} \cdot ($",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="left",
)
ax.text(
    0.5225,
    0.90,
    r"$\mathbf{5}$",
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment="bottom",
    horizontalalignment="left",
    color="red",
)
ax.text(
    0.56,
    0.905,
    r"$ \cdot \; \mathbf{d_{model}} \cdot \mathbf{kv_{size}} \cdot \mathbf{n_{heads}})$",
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="left",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=relative_error_x=reported_parameters_hue=equation",
)
plt.show()

# plt.close()
# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     data=chinchilla_parameters_df,
#     x="Reported Parameters",
#     y="Correct Eqn. Parameters",
#     label="Correct Formula",
# )
# g = sns.scatterplot(
#     data=chinchilla_parameters_df,
#     x="Reported Parameters",
#     y="Best Fit Formula Parameters",
#     label="Best Fit Formula",
# )
# plt.plot(
#     [3.16e7, 3.16e10],
#     [3.16e7, 3.16e10],
#     color="black",
#     linestyle="--",
# )
# g.set(
#     xlim=(3.16e7, 3.16e10),
#     ylim=(3.16e7, 3.16e10),
#     xlabel="Reported Parameters",
#     ylabel="Calculated Parameters",
#     xscale="log",
#     yscale="log",
# )
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=calculated_parameters_x=reported_parameters_hue=equation",
# )
# # plt.show()


print("Finished 00_correcting_parameters!")
