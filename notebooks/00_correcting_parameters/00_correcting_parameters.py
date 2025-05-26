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

plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=chinchilla_parameters_df,
    x="Reported Parameters",
    y="Relative Error Correct Eqn.",
    label="Correct Eqn.",
)
sns.scatterplot(
    data=chinchilla_parameters_df,
    x="Reported Parameters",
    y="Relative Error Incorrect Eqn.",
    label="Incorrect Eqn.",
)
g.set(
    xscale="log",
    ylabel=r"$100 * \frac{\text{Reported Params - Calculated Params}}{\text{Reported Params}} $",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=relative_error_x=reported_parameters_hue=equation",
)
plt.show()

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
plt.show()


print("Finished 00_correcting_parameters!")
