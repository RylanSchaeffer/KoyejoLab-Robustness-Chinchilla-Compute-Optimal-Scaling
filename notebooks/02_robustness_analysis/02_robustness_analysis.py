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

print("Finished 02_robustness_analysis!")
