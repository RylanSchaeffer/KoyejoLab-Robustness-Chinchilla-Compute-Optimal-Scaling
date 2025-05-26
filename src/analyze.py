from autograd import grad
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from typing import Tuple

import src.epoch_research_chinchilla_fit


def compute_or_load_chinchilla_fit_dataframes(
    data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    chinchilla_fits_df_path = os.path.join(data_dir, "chinchilla_fits.csv")
    chinchilla_tokens_per_parameter_df_path = os.path.join(
        data_dir, "chinchilla_tokens_per_parameter.csv"
    )
    if os.path.exists(chinchilla_fits_df_path) and os.path.exists(
        chinchilla_tokens_per_parameter_df_path
    ):
        chinchilla_fits_df = pd.read_csv(chinchilla_fits_df_path)
        chinchilla_tokens_per_parameter_df = pd.read_csv(
            chinchilla_tokens_per_parameter_df_path
        )
        return chinchilla_fits_df, chinchilla_tokens_per_parameter_df

    training_df = load_epoch_research_svg_extracted_data_csv()
    training_df["rounded_ms"] = np.round(training_df["Model Size"] / 1e6).astype(int)
    chinchilla_parameters_df = load_chinchilla_model_parameters_csv()
    training_df = training_df.merge(
        chinchilla_parameters_df,
        left_on="rounded_ms",
        right_on="Reported Parameters (M)",
    )

    models_parameters_columns = [
        "Model Size",
        "Reported Parameters",
        "Incorrect Eqn. Parameters",
        "Correct Eqn. Parameters",
    ]
    parameter_labels = ["A", "B", "E", "alpha", "beta"]
    nr_of_models_excluded = 5
    true_params = np.array([6.0073404, 6.0179186, 0.5267228, 0.33917084, 0.2849083])
    init_params = true_params.tolist()

    # Determine compute-optimal number of tokens per parameter.
    compute_thresholds = 10 ** np.arange(18, 28, 0.05)
    conf_int_percentile = 80
    low, high = (100 - conf_int_percentile) / 2, 100 - (100 - conf_int_percentile) / 2
    chinchilla_fits_data, chinchilla_tokens_per_parameter_data = [], []

    for models_parameters_column in models_parameters_columns:

        # N = training_df["Model Size"].values
        N = training_df[models_parameters_column].values
        D = training_df["Training Tokens"].values
        losses = training_df["loss"].values

        # Compute standard errors.
        # Set up the grid for initial parameter values
        param_list = []
        # bootstraps = 100
        bootstraps = 4000
        sorted_losses = sorted(losses)
        indices = [
            i
            for i in range(len(N))
            if losses[i] < sorted_losses[-nr_of_models_excluded]
        ]
        np.random.seed(42)
        random_indices = [
            np.random.choice(indices, size=len(indices), replace=True)
            for _ in range(bootstraps)
        ]
        for num, indices in enumerate(random_indices):
            best_loss = np.inf
            best_params = None

            result = minimize(
                src.epoch_research_chinchilla_fit.huber_loss_objective,
                np.array(init_params),
                args=(N[indices], D[indices], losses[indices]),
                jac=grad(src.epoch_research_chinchilla_fit.huber_loss_objective),
                method="BFGS",
            )

            # best_loss = result.fun
            # best_params = result.x
            # # print(f"New best loss: {best_loss}")
            # # print(f"Best params: {best_params}")

            if num % 100 == 99:
                print("Bootstrap step %d completed" % (num + 1))

            param_list.append(result.x)

        param_list = np.array(param_list)
        cov_matrix = np.cov(np.transpose(param_list))
        param_list_untransformed = src.epoch_research_chinchilla_fit.untransform_params(
            param_list
        )
        cov_matrix_untransformed = np.cov(np.transpose(param_list_untransformed))
        standard_errors_untransformed = np.sqrt(
            np.diag(cov_matrix_untransformed[:5, :5])
        )

        # Fit the parameters.
        indices = (
            list(range(len(N)))
            if nr_of_models_excluded == 0
            else [
                i
                for i in range(len(N))
                if losses[i] < sorted(losses)[-nr_of_models_excluded]
            ]
        )
        result = minimize(
            src.epoch_research_chinchilla_fit.objective,
            np.array(init_params + [0]),
            args=(N[indices], D[indices], losses[indices]),
            method="BFGS",
            jac=grad(src.epoch_research_chinchilla_fit.objective),
        )

        # Print the parameters.
        estimated_params = result.x[:5]
        estimated_params_untransformed = (
            src.epoch_research_chinchilla_fit.untransform_params(estimated_params)
        )
        print("Models' Parameters Column: ", models_parameters_column)
        # print(estimated_params_untransformed)
        for index, label in enumerate(parameter_labels):
            print(
                "%s: %.3f (%.3f)"
                % (
                    label,
                    estimated_params_untransformed[index],
                    standard_errors_untransformed[index],
                )
            )

        fit_data = {
            f"{k}_fit": v
            for k, v in zip(parameter_labels, estimated_params_untransformed)
        }
        fit_data.update(
            {
                f"{k}_se": v
                for k, v in zip(parameter_labels, standard_errors_untransformed)
            }
        )
        fit_data = pd.DataFrame(fit_data, index=[models_parameters_column])
        chinchilla_fits_data.append(fit_data)

        D_N_ratio_conf_int = [[], [], []]
        D_N_ratios = []
        chinchilla_D_N_ratio = []

        compute_loss_factors = []

        simulated_params_list = multivariate_normal.rvs(
            mean=estimated_params, cov=cov_matrix[:5, :5], size=10000
        )

        for threshold in compute_thresholds:
            D_N_ratio = []
            compute_loss_factor = []

            a_low = 0.454
            a_high = 0.455
            a_mid = np.mean([a_low, a_high])

            N_true_opt, D_true_opt = (
                src.epoch_research_chinchilla_fit.compute_optimal_allocation_from_shares(
                    threshold, src.epoch_research_chinchilla_fit.G(true_params), a_mid
                )
            )
            D_N_true_ratio = D_true_opt / N_true_opt

            for simulated_params in simulated_params_list:
                N_opt, D_opt = (
                    src.epoch_research_chinchilla_fit.compute_optimal_allocation(
                        threshold, simulated_params
                    )
                )
                D_N_ratio.append(D_opt / N_opt)

                loss_achieved_by_chinchilla = (
                    src.epoch_research_chinchilla_fit.scaling_law_reducible(
                        N_true_opt, D_true_opt, simulated_params
                    )
                )
                compute_needed_for_loss = src.epoch_research_chinchilla_fit.optimal_compute_from_reducible_loss(
                    loss_achieved_by_chinchilla, simulated_params
                )

                compute_loss_factor.append(threshold / compute_needed_for_loss)

            D_N_ratio_conf_int[0].append(np.percentile(D_N_ratio, low))
            D_N_ratio_conf_int[1].append(np.median(D_N_ratio))
            D_N_ratio_conf_int[2].append(np.percentile(D_N_ratio, high))

            chinchilla_D_N_ratio.append(D_N_true_ratio)

            D_N_ratios.append(D_N_ratio)
            compute_loss_factors.append(compute_loss_factor)

        chinchilla_tokens_per_parameter_data.append(
            pd.DataFrame.from_dict(
                {
                    models_parameters_column + " Low": D_N_ratio_conf_int[0],
                    models_parameters_column + " Median": D_N_ratio_conf_int[1],
                    models_parameters_column + " High": D_N_ratio_conf_int[2],
                },
            )
        )
        print("\n\n")

    chinchilla_fits_df = pd.concat(chinchilla_fits_data).T
    chinchilla_fits_df.to_csv(chinchilla_fits_df_path, index=False)
    chinchilla_tokens_per_parameter_df = pd.concat(
        chinchilla_tokens_per_parameter_data, axis=1
    )
    chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"] = compute_thresholds
    chinchilla_tokens_per_parameter_df.to_csv(
        chinchilla_tokens_per_parameter_df_path, index=False
    )
    return chinchilla_fits_df, chinchilla_tokens_per_parameter_df


def load_chinchilla_model_parameters_csv() -> pd.DataFrame:
    chinchilla_parameters_df = pd.read_csv("data/chinchilla_model_parameters.csv")
    chinchilla_parameters_df["Reported Parameters"] = (
        1e6 * chinchilla_parameters_df["Reported Parameters (M)"]
    )
    chinchilla_parameters_df["Incorrect Eqn. Parameters"] = (
        1e6
        * chinchilla_parameters_df[
            "Tied (Un)Embed, No Gating, Incorrect Attn Params Prefactor"
        ]
    )
    chinchilla_parameters_df["Correct Eqn. Parameters"] = (
        1e6 * chinchilla_parameters_df["Tied (Un)Embed, No Gating"]
    )
    chinchilla_parameters_df["Relative Error Correct Eqn."] = (
        100.0
        * (
            chinchilla_parameters_df["Reported Parameters"]
            - chinchilla_parameters_df["Correct Eqn. Parameters"]
        )
        / chinchilla_parameters_df["Reported Parameters"]
    )
    chinchilla_parameters_df["Relative Error Incorrect Eqn."] = (
        100.0
        * (
            chinchilla_parameters_df["Reported Parameters"]
            - chinchilla_parameters_df["Incorrect Eqn. Parameters"]
        )
        / chinchilla_parameters_df["Reported Parameters"]
    )
    return chinchilla_parameters_df


def load_epoch_research_svg_extracted_data_csv() -> pd.DataFrame:
    # Partially adapted from https://github.com/epoch-research/analyzing-chinchilla/blob/main/data_analysis.ipynb.
    training_df = pd.read_csv("data/epoch_research_svg_extracted_data.csv")
    training_df["Training Tokens"] = training_df["Training FLOP"] / (
        6.0 * training_df["Model Size"]
    )
    training_df = (
        training_df[["Model Size", "Training Tokens", "Training FLOP", "loss"]]
        .dropna()
        .copy()
    )
    training_df["d_n_ratio"] = (
        training_df["Training Tokens"] / training_df["Model Size"]
    )
    return training_df


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
