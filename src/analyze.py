from autograd import grad
from functools import partial
import multiprocessing
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from typing import List, Tuple

import src.epoch_research_chinchilla_fit


def compute_chinchilla_fit_bootstrap_iteration(args):
    """
    Performs the optimization for a single bootstrap sample.
    It takes a single 'args' tuple to be compatible with Pool.map.
    """
    # Unpack the arguments
    indices, init_params, all_parameters, all_tokens, all_losses = args

    result = minimize(
        src.epoch_research_chinchilla_fit.huber_loss_objective,
        np.array(init_params),
        args=(all_parameters[indices], all_tokens[indices], all_losses[indices]),
        jac=grad(src.epoch_research_chinchilla_fit.huber_loss_objective),
        method="BFGS",
        options={"maxiter": 10000},
    )
    return result.x


def compute_chinchilla_fit_dataframes(
    parameters: np.ndarray,
    tokens: np.ndarray,
    losses: np.ndarray,
    bootstraps: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    parameter_labels = ["A", "B", "E", "alpha", "beta"]
    nr_of_models_excluded = 5
    true_params = np.array([6.0073404, 6.0179186, 0.5267228, 0.33917084, 0.2849083])
    init_params = true_params.tolist()

    # Determine compute-optimal number of tokens per parameter.
    compute_thresholds = 10 ** np.arange(18, 28, 0.05)
    conf_int_percentile = 80
    low, high = (100 - conf_int_percentile) / 2, 100 - (100 - conf_int_percentile) / 2

    # Compute standard errors.
    # Set up the grid for initial parameter values
    param_array = []
    sorted_losses = sorted(losses)
    indices = [
        i
        for i in range(len(parameters))
        if losses[i] < sorted_losses[-nr_of_models_excluded]
    ]
    random_indices = [
        np.random.choice(indices, size=len(indices), replace=True)
        for _ in range(bootstraps)
    ]

    # Prepare arguments for each parallel task
    task_args = [
        (indices, init_params, parameters, tokens, losses) for indices in random_indices
    ]

    with multiprocessing.Pool() as pool:
        # map() distributes the 'task_args' across the worker processes
        # and collects the results in a list.
        param_list = pool.map(compute_chinchilla_fit_bootstrap_iteration, task_args)

    # for num, indices in enumerate(random_indices):
    #     best_loss = np.inf
    #     best_params = None
    #
    #     result = minimize(
    #         src.epoch_research_chinchilla_fit.huber_loss_objective,
    #         np.array(init_params),
    #         args=(parameters[indices], tokens[indices], losses[indices]),
    #         jac=grad(src.epoch_research_chinchilla_fit.huber_loss_objective),
    #         method="BFGS",
    #     )
    #
    #     # best_loss = result.fun
    #     # best_params = result.x
    #     # # print(f"New best loss: {best_loss}")
    #     # # print(f"Best params: {best_params}")
    #
    #     if num % 100 == 99:
    #         print("Bootstrap step %d completed" % (num + 1))
    #
    #     param_array.append(result.x)

    param_array = np.array(param_list)
    cov_matrix = np.cov(np.transpose(param_array))
    param_list_untransformed = src.epoch_research_chinchilla_fit.untransform_params(
        param_array
    )
    cov_matrix_untransformed = np.cov(np.transpose(param_list_untransformed))
    standard_errors_untransformed = np.sqrt(np.diag(cov_matrix_untransformed[:5, :5]))

    # Fit the parameters.
    indices = (
        list(range(len(parameters)))
        if nr_of_models_excluded == 0
        else [
            i
            for i in range(len(parameters))
            if losses[i] < sorted(losses)[-nr_of_models_excluded]
        ]
    )
    result = minimize(
        src.epoch_research_chinchilla_fit.objective,
        np.array(init_params + [0]),
        args=(parameters[indices], tokens[indices], losses[indices]),
        method="BFGS",
        jac=grad(src.epoch_research_chinchilla_fit.objective),
    )

    # Print the parameters.
    estimated_params = result.x[:5]
    estimated_params_untransformed = (
        src.epoch_research_chinchilla_fit.untransform_params(estimated_params)
    )
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
    print("\n\n")

    fit_data = {
        f"{k}_fit": v for k, v in zip(parameter_labels, estimated_params_untransformed)
    }
    fit_data.update(
        {f"{k}_se": v for k, v in zip(parameter_labels, standard_errors_untransformed)}
    )
    single_fit_df = pd.DataFrame(fit_data, index=["will_be_overridden"])

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
            N_opt, D_opt = src.epoch_research_chinchilla_fit.compute_optimal_allocation(
                threshold, simulated_params
            )
            D_N_ratio.append(D_opt / N_opt)

            loss_achieved_by_chinchilla = (
                src.epoch_research_chinchilla_fit.scaling_law_reducible(
                    N_true_opt, D_true_opt, simulated_params
                )
            )
            compute_needed_for_loss = (
                src.epoch_research_chinchilla_fit.optimal_compute_from_reducible_loss(
                    loss_achieved_by_chinchilla, simulated_params
                )
            )

            compute_loss_factor.append(threshold / compute_needed_for_loss)

        D_N_ratio_conf_int[0].append(np.percentile(D_N_ratio, low))
        D_N_ratio_conf_int[1].append(np.median(D_N_ratio))
        D_N_ratio_conf_int[2].append(np.percentile(D_N_ratio, high))

        chinchilla_D_N_ratio.append(D_N_true_ratio)

        D_N_ratios.append(D_N_ratio)
        compute_loss_factors.append(compute_loss_factor)

    single_tokens_per_parameter_df = pd.DataFrame.from_dict(
        {
            "Low": D_N_ratio_conf_int[0],
            "Median": D_N_ratio_conf_int[1],
            "High": D_N_ratio_conf_int[2],
        },
    )
    return single_fit_df, single_tokens_per_parameter_df


def compute_or_load_chinchilla_fit_dataframes(
    data_dir: str,
    models_parameters_columns: List[str],
    refresh: bool = False,
    bootstraps: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    chinchilla_fits_df_path = os.path.join(data_dir, "chinchilla_fits.csv")
    chinchilla_tokens_per_parameter_df_path = os.path.join(
        data_dir, "chinchilla_tokens_per_parameter.csv"
    )
    if (
        os.path.exists(chinchilla_fits_df_path)
        and os.path.exists(chinchilla_tokens_per_parameter_df_path)
        and not refresh
    ):
        chinchilla_fits_df = pd.read_csv(chinchilla_fits_df_path, index_col=0)
        chinchilla_tokens_per_parameter_df = pd.read_csv(
            chinchilla_tokens_per_parameter_df_path
        )
        # Confirm that our loaded data has the columns we expect.
        for models_parameters_column in models_parameters_columns:
            assert models_parameters_column in chinchilla_fits_df.columns
            for val in ["Low", "Median", "High"]:
                assert (
                    f"{models_parameters_column} {val}"
                    in chinchilla_tokens_per_parameter_df.columns
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

    chinchilla_fits_list_of_dfs, chinchilla_tokens_per_parameter_list_of_dfs = [], []

    for models_parameters_column in models_parameters_columns:
        single_fit_df, single_tokens_per_parameter_df = (
            compute_chinchilla_fit_dataframes(
                parameters=training_df[models_parameters_column].values,
                tokens=training_df["Training Tokens"].values,
                losses=training_df["loss"].values,
                bootstraps=bootstraps,
            )
        )
        # Overwrite the columns to additionally specify
        single_fit_df.index = [models_parameters_column]
        single_tokens_per_parameter_df.columns = [
            f"{models_parameters_column} {col}"
            for col in single_tokens_per_parameter_df.columns
        ]
        chinchilla_fits_list_of_dfs.append(single_fit_df)
        chinchilla_tokens_per_parameter_list_of_dfs.append(
            single_tokens_per_parameter_df
        )

    chinchilla_fits_df = pd.concat(chinchilla_fits_list_of_dfs).T
    chinchilla_fits_df.to_csv(chinchilla_fits_df_path)
    chinchilla_tokens_per_parameter_df = pd.concat(
        chinchilla_tokens_per_parameter_list_of_dfs, axis=1
    )
    chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"] = 10 ** np.arange(
        18, 28, 0.05
    )
    chinchilla_tokens_per_parameter_df.to_csv(
        chinchilla_tokens_per_parameter_df_path, index=False
    )
    return chinchilla_fits_df, chinchilla_tokens_per_parameter_df


def compute_or_load_chinchilla_robustness_fit_dataframes(
    data_dir: str,
    models_parameters_columns: List[str],
    refresh: bool = False,
    bootstraps: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    chinchilla_fits_df_path = os.path.join(data_dir, "chinchilla_robustness_fits.csv")
    chinchilla_tokens_per_parameter_df_path = os.path.join(
        data_dir, "chinchilla_robustness_tokens_per_parameter.csv"
    )
    if (
        os.path.exists(chinchilla_fits_df_path)
        and os.path.exists(chinchilla_tokens_per_parameter_df_path)
        and not refresh
    ):
        chinchilla_fits_df = pd.read_csv(chinchilla_fits_df_path, index_col=0)
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

    parameter_transformation_fns = (
        {
            "Identity": lambda params: params,
        }
        | {
            f"Systematic Bias_{slope}": partial(
                parameter_transformation_systematic_bias,
                slope=slope,
            )
            for slope in np.logspace(-0.5, 0.5, 11)
        }
        | {
            f"Log Normal Noise_{sigma}": partial(
                parameter_transformation_log_normal_noise,
                sigma=sigma,
                repeat_idx=repeat_idx,
            )
            for sigma in np.logspace(-2, 2, num=11)
            for repeat_idx in np.arange(1)
        }
        | {
            f"Multiplicative Constant_{c}": partial(
                parameter_transformation_multiplicative_constant, c=c
            )
            for c in np.logspace(-3, 3, num=11)
        }
    )

    chinchilla_fits_list_of_dfs, chinchilla_tokens_per_parameter_list_of_dfs = [], []
    for models_parameters_column in models_parameters_columns:
        parameters = training_df[models_parameters_column].values
        for param_trans_fn_name, param_trans_fn in parameter_transformation_fns.items():
            single_fit_df, single_tokens_per_parameter_df = (
                compute_chinchilla_fit_dataframes(
                    parameters=param_trans_fn(parameters.copy()),
                    tokens=training_df["Training Tokens"].values,
                    losses=training_df["loss"].values,
                    bootstraps=bootstraps,
                )
            )
            # Overwrite the columns to additionally specify
            single_fit_df.index = [f"{models_parameters_column}_{param_trans_fn_name}"]
            single_tokens_per_parameter_df.columns = [
                f"{models_parameters_column}_{param_trans_fn_name}_{col}"
                for col in single_tokens_per_parameter_df.columns
            ]
            chinchilla_fits_list_of_dfs.append(single_fit_df)
            chinchilla_tokens_per_parameter_list_of_dfs.append(
                single_tokens_per_parameter_df
            )

    chinchilla_fits_df = pd.concat(chinchilla_fits_list_of_dfs).T
    chinchilla_fits_df.to_csv(chinchilla_fits_df_path)
    chinchilla_tokens_per_parameter_df = pd.concat(
        chinchilla_tokens_per_parameter_list_of_dfs, axis=1
    )
    chinchilla_tokens_per_parameter_df["Training Compute (FLOP)"] = 10 ** np.arange(
        18, 28, 0.05
    )
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
    chinchilla_parameters_df["Relative Error Reported Parameters"] = 0.0
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


# This is a gnarly implementation, but I just want something that works quickly and
# I don't want to have to refactor.
def parameter_transformation_log_normal_noise(
    parameters: np.ndarray, sigma: float, repeat_idx: int
) -> np.ndarray:
    return parameters * np.exp(
        np.random.normal(size=parameters.shape, loc=0.0, scale=sigma)
    )


def parameter_transformation_multiplicative_constant(
    parameters: np.ndarray, c: float
) -> np.ndarray:
    return parameters * c


def parameter_transformation_systematic_bias(
    parameters: np.ndarray, slope: float
) -> np.ndarray:
    log_parameters = np.log10(parameters)
    mean_log_parameters = np.mean(log_parameters)
    new_parameters = np.power(
        10.0, slope * (log_parameters - mean_log_parameters) + mean_log_parameters
    )
    return new_parameters


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
