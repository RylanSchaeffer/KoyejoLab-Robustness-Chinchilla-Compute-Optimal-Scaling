# Adapted from https://github.com/epoch-research/analyzing-chinchilla/blob/main/data_analysis.ipynb.
import autograd.numpy as np
from autograd.scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import erf

# true_params = np.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])
true_params = np.array([6.0073404, 6.0179186, 0.5267228, 0.33917084, 0.2849083])
true_params_rounded = np.array([np.log(406.4), np.log(410.7), np.log(1.69), 0.34, 0.28])


# Define the log-sum-exp function
def log_sum_exp(a, b, e, alpha, beta, N, D):
    return np.log(
        np.exp(a - alpha * np.log(N)) + np.exp(b - beta * np.log(D)) + np.exp(e)
    )


# Define the Huber loss function
def custom_huber_loss(y_true, y_pred, delta=1e-3):
    # Calculate the difference
    diff = y_true - y_pred
    # Calculate the condition for Huber loss
    cond = np.abs(diff) <= delta
    # Apply Huber loss formula
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)


def huber_normalizing_factor(delta=1e-3):
    return (
        np.sqrt(2 * np.pi) * (1 - 2 * norm.sf(delta))
        + 2 * np.exp(-0.5 * delta**2) / delta
    )


def huber_logpdf(x, delta=1e-3, loc=0, scale=1):
    x = (x - loc) / scale

    cond = np.abs(x) <= delta
    loss = np.where(cond, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))
    return -loss - np.log(huber_normalizing_factor(delta=delta)) - np.log(scale)


def huber_pdf(x, delta=1e-3, loc=0, scale=1):
    return np.exp(huber_logpdf(x, delta=delta, loc=loc, scale=scale))


# Define the objective function to be minimized
def objective(params, N, D, losses):
    a, b, e, alpha, beta, sigma = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -np.sum(
        huber_logpdf(np.log(losses), loc=predictions, scale=np.exp(sigma), delta=1e-3)
    )
    # return custom_huber_loss(np.log(losses), predictions, delta=1e-3)


def scale_objective(sigma, params, N, D, losses):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -np.sum(
        huber_logpdf(np.log(losses), loc=predictions, scale=np.exp(sigma), delta=1e-3)
    )
    # return custom_huber_loss(np.log(losses), predictions, delta=1e-3)


def constant_term_objective(params, a, b, alpha, beta, N, D, losses):
    e, sigma = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -np.sum(
        huber_logpdf(np.log(losses), loc=predictions, scale=np.exp(sigma), delta=1e-3)
    )


def huber_loss_objective(params, N, D, losses):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return custom_huber_loss(np.log(losses), predictions, delta=1e-3)


# Define the parameter untransform
def untransform_params(param_array):
    if len(np.shape(param_array)) == 2:
        return np.hstack((np.exp(param_array[:, :3]), param_array[:, 3:]))
    else:
        return np.hstack((np.exp(param_array[:3]), param_array[3:]))


# Define the Huber loss function on residuals
def huber_loss(residuals, delta=1e-3):
    # Calculate the difference
    diff = residuals
    # Calculate the condition for Huber loss
    cond = np.abs(diff) <= delta
    # Apply Huber loss formula
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return loss


def scaling_law_reducible(N, D, params):
    a, b, e, alpha, beta = params
    A, B, E = np.exp([a, b, e])

    return A / N**alpha + B / D**beta


def G(params):
    a, b, e, alpha, beta = params
    A, B, E = np.exp([a, b, e])

    return ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))


def compute_optimal_allocation(compute, params):
    a, b, e, alpha, beta = params
    A, B, E = np.exp([a, b, e])

    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    a = beta / (alpha + beta)
    b = 1 - a

    return G * (compute / 6) ** a, G ** (-1) * (compute / 6) ** b


def compute_optimal_reducible_loss(compute, params):
    N_opt, D_opt = compute_optimal_allocation(compute, params)
    return scaling_law_reducible(N_opt, D_opt, params)


def optimal_compute_from_reducible_loss(loss, params):
    a, b, e, alpha, beta = params
    A, B, E = np.exp([a, b, e])

    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    a = beta / (alpha + beta)
    b = 1 - a

    return 6 * (loss / (G ** (-alpha) * A + G**beta * B)) ** (
        -(alpha + beta) / (alpha * beta)
    )


def compute_optimal_allocation_from_shares(compute, G, a):
    b = 1 - a
    return G * (compute / 6) ** a, G ** (-1) * (compute / 6) ** b


def ratio(params_and_tokens):
    params, tokens = params_and_tokens
    return tokens / params
