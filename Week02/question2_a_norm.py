import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from tabulate import tabulate

data = pd.read_csv('problem2.csv')

x = data['x']
y = data['y']

def MLE_Norm(params, x, y):
    yhat = params[0] + params[1] * x  # predictions
    negLL = -1 * np.sum(norm.logpdf(y, yhat, params[2]))
    return negLL

results_norm = minimize(MLE_Norm, x0=(1, 1, 1), args=(x, y))

# Print the MLE estimated parameters using tabulate
table_params = tabulate(
    [
        ["Intercept", results_norm.x[0]],
        ["Slope", results_norm.x[1]],
        ["Standard Deviation", results_norm.x[2]],
    ],
    headers=["Parameter", "MLE Estimate"],
    tablefmt='pretty'
)

print(table_params)

data = pd.read_csv('problem2.csv')

x = data['x']
y = data['y']

def MLE_Norm(params, x, y):
    yhat = params[0] + params[1] * x  # predictions
    negLL = -1 * np.sum(norm.logpdf(y, yhat, params[2]))
    return negLL

results_norm = minimize(MLE_Norm, x0=(1, 1, 1), args=(x, y))
print("MLE estimated parameters for Normal distribution:", results_norm.x)

def log_likelihood(params, data):
    mean, std = params
    return -np.sum(norm.logpdf(data, loc=mean, scale=std))

initial_guess = [1, 1]
result = minimize(log_likelihood, initial_guess, args=(y,), method='Nelder-Mead')

# MLE estimated standard deviation
std_deviation_estimate = result.x[1]

# Calculate Log Likelihood, AIC, and BIC
log_likelihood_value = -result.fun
n_params = len(result.x)
n_observations = len(y)
aic = 2 * n_params - 2 * log_likelihood_value
bic = n_params * np.log(n_observations) - 2 * log_likelihood_value

# Print the results using tabulate with a specified table name "Norm"
table = tabulate(
    [
        ["MLE estimated standard deviation", std_deviation_estimate],
        ["Log Likelihood", log_likelihood_value],
        ["AIC", aic],
        ["BIC", bic],
    ],
    headers=["Metric", "Value"],
    tablefmt='pretty',
    showindex=False,
)

# Print the table with the specified name
print("\nNorm Table:")
print(table)

