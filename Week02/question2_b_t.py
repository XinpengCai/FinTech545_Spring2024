import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t
from tabulate import tabulate

# Load data from 'problem2.csv'
data = pd.read_csv('problem2.csv')

# Extract 'x' and 'y' columns from the data
x = data['x']
y = data['y']

def log_likelihood_t(params, data):
    # Define a log-likelihood function for a t-distribution
    mean, std, df = params
    return -np.sum(t.logpdf(data, df, loc=mean, scale=std))

# Initial guess for the parameters
initial_guess = [0, 1, 1]

# Use Scipy's minimize function for Maximum Likelihood Estimation (MLE)
result = minimize(log_likelihood_t, initial_guess, args=(data,), method='Nelder-Mead')

# Calculate Log Likelihood
log_likelihood = -result.fun

# Calculate AIC
n_params = len(result.x)
n_observations = len(y)
aic = 2 * n_params - 2 * log_likelihood

# Calculate BIC
bic = n_params * np.log(n_observations) - 2 * log_likelihood

# Calculate Residual Sum of Squares (RSS)
rss = np.sum(result.fun**2)

# Calculate Total Sum of Squares (TSS)
tss = np.sum((y - np.mean(y))**2)

# Calculate R-squared
rsquared = 1 - rss / tss

# Calculate the number of predictors (including intercept)
n_predictors = len(result.x)

# Calculate the number of observations
n_observations = len(y)

# Calculate the degrees of freedom for the model
df_model = n_predictors - 1

# Calculate the degrees of freedom for the residuals
df_residuals = n_observations - n_predictors

# Calculate Adjusted R-squared
adjusted_rsquared = 1 - (rss / df_residuals) / (tss / df_model)

# Output the Adjusted R-squared
print("Adjusted R-squared:", adjusted_rsquared)

# Output the results using tabulate with a specified table name "T-distribution"
table = tabulate(
    [
        ["MLE estimate of mean", result.x[0]],
        ["MLE estimate of standard deviation", result.x[1]],
        ["MLE estimate of degrees of freedom", result.x[2]],
        ["Log Likelihood", log_likelihood],
        ["AIC", aic],
        ["BIC", bic],
    ],
    headers=["Metric", "Value"],
    tablefmt='pretty',
    showindex=False,
)

# Print the table with the specified name
print("\nT-distribution Table:")
print(table)


