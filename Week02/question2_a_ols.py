import pandas as pd
import numpy as np
import statsmodels.api as sm
from tabulate import tabulate

# Read the data
data = pd.read_csv('problem2.csv')

# Add a constant term to the independent variable
data['constant'] = 1

# Extract independent (x) and dependent (y) variables
x = data['x']
y = data['y']

# Create a design matrix with a constant term
X = data[['constant', 'x']]

# Fit the OLS (Ordinary Least Squares) model
est = sm.OLS(y, X)
model = est.fit()

# Calculate the standard deviation of OLS residuals
residuals = y - model.predict(X)
std_error = np.std(residuals)

# Print the standard deviation of OLS residuals
print("Standard Deviation of OLS Error:", std_error)

# Generate a tabulated string for the model summary
table = tabulate(model.summary().tables[1], headers='keys', tablefmt='pretty')

# Print the tabulated model summary
print(table)


