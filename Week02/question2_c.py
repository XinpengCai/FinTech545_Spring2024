import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, t, ttest_1samp, norm
import statsmodels.api as sm
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro
import pylab

data_x1 = pd.read_csv('problem2_x1.csv')


x1_observed = data_x1['x1'].values


data_full = pd.read_csv('problem2_x.csv')


x2_full = data_full['x2'].values


mean_vector = np.mean(data_full[['x1', 'x2']], axis=0)
covariance_matrix = np.cov(data_full[['x1', 'x2']], rowvar=False)


simulated_x2_values = []


for x1 in x1_observed:
 
    conditional_mean_x2 = mean_vector[1] + covariance_matrix[1, 0] / covariance_matrix[0, 0] * (x1 - mean_vector[0])
    conditional_variance_x2 = covariance_matrix[1, 1] - covariance_matrix[1, 0] / covariance_matrix[0, 0] * covariance_matrix[0, 1]


    simulated_x2_value = np.random.normal(loc=conditional_mean_x2, scale=np.sqrt(conditional_variance_x2))
    
  
    simulated_x2_values.append(simulated_x2_value)


plt.hist(simulated_x2_values, bins=30, density=True, alpha=0.5, color='blue', label='Simulated X2 Distribution')
plt.xlabel('x2')
plt.ylabel('Probability Density')
plt.title('Simulated Distribution of X2 given X1')
plt.legend()


statistic, p_value = shapiro(simulated_x2_values)
print(f'Shapiro-Wilk Test Statistic: {statistic}, p-value: {p_value}')


sm.qqplot(np.array(simulated_x2_values), line='s')
plt.title('Normal Q-Q Plot')
plt.show()