import pandas as pd
import numpy as np
from scipy.stats import moment

data = pd.read_csv('problem1.csv')
ts = data["x"]

mean_value = np.mean(data)
variance_value = moment(ts, moment=2)
skewness_value = moment(ts, moment=3)
kurtosis_value = moment(ts, moment=4)


print(f"Mean: {mean_value}")
print(f"Variance: {variance_value}")
print(f"Skewness: {skewness_value}")
print(f"Kurtosis: {kurtosis_value}")


