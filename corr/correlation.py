import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sentiment_df = pd.read_csv('/Users/adi/zProjects/Research/AAPL_SENTIMENT.csv')
df = pd.read_csv('/Users/adi/zProjects/Research/AAPL_PRICE2.csv')

df['DATE'] = pd.to_datetime(df['DATE'])

start_date = min(df['DATE'])
df['days_from_start'] = (df['DATE'] - start_date).dt.days

def polynomial_regression(x, *coefficients):
    return sum(coefficients[i] * x**i for i in range(len(coefficients)))

x_data = df['days_from_start']
y_data = df['Close']
degree = 20
coefficients = np.polyfit(x_data, y_data, degree)
x_fit = np.linspace(min(x_data), max(x_data), 790)
y_fit = np.polyval(coefficients, x_fit)

x_fit_dates = start_date + pd.to_timedelta(x_fit, unit='D')

derivative_coefficients = np.polyder(coefficients)

y_derivative = np.polyval(derivative_coefficients, x_fit)

correlation, p_value = pearsonr(sentiment_df['CleanS'], df['Close'])
print("Correlation:", correlation)
print("P-value:", p_value)
