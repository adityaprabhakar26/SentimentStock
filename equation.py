import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

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

plt.figure(figsize=(10, 6))
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = np.polyval(coefficients, x_fit)

x_fit_dates = start_date + pd.to_timedelta(x_fit, unit='D')

plt.plot(x_fit_dates, y_fit, label='Fitted Polynomial', color='red')


plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Polynomial Regression Model')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

derivative_coefficients = np.polyder(coefficients)

y_derivative = np.polyval(derivative_coefficients, x_fit)

plt.plot(x_fit_dates, y_derivative, label='Derivative of Polynomial', color='green')


plt.xlabel('Date')
plt.ylabel('Derivative of Close Price')
plt.title('Derivative of Polynomial Regression Model')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
sentiment_df = pd.read_csv('/Users/adi/zProjects/Research/AAPL_SENTIMENT.csv')
sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'])

plt.figure(figsize=(10, 6))

plt.plot(sentiment_df['created_at'], sentiment_df['CleanS'], label='Sentiment', color='blue')

plt.xlabel('Date')
plt.ylabel('Sentiment Value')
plt.title('Sentiment Data')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
