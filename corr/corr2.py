import pandas as pd

data = pd.read_csv('/Users/adi/zProjects/Research/Weekly_Averages.csv')

data['Sentiment_Change'] = data['Weekly_Avg_Sentiment'].diff()
data['Price_Change'] = data['Weekly_Avg_Close_Price'].diff()

t = 1  
data['ddd'] = data['Price_Change'].shift(-t)

data = data.dropna()

correlation = data['Sentiment_Change'].corr(data['ddd'])

print("Correlation between change in sentiment and change in price after {} week(s): {:.2f}".format(t, correlation))
