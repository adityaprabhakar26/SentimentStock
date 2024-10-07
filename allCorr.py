import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('/Users/adi/zProjects/Research/Weekly_Averages.csv')

correlation, p_value = pearsonr(df['Weekly_Avg_Sentiment'], df['Weekly_Avg_Close_Price'])

print("Correlation:", correlation)
print("P-value:", p_value)

df = pd.read_csv('/Users/adi/zProjects/Research/Monthly_Averages.csv')

correlation, p_value = pearsonr(df['Monthly_Avg_Sentiment'], df['Monthly_Avg_Close_Price'])

print("Correlation:", correlation)
print("P-value:", p_value)
