import pandas as pd
import matplotlib.pyplot as plt
sentiment_df = pd.read_csv('/Users/adi/ComputerScience/zProjects/Research/AAPL_SENTIMENT.csv')
price_df = pd.read_csv('/Users/adi/ComputerScience/zProjects/Research/AAPL_PRICE2.csv')

sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'])
price_df['DATE'] = pd.to_datetime(price_df['DATE'])

sentiment_df.set_index('created_at', inplace=True)
price_df.set_index('DATE', inplace=True)

sentiment_weekly_avg = sentiment_df['CleanS'].resample('M').mean()
price_weekly_avg = price_df['Close'].resample('M').mean()



plt.plot(sentiment_weekly_avg.index, sentiment_weekly_avg, label='Sentiment Score', color='blue')

plt.xlabel('Date')
plt.ylabel('Average Value')
plt.title('Weekly Average Sentiment Score and Close Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(price_weekly_avg.index, price_weekly_avg, label='Close Price', color='green')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title('Price and Date')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
