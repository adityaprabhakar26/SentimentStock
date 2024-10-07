import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/adi/ComputerScience/zProjects/Research/Weekly_Averages.csv')

data['Sentiment_Change'] = data['Weekly_Avg_Sentiment'].diff()
data['Price_Change'] = data['Weekly_Avg_Close_Price'].diff()

data['Next_Week_Sentiment_Change'] = data['Sentiment_Change'].shift(-1)

data = data.dropna()

window_size = 7
X, y = [], []
for i in range(len(data) - window_size):
    X.append(data['Next_Week_Sentiment_Change'][i:i + window_size])  
    y.append(data['Price_Change'][i + window_size])  

X, y = np.array(X), np.array(y)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=1))  
optimizer = RMSprop(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, validation_split=0.2)

y_pred = model.predict(X_test)

y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

actual_derivatives = np.gradient(y_test_original, axis=0)
predicted_derivatives = np.gradient(y_pred_original, axis=0)

correct = 0
wrong = 0

for i in range(len(actual_derivatives)):
    if np.sign(actual_derivatives[i]) == np.sign(predicted_derivatives[i]):
        correct += 1
    else:
        wrong += 1

total_data_points = correct+wrong
percent_correct = (correct / total_data_points) * 100
percent_wrong = (wrong / total_data_points) * 100

print("Percent Correct:", percent_correct)
print("Percent Wrong:", percent_wrong)
print(correct)
print(wrong)
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)
n = len(X_test)  
p = X_test.shape[1]  
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)


def calculate_aic(n, mse, k):
    aic = n * np.log(mse) + 2 * k
    return aic

k = model.count_params()  
aic = calculate_aic(len(y_test), mse, k)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(y_test_original, y_pred_original)
ols_score = regressor.score(y_test_original, y_pred_original)
print("OLS Score:", ols_score)

print("AIC:", aic)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Adjusted R-squared:", adjusted_r2)

plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual Values', color='blue')
plt.plot(y_pred_original, label='Predictions', color='red')
plt.xlabel('Weeks')
plt.ylabel('Price Change')
plt.title('Actual Price Change vs. LSTM Predictions')
plt.legend()
plt.show()
