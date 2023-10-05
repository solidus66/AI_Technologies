import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('TkAgg')

data = pd.read_csv('C:/Users/solidus66/OneDrive/ВГУ/4 курс 1 сем/ТИИ/lab2/milk.csv')

model = SARIMAX(data['milk'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

forecast_steps = 8
forecast = results.get_forecast(steps=forecast_steps)

plt.figure(figsize=(12, 6))
plt.plot(data['milk'], label='Initial data', marker='o')
plt.plot(forecast.predicted_mean, label='Forecast', color='red', linestyle='dashed', marker='o')
plt.title('Monthly milk production forecast')
plt.xlabel('Month')
plt.ylabel('Milk Production')
plt.legend()
plt.show()

forecast_values = forecast.predicted_mean[-forecast_steps:]
forecast_values_with_index = forecast_values.reset_index()
forecast_values_with_index.columns = ['Month', 'Production forecast']
forecast_values_with_index['Month'] += 1

print("Forecast for 8 months ahead:")
print(forecast_values_with_index)
