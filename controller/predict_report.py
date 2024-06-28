import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Example time series data (replace with your own data)
data = {
    'timestamp': pd.date_range('2024-01-01', periods=60, freq='T'),
    'value': [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 
              40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68,
              70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98,
              100, 102, 105, 108, 110, 112, 115, 118, 120, 122,
              125, 128, 130, 132, 135, 138, 140, 142, 145, 148,
              150, 152, 155, 158]
}
df = pd.DataFrame(data)
df.set_index('timestamp', inplace=True)

# Fit ARIMA model
model = ARIMA(df['value'], order=(1, 1, 1))  # Example order, you may need to tune this
fitted_model = model.fit()

# Forecast next 10 data points
forecast_horizon = 10
forecast = fitted_model.forecast(steps=forecast_horizon)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Actual Data')
plt.plot(pd.date_range(df.index[-1], periods=forecast_horizon+1, freq='T')[1:], forecast, label='Forecast')
plt.title('ARIMA Forecasting Example')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.show()

print("Forecasted values for the next 10 time steps:")
print(forecast)
