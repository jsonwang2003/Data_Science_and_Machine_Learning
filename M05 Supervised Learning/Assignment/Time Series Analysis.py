# Using the taco_sales.csv file, run a time series analysis 
# to model the next 30 days of multiple linear regression on 
# the student ‘marketing_dollars_spent’ based on all the 
# variables in the dataset 

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df= pd.read_csv('../taco_sales.csv')

# Convert date column to datetime format and set it as index
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.set_index('date', inplace=True)

# Remove rows with missing date values
df_cleaned = df.dropna(subset=['marketing_dollars_spent'])

# Ensure dataset is sorted by date
df_cleaned = df_cleaned.sort_index()

# Resample to daily frequency, filling missing values with 0
df_daily = df_cleaned.resample('D').sum()

# Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['marketing_dollars_spent'], label='Marketing Dollars Spent')
plt.xlabel('Date')
plt.ylabel('Marketing Dollars Spent')
plt.title('Marketing Dollars Spent Over Time')
plt.legend()
plt.grid()
plt.show()

# Fit an ARIMA model (AutoRegressive Integrated Moving Average)
model = ARIMA(df_daily['marketing_dollars_spent'], order=(5, 1, 0))  # ARIMA(p,d,q)
model_fit = model.fit()

# Forecast the next 60 days (2 months)
forecast_steps = 60
forecast = model_fit.forecast(steps=forecast_steps)

# Generate future dates
future_dates = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Create a forecast DataFrame
forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_marketing_dollars_spent': forecast.values})

# Display forecasted sales using pandas head 
# Display the first few rows for preview
print("Forecasted Marketing Dollars Spent:")
print(forecast_df.head())