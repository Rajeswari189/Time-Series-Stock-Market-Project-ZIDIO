from prophet import Prophet
# import pandas as pd
import matplotlib.pyplot as plt
import os


def run_prophet(df):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title('Prophet Stock Forecast')
    
    os.makedirs('outputs/forecast_plots', exist_ok=True)
    fig.savefig('outputs/forecast_plots/prophet_forecast.png')
    plt.close()

    return forecast
