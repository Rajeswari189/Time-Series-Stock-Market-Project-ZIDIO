from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os


def run_arima(df, order=(5, 1, 0)):
    model = ARIMA(df['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    
    # Plot
    df['Close'].plot(label='Actual')
    forecast.plot(label='ARIMA Forecast')
    plt.title('ARIMA Stock Forecast')
    plt.legend()
    
    os.makedirs('outputs/forecast_plots', exist_ok=True)
    plt.savefig('outputs/forecast_plots/arima_forecast.png')
    plt.close()
    
    return forecast
