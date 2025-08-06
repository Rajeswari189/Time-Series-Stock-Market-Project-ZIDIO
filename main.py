from src.preprocess import preprocess
from src.arima_model import run_arima
from src.prophet_model import run_prophet
from src.lstm_model import run_lstm

# Load and preprocess data
df = preprocess('data/nifty50.csv')

# Run forecasting models
print("Running ARIMA...")
run_arima(df)

print("Running Prophet...")
run_prophet(df)

print("Running LSTM...")
run_lstm(df)

print("All forecasts complete! Check outputs/forecast_plots/")
