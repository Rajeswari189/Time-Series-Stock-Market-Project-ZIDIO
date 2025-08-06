# Time-Series-Stock-Market-Project-ZIDIO
Time Series Stock Market Project – ZIDIO internship program 

# Time Series Forecasting: NIFTY 50 Stock Data

##  Project Overview
This project forecasts NIFTY 50 stock prices using time series models:
- ARIMA (Auto Regressive Integrated Moving Average)
- Prophet (by Facebook)
- LSTM (to be added)

##  Folder Structure
- `data/`: Contains NIFTY 50 stock data (from Kaggle)
- `src/`: Source code for preprocessing and models
- `outputs/`: Forecast plots
- `main.py`: Main script to run the full pipeline

##  Steps to Run
1. Place `nifty50.csv` in the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

##  Output
Forecast plots will be saved in `outputs/forecast_plots/`.

##  Learning Outcomes
- Real stock forecasting with ARIMA & Prophet
- Time series data cleaning and visualization
- Hands-on with real market data
