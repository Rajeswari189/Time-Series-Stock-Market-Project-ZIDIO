import pandas as pd


def preprocess(file_path='data/nifty50.csv'):
    df = pd.read_csv(
     file_path,
     encoding='ISO-8859-1',
     parse_dates=['Date'],
     index_col='Date'
                    )

    df = df[['Open', 'High', 'Low', 'LTP', 'Volume (lacs)']]
    df.rename(
        columns={
            'LTP': 'Close',
            'Volume (lacs)': 'Volume'
        },
        inplace=True
    )

    df = df.dropna()
    return df
