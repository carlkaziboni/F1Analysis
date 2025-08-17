import pandas as pd

def process_lap_data(df):
    if df.empty:
        return pd.DataFrame()
    df = df[df['lap_duration'].notna()]
    df = df.sort_values(by=['driver_number', 'lap_number'])
    return df