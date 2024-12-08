# TRIX
def calculate_trix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    trix = ta.trend.TRIXIndicator(close=df['close'], window=15)
    df['trix'] = trix.trix()
    df['entry_signal'] = (df['trix'] > 0).astype(int)
    df['exit_signal'] = (df['trix'] < 0).astype(int)
    return df