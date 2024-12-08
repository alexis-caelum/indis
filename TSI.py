# TSI (True Strength Index)
def calculate_tsi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tsi = ta.momentum.TSIIndicator(close=df['close'], window_slow=25, window_fast=13)
    df['tsi'] = tsi.tsi()
    df['entry_signal'] = (df['tsi'] > 0).astype(int)
    df['exit_signal'] = (df['tsi'] < 0).astype(int)
    return df