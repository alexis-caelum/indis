# Williams %R
def calculate_williams_r(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wr = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'])
    df['williams_r'] = wr.williams_r()
    df['entry_signal'] = (df['williams_r'] < -80).astype(int)
    df['exit_signal'] = (df['williams_r'] > -20).astype(int)
    return df