# Coppock
def wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window+1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def calculate_coppock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    roc11 = ta.momentum.ROCIndicator(close=df['close'], window=11).roc()
    roc14 = ta.momentum.ROCIndicator(close=df['close'], window=14).roc()
    coppock_raw = roc11 + roc14
    df['coppock'] = wma(coppock_raw, 10)
    df['entry_signal'] = (df['coppock'] > 0).astype(int)
    df['exit_signal'] = (df['coppock'] < 0).astype(int)
    return df