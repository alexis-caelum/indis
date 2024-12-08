# Stochastic Oscillator
def calculate_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['entry_signal'] = (df['stoch_k'] < 20).astype(int)
    df['exit_signal'] = (df['stoch_k'] > 80).astype(int)
    return df