# Keltner Channels
def calculate_keltner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low'] = kc.keltner_channel_lband()
    df['entry_signal'] = (df['close'] < df['kc_low']).astype(int)
    df['exit_signal'] = (df['close'] > df['kc_high']).astype(int)
    return df