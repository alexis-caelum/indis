# Ichimoku Cloud
def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()

    df['ichimoku_shift_c'] = df['ichimoku_conversion'].shift(1)
    df['ichimoku_shift_b'] = df['ichimoku_base'].shift(1)

    df['entry_signal'] = ((df['ichimoku_conversion'] > df['ichimoku_base']) &
                          (df['ichimoku_shift_c'] <= df['ichimoku_shift_b'])).astype(int)

    df['exit_signal'] = ((df['ichimoku_conversion'] < df['ichimoku_base']) &
                         (df['ichimoku_shift_c'] >= df['ichimoku_shift_b'])).astype(int)

    df.drop(['ichimoku_shift_c', 'ichimoku_shift_b'], axis=1, inplace=True)
    return df