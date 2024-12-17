def calculate_swing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Calculate swing points with a default window of 2
    def swing_high_low(df: pd.DataFrame, swing_window=2):
        swing_high = pd.Series(False, index=df.index)
        swing_low = pd.Series(False, index=df.index)

        for i in range(swing_window, len(df) - swing_window):
            # Check for swing high: current high > all surrounding highs
            if df['high'].iloc[i] > df['high'].iloc[i - swing_window:i + swing_window + 1].drop(df.index[i]).max():
                swing_high.iloc[i] = True
            # Check for swing low: current low < all surrounding lows
            if df['low'].iloc[i] < df['low'].iloc[i - swing_window:i + swing_window + 1].drop(df.index[i]).min():
                swing_low.iloc[i] = True

        return swing_high, swing_low

    # Calculate swing points
    swing_high, swing_low = swing_high_low(df)

    # Generate signals
    df['entry_signal'] = swing_low.astype(int)  # Buy at swing lows
    df['exit_signal'] = swing_high.astype(int)  # Sell at swing highs

    return df