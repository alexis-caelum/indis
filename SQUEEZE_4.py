def dema_squeeze_3(df):
    window = 3
    # Calculate DEMA
    ema1 = df['Close'].ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ma = 2 * ema1 - ema2

    # Calculate Bollinger Bands
    std = df['Close'].rolling(window).std()
    upper_band = ma + (std * 2.0)
    lower_band = ma - (std * 2.0)

    # Calculate squeeze conditions
    band_width = upper_band - lower_band
    squeeze_threshold = band_width.quantile(0.25)
    squeeze = band_width < squeeze_threshold

    # Generate signals
    signals = pd.Series(0, index=df.index)
    buy_signals = (squeeze.shift(1) == True) & (squeeze == False) & (df['Close'] > ma)
    sell_signals = (squeeze.shift(1) == False) & (squeeze == False) & (df['Close'] < ma)

    signals[buy_signals] = 1
    signals[sell_signals] = -1

    return signals