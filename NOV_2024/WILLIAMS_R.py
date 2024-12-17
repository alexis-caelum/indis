def wma_squeeze_4(df):
    # Calculate WMA
    window = 4
    weights = np.arange(1, window + 1)
    ma = df['Close'].rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

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