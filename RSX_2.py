# 2. DEMA RSX (Double Exponential Moving Average RSX)
def dema_rsx(data, window=5):
    """
    Double Exponential Moving Average RSX - Second best performer
    """

    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()

    # Calculate base RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Apply DEMA smoothing
    ema1 = ema(rsi, window)
    ema2 = ema(ema1, window)
    dema_rsx = 2 * ema1 - ema2

    return dema_rsx.replace([np.inf, -np.inf], 50).fillna(50)