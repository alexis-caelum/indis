# 3. TEMA RSX (Triple Exponential Moving Average RSX)
def tema_rsx(data, window=5):
    """
    Triple Exponential Moving Average RSX - Third best performer
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

    # Apply TEMA smoothing
    ema1 = ema(rsi, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    tema_rsx = 3 * ema1 - 3 * ema2 + ema3

    return tema_rsx.replace([np.inf, -np.inf], 50).fillna(50)