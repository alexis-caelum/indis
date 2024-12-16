# 4. HMA RSX (Hull Moving Average RSX)
def hma_rsx(data, window=5):
    """
    Hull Moving Average RSX - Fourth best performer
    """

    def wma(data, window):
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # Calculate base RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Apply HMA smoothing
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))

    wma_half = wma(rsi, half_length)
    wma_full = wma(rsi, window)
    diff = 2 * wma_half - wma_full
    hma_rsx = wma(diff, sqrt_length)

    return hma_rsx.replace([np.inf, -np.inf], 50).fillna(50)