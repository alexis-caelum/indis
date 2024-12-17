# 5. VWMA RSX (Volume Weighted Moving Average RSX)
def vwma_rsx(data, window=5):
    """
    Volume Weighted Moving Average RSX - Fifth best performer
    """
    # Calculate base RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Apply VWMA smoothing
    vwma_rsx = (rsi * data['Volume']).rolling(window).sum() / data['Volume'].rolling(window).sum()

    return vwma_rsx.replace([np.inf, -np.inf], 50).fillna(50)