# 6. ALMA RSX (Arnaud Legoux Moving Average RSX)
def alma_rsx(data, window=5, offset=0.85, sigma=6):
    """
    Arnaud Legoux Moving Average RSX - Sixth best performer
    """
    # Calculate base RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Apply ALMA smoothing
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-0.5 * ((np.arange(window) - m) / s) ** 2)
    weights /= weights.sum()

    alma_rsx = rsi.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

    return alma_rsx.replace([np.inf, -np.inf], 50).fillna(50)
