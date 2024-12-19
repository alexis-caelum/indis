import pandas as pd
import numpy as np
import talib as ta


def ta_rma(series, length):
    """
        RMA is an EMA with different weighting.
        EMA aplha is 2/(length+1), while RMA is 1/length.
    """

    # We can just adjust the EMA length in order to get our RMA
    ema_length = 2 * length - 1

    return ta.EMA(series, ema_length)

def sma_standard_deviation(data, len_dema=7, len_sma=60, len_sd=59, len_dema_sd=7):

    # DEMA calculations
    demas = calculate_dema(data['Close'], len_dema)
    demal = calculate_dema(data['Low'], len_dema)

    # SMA calculation
    sma = data['Close'].rolling(window=len_sma, min_periods=1).mean()

    # Main conditions
    mainl = demal > sma
    mains = demas < sma

    # Standard deviation calculations
    sd = sma.rolling(window=len_sd, min_periods=1).std()
    sd_upper = sma + sd
    sd_lower = sma - sd

    # Smoothed DEMA for standard deviation source
    dema_sd = calculate_dema(data['Close'], len_dema_sd)

    # SD conditions
    sd_s = dema_sd < sd_upper
    invert_l = ~sd_s

    # Final Long and Short Conditions
    L = mainl & invert_l
    S = mains

    # Generate signals
    signals = np.where(L & ~S, 1, np.where(S, -1, 0))

    return pd.Series(signals, index=data.index)

def dema_dmi(close, high, low, len_dema=15, adx_smoothing_len=3, di_len=18):
    """
        DMI on roids with the power of @IRS and DEMA.

        https://www.tradingview.com/script/rzPRTDQu-Dema-DMI-viResearch/
    """

    # Find the DEMA of the highs and lows
    demah = ta.DEMA(high, len_dema)
    demal = ta.DEMA(low, len_dema)

    # Calculate the direction of each DEMA
    u = np.diff(demah, prepend=np.nan)
    d = -np.diff(demal, prepend=np.nan)

    # P is where the DEMA high is going up,
    # and m where DEMA low is going down
    p = np.where((u > d) & (u > 0), u, 0)
    m = np.where((d > u) & (d > 0), d, 0)

    # True Range Calculation
    t = ta_rma(ta.TRANGE(high, low, close), di_len)

    # DI+ and DI- Calculation
    plus = 100 * ta_rma(p, di_len) / t
    minus = 100 * ta_rma(m, di_len) / t

    # Fix Nan values
    plus = np.nan_to_num(plus)
    minus = np.nan_to_num(minus)

    # ADX Calculation
    sum_dm = plus + minus
    adx = 100 * ta_rma(np.abs(plus - minus) / np.where(sum_dm == 0, 1, sum_dm), adx_smoothing_len)

    # Conditions
    adx_rising = adx > np.roll(adx, 1)
    dmil = (plus > minus) & adx_rising
    dmis = minus > plus

    # Convert to Series and forward-fill when signal is 0
    signal = np.where(dmil & ~dmis, 1, np.where(dmis, -1, np.nan))
    return pd.Series(signal, index=close.index).ffill()


def ta_linreg(series, length, offset=0):
    """
        Calculate linear regression using TA-Lib's LINEARREG function with correct offset handling.
    """

    # Calculate the linear regression using TA-Lib's LINEARREG function
    intercept = ta.LINEARREG_INTERCEPT(series, length)

    slope = ta.LINEARREG_SLOPE(series, length)

    return intercept + slope * (length - 1 - offset)


def ta_hma(series, length):
    """
        Calculate HMA (Hull Moving Average)
    """

    half_length = length // 2
    sqrt_length = int(np.sqrt(length))

    wma_half = ta.WMA(series, half_length)
    wma_full = ta.WMA(series, length)

    hma = ta.WMA(2 * wma_half - wma_full, sqrt_length)
    return np.array(hma)


def trendilo(close, hl2, lsma_length=25, offset=0, smooth=5, hma_length=50):
    """
        Trendilo LSMA uses a combination of moving averages and
        linear regressions to predict trend switches.

        https://www.tradingview.com/script/Xb4BQ8GS-Trendilo-LSMA-Band-Example/
    """

    # Find the EMA of hl2
    src = ta.EMA(hl2, smooth)

    # Calculate linear regression and HMA
    lsma = ta_linreg(src, lsma_length, offset)
    hma = ta_hma(src, hma_length)

    # When their average is rising, go long and vice-versa
    combine = (lsma + hma) / 2
    diff = np.diff(combine, prepend=0)

    # Generate signals: 1 for positive trend, -1 for negative
    trendilo_signals = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))

    return pd.Series(trendilo_signals, index=close.index)


def vii_stop(close, high, low, length=12, multiplier=2.8):
    '''
        Vii' Stop indicator with source set as the close.

        https://www.tradingview.com/script/cT7O5AAq-vii-Stop/
    '''

    # Convert prices to numpy arrays
    close = np.asarray(close)

    # Calculate ATR using high, low, and close prices
    atr = ta.ATR(high.to_numpy(), low.to_numpy(), close, length)

    # Initialize arrays to store results
    signals = np.zeros(len(close))
    stop = np.zeros(len(close))

    # Initialize max and min values
    max_val = close[0]
    min_val = close[0]
    uptrend = True  # Start with an uptrend assumption

    for i in range(1, len(close)):

        if np.isnan(close[i]):
            # Reset max_val, min_val, and stop if NaN is encountered
            max_val = close[i] if not np.isnan(close[i]) else 0
            min_val = close[i] if not np.isnan(close[i]) else 0
            stop[i] = 0
            uptrend = True  # Reinitialize trend assumption to uptrend
            continue

        atrM = atr[i] * multiplier

        if uptrend:
            max_val = max(max_val, close[i])
            stop[i] = max(stop[i - 1], max_val - atrM)
            if close[i] < stop[i]:  # Trend reversal to downtrend
                uptrend = False
                min_val = close[i]  # Reset min value on trend change
                stop[i] = min_val + atrM
        else:
            min_val = min(min_val, close[i])
            stop[i] = min(stop[i - 1], min_val + atrM)
            if close[i] > stop[i]:  # Trend reversal to uptrend
                uptrend = True
                max_val = close[i]  # Reset max value on trend change
                stop[i] = max_val - atrM

        # Set signals based on the current trend
        signals[i] = 1 if uptrend else -1

    # Return as pandas Series with the original index
    return pd.Series(signals, index=high.index)


def hullloop(close, length=5, a=1, b=50):
    """
    Calculate Hullloop based on the Hull Moving Average (HMA) and comparison scores.
    """
    hma_values = ta_hma(close, length)

    # Create shifted HMA values using broadcasting
    shift_indices = np.arange(a, b + 1)
    # Generate a 2D array of shifted HMA values
    shifted_hma_matrix = np.array([np.roll(hma_values, shift) for shift in shift_indices]).T

    # Calculate scores without loops
    score_matrix = np.where(hma_values[:, np.newaxis] > shifted_hma_matrix, 1, -1)
    scoreic = np.sum(score_matrix, axis=1)

    # Buy/Sell signals
    L = scoreic > 40
    S = scoreic < -10

    # Initialize and update variable
    hullloop = np.where(L & ~S, 1, np.where(S, -1, np.nan))

    return pd.Series(hullloop, index=close.index).ffill()


def percentile_nearest_rank_vectorized(series, length, percentile):
    """
        Find the value closest to a percentile
    """

    # Calculate the rank index for the nearest percentile
    rank = int(np.ceil(percentile / 100 * length)) - 1

    # Use a sliding window over the array to get all lookback periods (rolling windows)
    windows = np.lib.stride_tricks.sliding_window_view(series, length)

    # Sort each window
    sorted_windows = np.sort(windows, axis=1)

    # Get the value at the rank position for each sorted window
    percentiles = sorted_windows[:, rank]

    # Pad and return
    return np.concatenate([np.full(length - 1, np.nan), percentiles])


def aamdmom(close, len_momentum=4, len_percentile=35):
    momentum = ta.ROC(close, len_momentum)

    percentile_75 = percentile_nearest_rank_vectorized(close, len_percentile, 75)
    percentile_25 = percentile_nearest_rank_vectorized(close, len_percentile, 25)

    # Calculate rolling windows for percentiles
    rolling_windows = np.lib.stride_tricks.sliding_window_view(close, window_shape=len_percentile)

    # Calculate the 75th and 25th percentiles for each rolling window
    percentiles_75 = np.percentile(rolling_windows, 75, axis=1)
    percentiles_25 = np.percentile(rolling_windows, 25, axis=1)

    score = np.where(
        (momentum > 0) & (close > percentile_75), 1,
        np.where((momentum < 0) & (close < percentile_25), -1, np.nan)
    )

    return pd.Series(score, index=close.index).ffill()


def lsma_atr(close, high, low, len_lsma=75, len_atr=20, atr_multiplier=1.1):
    # 1. Calculate LSMA (Linear Regression Moving Average)
    lsma = ta_linreg(close, len_lsma)

    # 2. Calculate ATR and smoothed ATR (SMA of ATR)
    atr = ta.ATR(high, low, close, len_atr)  # Using close prices for simplicity
    smoothed_atr = ta.SMA(atr, len_atr)

    # 3. Calculate LSMA slope
    lsma_slope = np.diff(lsma, prepend=np.nan)  # LSMA slope as difference between current and previous LSMA

    # 4. Determine trend direction
    uptrend = (lsma_slope > 0) & (close > lsma + atr_multiplier * smoothed_atr)
    downtrend = (lsma_slope < 0) & (close < lsma - atr_multiplier * smoothed_atr)

    # 5. Score logic based on trend direction
    lsma_atr = np.full_like(close, np.nan)  # Initialize the score array
    lsma_atr[uptrend & ~downtrend] = 1
    lsma_atr[downtrend] = -1

    return pd.Series(lsma_atr, index=close.index).ffill()


def lrsi(close, alpha=0.2, long_threshold=0.5, short_threshold=0.5):
    """
    LRSI Indicator partially vectorized.

    :param close: Price series (usually close prices)
    :param alpha: Alpha value used for calculation (default 0.7)
    :return: lrsi values
    """
    length = len(close)

    # Initialize arrays for L0, L1, L2, L3
    L0 = np.zeros(length)
    L1 = np.zeros(length)
    L2 = np.zeros(length)
    L3 = np.zeros(length)

    # Perform recursive calculations (this part can't be vectorized)
    for i in range(1, length):
        if np.isnan(close.iloc[i]):
            # Reset recursive values when NaN is encountered
            L0[i] = L1[i] = L2[i] = L3[i] = 0
            continue

        L0[i] = alpha * close.iloc[i] + (1 - alpha) * L0[i - 1]
        L1[i] = -(1 - alpha) * L0[i] + L0[i - 1] + (1 - alpha) * L1[i - 1]
        L2[i] = -(1 - alpha) * L1[i] + L1[i - 1] + (1 - alpha) * L2[i - 1]
        L3[i] = -(1 - alpha) * L2[i] + L2[i - 1] + (1 - alpha) * L3[i - 1]

    # Vectorized CU and CD calculations
    CU = ((L0 >= L1) * (L0 - L1)) + ((L1 >= L2) * (L1 - L2)) + ((L2 >= L3) * (L2 - L3))
    CD = ((L0 < L1) * (L1 - L0)) + ((L1 < L2) * (L2 - L1)) + ((L2 < L3) * (L3 - L2))

    # Avoid divide by zero using np.where
    lrsi_val = CU / (CU + CD + 1e-10)
    lrsi = np.where(
        lrsi_val > long_threshold, 1,
        (np.where(lrsi_val < short_threshold, -1, np.nan)))

    return pd.Series(lrsi, index=close.index).ffill()


def rti(close, trend_data_count=100, trend_sensitivity_percentage=95, signal_length=20, long_threshold=50,
        short_threshold=50):
    '''
    RTI uses the standard deviation of price to create upper and lower bounds
    and uses their distance to price in order to estimate the trend.

    https://www.tradingview.com/script/VwmUNNwp-Relative-Trend-Index-RTI-by-Zeiierman/
    '''

    # Calculate the rolling standard deviation
    stdev = ta.STDDEV(close, 2)

    # Calculate the upper and lower trends
    upper_trend = close + stdev
    lower_trend = close - stdev

    # Use a rolling window to capture subarrays for sorting and trend calculations
    upper_windows = np.lib.stride_tricks.sliding_window_view(upper_trend, trend_data_count)
    lower_windows = np.lib.stride_tricks.sliding_window_view(lower_trend, trend_data_count)

    # Sort the windows along each subarray (axis=1 for each window)
    upper_sorted = np.sort(upper_windows, axis=1)
    lower_sorted = np.sort(lower_windows, axis=1)

    # Calculate the indices based on sensitivity
    upper_index = int(np.round(trend_sensitivity_percentage / 100 * trend_data_count)) - 1
    lower_index = int(np.round((100 - trend_sensitivity_percentage) / 100 * trend_data_count)) - 1

    # Extract the upper and lower trend values based on sensitivity
    UpperTrend = upper_sorted[:, upper_index]
    LowerTrend = lower_sorted[:, lower_index]

    # Calculate RTI for the full series
    rti = ((close[trend_data_count - 1:] - LowerTrend) / (UpperTrend - LowerTrend)) * 100

    # Prepend NaNs to match the original length of the close series
    rti_full = np.concatenate([np.full(trend_data_count - 1, np.nan), rti])

    # Calculate the Exponential Moving Average (EMA) of the RTI
    ma_rti = ta.EMA(rti_full, signal_length)

    return pd.Series(np.where(
        ma_rti > long_threshold, 1,
        (np.where(ma_rti < short_threshold, -1, np.nan))), index=close.index).ffill()


def ta_lowest(series, length):
    """
        Find the rolling lowest of a data series.
    """

    # Split the series into windows of length, and find their minimum value
    lowest = np.lib.stride_tricks.sliding_window_view(series, length).min(axis=1)

    # Pad the array with Nans and return it
    return np.concatenate([np.full(length - 1, np.nan), lowest])


def ta_highest(series, length):
    """
        Find the rolling highest of a data series.
    """

    # Split the series into windows of length, and find their maximum value
    highest = np.lib.stride_tricks.sliding_window_view(series, length).max(axis=1)

    # Pad the array with Nans and return it
    return np.concatenate([np.full(length - 1, np.nan), highest])


def ta_rising(series, length):
    """
        Rolling calculation of whether the series is rising over a specified window.
    """

    # Split the series into windows
    windows = np.lib.stride_tricks.sliding_window_view(series, length + 1)

    # Check if they are rising
    rising = np.all(np.diff(windows) > 0, axis=1)

    # Pad with some False to retain length
    return np.concatenate([np.full(length, False), rising])


def stc(close, stc_length=12, fast_length=26, slow_length=50, factor=0.5):
    """
        Schaff Trend Cycle.

        https://www.tradingview.com/script/WhRRThMI-STC-Indicator-A-Better-MACD-SHK/
    """

    # Calculate MACD diff
    macd = ta.EMA(close, fast_length) - ta.EMA(close, slow_length)

    # Calculate %K
    lowest_k = ta_lowest(macd, stc_length)
    highest_k = ta_highest(macd, stc_length)
    range_k = highest_k - lowest_k
    percent_k = np.where(range_k > 0, (macd - lowest_k) / range_k * 100, 0)

    def stc_smoothing(series):
        smoothed_series = np.zeros(len(series))
        for i in range(1, len(series)):
            smoothed_series[i] = smoothed_series[i - 1] + factor * (series[i] - smoothed_series[i - 1])
        return smoothed_series

    # Smoothed %K
    smoothed_k = stc_smoothing(percent_k)

    # Calculate %D
    lowest_d = ta_lowest(smoothed_k, stc_length)
    highest_d = ta_highest(smoothed_k, stc_length)
    range_d = highest_d - lowest_d
    # Avoid division by zero
    range_d = np.where(range_d == 0, 1e-10, range_d)
    percent_d = np.where(range_d > 0, (smoothed_k - lowest_d) / range_d * 100, 0)

    # Smoothed %D. Round the results to 10 floating points like TV
    stc = np.round(stc_smoothing(percent_d), 10)

    # Signal is 1 when rising, and -1 when falling
    return pd.Series(np.where(ta_rising(stc, 1), 1, -1), index=close.index)


def ta_sum(series, length):
    """
        Perform a rolling sum.
        Equivalent of Pine's math.sum
    """

    # Calculate the cumulative sum
    rsum = np.cumsum(series)

    # Subtract lagged versions of the sum to get the rolling sum
    rsum[length:] = rsum[length:] - rsum[:-length]

    # First values are invalid
    rsum[:length] = np.nan

    return rsum


def pulsarRsi(close_prices, rsiMax=60, rsiMin=44):
    """
    Calculate the Pulsar RSI indicator.

    :param close_prices: Series of close prices.
    :param rsiMax: Threshold for long signals.
    :param rsiMin: Threshold for short signals.
    :return: Series of signals (1 for long, -1 for short, 0 for neutral).
    """
    # Calculate RSI
    rsi = ta.RSI(close_prices, timeperiod=6)

    # Smooth the RSI using DEMA
    rsi_smooth = ta.DEMA(rsi.dropna(), timeperiod=8).reindex(close_prices.index).ffill()

    # Generate signals based on rsiMax and rsiMin
    signals = pd.Series(0, index=close_prices.index)
    signals[rsi_smooth > rsiMax] = 1  # Long signal
    signals[rsi_smooth < rsiMin] = -1  # Short signal

    return pd.Series(signals, index=close_prices.index)


def kama(close, fast_period=7, slow_period=19, er_period=8, norm_period=50):
    """
        Kaufman Adaptive Moving Average normalized.

        https://www.tradingview.com/script/OwtiIzT3-Normalized-KAMA-Oscillator-Ikke-Omar/
    """

    # Find the absolute value of close - close[er_period]
    change = np.abs(close - np.roll(close, er_period))

    # Calculate the rolling sum of the daily difference
    diff = abs(np.diff(close, prepend=close.to_numpy()[0]))

    volatility = ta_sum(diff, er_period)

    # Find the efficiency ratio
    er = change / volatility

    # Calculate the smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc

    # Calculate KAMA
    ema_fast = ta.EMA(close, fast_period)
    kama = ema_fast + sc * (close - ema_fast)

    # Normalize
    lowest = ta_lowest(kama, norm_period)
    highest = ta_highest(kama, norm_period)
    normalized = (kama - lowest) / (highest - lowest) - 0.5

    # Generate signals: 1 if kama > 0, -1 if kama < 0, else 0
    signals = np.where(normalized > 0, 1, (np.where(normalized < 0, -1, 0)))

    return pd.Series(signals, index=close.index)


def system(rsi_vii_for_loop):
    total = np.sum([1 if rsi_vii_for_loop > i else -1 for i in range(25, 76)])
    return total


def vii_for_loop(close, for_loop_len=14):
    """
    Calculate the VII indicator using a for loop.

    :param close: Series of close prices.
    :param for_loop_len: Length of the for loop.
    :return: Series of signals.
    """
    # Convert to numpy array if needed
    close_values = close.values if isinstance(close, pd.Series) else close

    # Example logic for the indicator
    signals = np.zeros_like(close_values)
    for i in range(for_loop_len, len(close_values)):
        # Example calculation (replace with actual logic)
        signals[i] = 1 if close_values[i] > close_values[i - for_loop_len] else -1

    return pd.Series(signals, index=close.index)


def calculate_dema(series, length):
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    dema = 2 * ema1 - ema2
    return dema

# Fully vectorized SMA Standard Deviation Indicator
def sma_standard_deviation(data, len_dema=5, len_sma=50, len_sd=59, len_dema_sd=7):

    # DEMA calculations
    demas = calculate_dema(data['Close'], len_dema)
    demal = calculate_dema(data['Low'], len_dema)

    # SMA calculation
    sma = data['Close'].rolling(window=len_sma, min_periods=1).mean()

    # Main conditions
    mainl = demal > sma
    mains = demas < sma

    # Standard deviation calculations
    sd = sma.rolling(window=len_sd, min_periods=1).std()
    sd_upper = sma + sd
    sd_lower = sma - sd

    # Smoothed DEMA for standard deviation source
    dema_sd = calculate_dema(data['Close'], len_dema_sd)

    # SD conditions
    sd_s = dema_sd < sd_upper
    invert_l = ~sd_s

    # Final Long and Short Conditions
    L = mainl & invert_l
    S = mains

    # Generate signals
    signals = np.where(L & ~S, 1, np.where(S, -1, 0))

    return pd.Series(signals, index=data.index)

def rsi_momentum_trend(data, len2=14, pmom=70, nmom=30):
    """
    Calculate the RSI Momentum Trend indicator.

    :param data: DataFrame or Series with close prices.
    :param len2: Length for RSI calculation.
    :param pmom: Positive momentum threshold.
    :param nmom: Negative momentum threshold.
    :return: Series of signals.
    """
    # Ensure 'data' is a Series or extract 'Close' if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        close = data['Close'].values
    elif isinstance(data, pd.Series):
        close = data.values
    else:
        close = data  # Assume it's already a numpy array

    # Calculate RSI
    rsi = ta.RSI(close, timeperiod=len2)

    # Generate signals based on RSI thresholds
    signals = np.zeros_like(rsi)
    for i in range(len(rsi)):
        if rsi[i] > pmom:
            signals[i] = 1
        elif rsi[i] < nmom:
            signals[i] = -1

    return pd.Series(signals, index=data.index if isinstance(data, (pd.Series, pd.DataFrame)) else range(len(signals)))