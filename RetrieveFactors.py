import numpy as np
import pandas as pd

def compute_volatility_factors(log_returns, benchmark, horizons=[5, 21, 63, 126, 252], gamma=0.5, sigma=0.03):
    """
    Compute multi-horizon volatility shock factors.
    """
    max_horizon = max(horizons)
    raw_weights = np.array([(max_horizon - h + 1) ** (-gamma) for h in horizons])
    weights = raw_weights / raw_weights.sum()

    combined_vol = pd.DataFrame(0.0, index=log_returns.index, columns=log_returns.columns)

    for h, w in zip(horizons, weights):
        stock_lag = log_returns.shift(h)
        benchmark_lag = benchmark.shift(h)

        abs_dev = (stock_lag.sub(benchmark_lag, axis=0)).abs()
        G = abs_dev.subtract(sigma).clip(lower=0.0)
        combined_vol += w * G

    return combined_vol.fillna(0.0)


def compute_volume_factors(log_returns, price_volume, Industry_Map, tau=5):
    avg_vol = price_volume.rolling(tau).mean()
    volume_ratio = price_volume / avg_vol
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    volume_shock = np.log(volume_ratio)
    return_tau = log_returns.rolling(tau).sum()
    raw_volume_factor = volume_shock * return_tau

    sector_neutral = raw_volume_factor.copy()
    for industry, group in Industry_Map.groupby('Industry'):
        symbols = group['Symbol'].tolist()
        sector_mean = raw_volume_factor[symbols].mean(axis=1)
        sector_neutral[symbols] = raw_volume_factor[symbols].sub(sector_mean, axis=0)

    return sector_neutral.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def NormalisedSectorNeutralReturns(price, log_return, industry_map_df, horizon=5, volatility_days=21):
    m_log_return = np.log(price / price.shift(horizon))
    volatility = log_return.rolling(volatility_days).std().clip(lower=1e-5)

    sector_neutral = pd.DataFrame(index=log_return.index, columns=log_return.columns)

    for industry, group in industry_map_df.groupby('Industry'):
        symbols = group['Symbol'].tolist()
        sector_mean = m_log_return[symbols].mean(axis=1)
        sector_neutral[symbols] = m_log_return[symbols].subtract(sector_mean, axis=0)

    normalized = sector_neutral / volatility
    ranked_normalized = normalized.apply(rank_transform, axis=1)

    return ranked_normalized

def rank_transform(x):
    valid = x.dropna()
    N = len(valid)
    if N <= 1:
        return pd.Series(0.0, index=x.index)
    ranks = valid.rank(ascending=False, method='first')
    transformed = (N + 1 - 2 * ranks) / (N - 1)
    return transformed.reindex(x.index).fillna(0.0)

import pandas as pd
import numpy as np

def technical_factors(high_df, low_df, adjusted_df):
    # Bollinger Band % Position
    ma20 = adjusted_df.rolling(20).mean()
    std20 = adjusted_df.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_pct = (adjusted_df - lower) / (upper - lower)

    # True Range Components
    high_low = high_df - low_df
    high_close = (high_df - adjusted_df.shift(1)).abs()
    low_close = (low_df - adjusted_df.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=0).groupby(level=0).max()

    # ATR
    atr = tr.rolling(window=20).mean()

    # Final output
    factors = {
        'BB_Pct': bb_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        'ATR': atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    }

    return factors

def macro_factors(repo_series, yield_series, index):
    macro_df = pd.DataFrame({
        'repo': repo_series,
        'tenyr': yield_series,
    }).reindex(index).ffill().bfill().astype(float)
    
    return macro_df
