#!/usr/bin/env python3
"""
P5 — Geopolitical Stress Score
Data Pipeline: Fetch → Returns → Rolling Z-Scores → Composite Score → Event Merge
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR      = Path(r'C:\Projects\P5-GeopoliticalStressScore')
RAW_DIR       = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

START_DATE = '2000-01-01'
END_DATE   = pd.Timestamp.today().strftime('%Y-%m-%d')
WINDOW     = 252   # rolling z-score lookback window — 1 trading year

# direction:  +1  →  rising value signals stress
#             -1  →  falling value signals stress
# use_level:  True   →  z-score the raw price level  (VIX only — the level IS the fear signal)
#             False  →  z-score the daily percentage return  (all other assets)

ASSETS = {
    'SP500':        {'ticker': '^GSPC',     'direction': -1, 'use_level': False},
    'Gold':         {'ticker': 'GC=F',      'direction':  1, 'use_level': False},
    'Oil_WTI':      {'ticker': 'CL=F',      'direction':  1, 'use_level': False},
    'VIX':          {'ticker': '^VIX',      'direction':  1, 'use_level': True },
    'Treasury_10Y': {'ticker': '^TNX',      'direction': -1, 'use_level': False},
    'USD_Index':    {'ticker': 'DX-Y.NYB',  'direction':  1, 'use_level': False},
    'Bitcoin':      {'ticker': 'BTC-USD',   'direction': -1, 'use_level': False},
    'EM_Equities':  {'ticker': 'EEM',       'direction': -1, 'use_level': False},
    'HighYield':    {'ticker': 'HYG',       'direction': -1, 'use_level': False},
    'EURUSD':       {'ticker': 'EURUSD=X',  'direction': -1, 'use_level': False},
    'JPY_USD':      {'ticker': 'JPY=X',     'direction': -1, 'use_level': False},
    # JPY=X is USD/JPY — a falling value means the yen is strengthening
    # yen strengthening = safe-haven demand = stress signal → direction -1
}

MIN_ASSETS = 6  # minimum number of assets required on a given day to compute composite
                # handles the Bitcoin data gap cleanly (Bitcoin only available from 2014)


# ── Step 1: Fetch raw price data ───────────────────────────────────────────────

def fetch_prices():
    print('\n── Step 1: Fetching raw price data ──')
    prices = {}

    for name, config in ASSETS.items():
        ticker = config['ticker']
        print(f'  Downloading {name} ({ticker}) ...')

        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                print(f'  WARNING: no data returned for {name} — skipping.')
                continue

            close = df['Close'].squeeze()
            close.name = name
            prices[name] = close

            close.to_csv(RAW_DIR / f'{name}.csv', header=True)
            print(f'  {name}: {len(close)} rows  '
                  f'({close.index[0].date()} → {close.index[-1].date()})')

        except Exception as e:
            print(f'  ERROR downloading {name}: {e}')

    prices_df = pd.DataFrame(prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.to_csv(RAW_DIR / 'all_prices.csv')
    print(f'\n  All raw prices saved → {RAW_DIR / "all_prices.csv"}')
    return prices_df


# ── Step 2: Compute daily returns ──────────────────────────────────────────────

def compute_returns(prices_df):
    print('\n── Step 2: Computing daily returns ──')
    returns = {}

    for name, config in ASSETS.items():
        if name not in prices_df.columns:
            continue

        series = prices_df[name].dropna()

        if config['use_level']:
            # VIX: the level itself is the stress signal — no return transformation needed
            returns[name] = series
            print(f'  {name}: using raw level')
        else:
            ret = series.pct_change()
            returns[name] = ret
            print(f'  {name}: daily percentage return computed')

    returns_df = pd.DataFrame(returns)
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df.to_csv(PROCESSED_DIR / 'returns.csv')
    print(f'\n  Returns saved → {PROCESSED_DIR / "returns.csv"}')
    return returns_df


# ── Step 3: Rolling 252-day z-scores ──────────────────────────────────────────

def compute_zscores(returns_df):
    print(f'\n── Step 3: Computing rolling z-scores (window = {WINDOW} days) ──')
    zscores = {}

    for name, config in ASSETS.items():
        if name not in returns_df.columns:
            continue

        series = returns_df[name].dropna()

        rolling_mean = series.rolling(window=WINDOW, min_periods=WINDOW).mean()
        rolling_std  = series.rolling(window=WINDOW, min_periods=WINDOW).std()

        # replace zero standard deviation with NaN to avoid divide-by-zero
        # this can happen on assets with no movement over the lookback window
        rolling_std = rolling_std.replace(0, np.nan)

        z = (series - rolling_mean) / rolling_std

        # apply direction multiplier so that a positive z-score always means stress
        z = z * config['direction']
        z.name = name
        zscores[name] = z

        valid = z.dropna().shape[0]
        print(f'  {name}: {valid} valid z-score observations')

    zscores_df = pd.DataFrame(zscores)
    zscores_df.index = pd.to_datetime(zscores_df.index)
    zscores_df.to_csv(PROCESSED_DIR / 'zscores.csv')
    print(f'\n  Z-scores saved → {PROCESSED_DIR / "zscores.csv"}')
    return zscores_df


# ── Step 4: Composite GSS score ────────────────────────────────────────────────

def compute_composite(zscores_df):
    print('\n── Step 4: Computing composite GSS score ──')

    # count how many assets have a valid z-score on each trading day
    valid_counts = zscores_df.notna().sum(axis=1)

    # only compute the composite on days where we have enough assets
    # this prevents the early 2000s score from being distorted by only 2-3 assets
    eligible_rows = zscores_df[valid_counts >= MIN_ASSETS]
    composite = eligible_rows.mean(axis=1)
    composite.name = 'GSS'

    # re-index back to the full date range — days below MIN_ASSETS threshold become NaN
    composite = composite.reindex(zscores_df.index)

    output = pd.DataFrame({
        'GSS':                 composite,
        'assets_in_composite': valid_counts,
    })
    output.to_csv(PROCESSED_DIR / 'gss_scores.csv')

    valid_scores = composite.dropna()
    print(f'  Composite score computed across {len(valid_scores)} trading days')
    print(f'  Score range : {valid_scores.min():.3f}  →  {valid_scores.max():.3f}')
    print(f'  Score mean  : {valid_scores.mean():.3f}')
    print(f'  Score median: {valid_scores.median():.3f}')
    print(f'\n  GSS scores saved → {PROCESSED_DIR / "gss_scores.csv"}')
    return output


# ── Step 5: Merge events onto GSS scores ──────────────────────────────────────

def merge_events(gss_df):
    print('\n── Step 5: Merging events ──')

    events_path = BASE_DIR / 'data' / 'events.csv'

    if not events_path.exists():
        print(f'  WARNING: events.csv not found at {events_path} — skipping merge.')
        return gss_df

    events = pd.read_csv(events_path, parse_dates=['event_date'])
    gss_index = gss_df.index

    # for each event, find the nearest trading day in the GSS date index
    # this handles weekends and public holidays cleanly
    matched_dates = []
    for _, row in events.iterrows():
        target = row['event_date']
        nearest = (gss_index - target).to_series().abs().idxmin()
        matched_dates.append(nearest)

    events['trading_date']  = matched_dates
    events['GSS_on_event']  = events['trading_date'].map(gss_df['GSS'])

    events.to_csv(PROCESSED_DIR / 'events_with_gss.csv', index=False)

    print(f'  {len(events)} events annotated with their GSS score on event date')
    print(f'  Saved → {PROCESSED_DIR / "events_with_gss.csv"}')

    # print a quick summary sorted by GSS score so you can sanity-check the output
    top = (
        events[['event_name', 'event_date', 'category', 'GSS_on_event']]
        .dropna(subset=['GSS_on_event'])
        .sort_values('GSS_on_event', ascending=False)
        .head(10)
    )
    print('\n  Top 10 highest-stress events by GSS score on event date:')
    print(top.to_string(index=False))

    return gss_df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('  P5 — Geopolitical Stress Score: Data Pipeline')
    print('=' * 60)

    prices_df  = fetch_prices()
    returns_df = compute_returns(prices_df)
    zscores_df = compute_zscores(returns_df)
    gss_df     = compute_composite(zscores_df)
    gss_df     = merge_events(gss_df)

    print('\n' + '=' * 60)
    print('  Pipeline complete. Output files written to:')
    print(f'  {PROCESSED_DIR}')
    print('=' * 60)
