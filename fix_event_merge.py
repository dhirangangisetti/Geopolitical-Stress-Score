#!/usr/bin/env python3
"""
Fix for event merge v2:
- Uses next available trading day ON OR AFTER the event date (not nearest)
  because market impact can only materialise after an event occurs, never before
- Flags events that fall before the 252-day warmup window as unmeasurable
- Max forward-look window: 10 trading days (handles weekends, public holidays,
  and market closures like 9/11 which reopened 6 days later)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR      = Path(r'C:\Projects\P5-GeopoliticalStressScore')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

MAX_FORWARD_DAYS = 10  # maximum trading days to look forward for a market open

# ── Load GSS scores ────────────────────────────────────────────────────────────

print('Loading GSS scores ...')
gss_raw = pd.read_csv(PROCESSED_DIR / 'gss_scores.csv', index_col=0, parse_dates=True)

gss_raw.index = pd.to_datetime(gss_raw.index, utc=True).tz_convert(None).normalize()
gss_raw.index = gss_raw.index.strftime('%Y-%m-%d')

gss_dict = {
    date_str: row['GSS']
    for date_str, row in gss_raw.iterrows()
    if pd.notna(row['GSS'])
}

trading_days    = sorted(gss_dict.keys())
trading_days_ts = pd.to_datetime(trading_days)

first_valid = trading_days[0]
print(f'  {len(gss_dict)} trading days with valid GSS scores')
print(f'  First valid date: {first_valid}')

# ── Load events ────────────────────────────────────────────────────────────────

print('\nLoading events ...')
events = pd.read_csv(BASE_DIR / 'data' / 'events.csv', parse_dates=['event_date'])

# ── Match each event to next available trading day on or after event date ──────

print('Matching events to next available trading day on or after event date ...\n')

matched_dates  = []
gss_on_event   = []
match_notes    = []

for _, row in events.iterrows():
    event_date = pd.to_datetime(row['event_date']).normalize()
    event_str  = event_date.strftime('%Y-%m-%d')

    # check if the event falls before the warmup window even completes
    if event_str < first_valid:
        matched_dates.append(None)
        gss_on_event.append(np.nan)
        match_notes.append('before warmup window — unmeasurable')
        print(f'  SKIP  {row["event_name"]} ({event_str}) — before first valid score ({first_valid})')
        continue

    # find the next trading day on or after the event date
    # this handles weekends, public holidays, and market closures (e.g. 9/11)
    future_days = trading_days_ts[trading_days_ts >= event_date]

    if len(future_days) == 0:
        matched_dates.append(None)
        gss_on_event.append(np.nan)
        match_notes.append('no future trading day found')
        print(f'  SKIP  {row["event_name"]} ({event_str}) — no future trading day')
        continue

    next_day     = future_days[0]
    next_day_str = next_day.strftime('%Y-%m-%d')
    days_forward = (next_day - event_date).days

    if days_forward > MAX_FORWARD_DAYS:
        matched_dates.append(None)
        gss_on_event.append(np.nan)
        match_notes.append(f'next trading day {days_forward} days away — too far')
        print(f'  SKIP  {row["event_name"]} — next trading day {days_forward} days forward')
        continue

    score = gss_dict.get(next_day_str, np.nan)
    matched_dates.append(next_day_str)
    gss_on_event.append(score)

    if days_forward == 0:
        match_notes.append('matched on event date')
    else:
        match_notes.append(f'market closed on event date — matched {days_forward} day(s) forward')
        print(f'  NOTE  {row["event_name"]} ({event_str}) → matched to {next_day_str} '
              f'({days_forward} day(s) forward — market was closed)')

events['trading_date'] = matched_dates
events['GSS_on_event'] = gss_on_event
events['match_note']   = match_notes

# ── Save ───────────────────────────────────────────────────────────────────────

events.to_csv(PROCESSED_DIR / 'events_with_gss.csv', index=False)

# ── Full ranking ───────────────────────────────────────────────────────────────

valid   = events.dropna(subset=['GSS_on_event'])
skipped = events[events['GSS_on_event'].isna()]

print(f'\n{len(valid)} events with valid GSS scores')
print(f'{len(skipped)} events skipped (pre-warmup window)')

if not skipped.empty:
    print('\nSkipped events:')
    print(skipped[['event_name', 'event_date', 'match_note']].to_string(index=False))

ranked = (
    valid[['event_name', 'trading_date', 'category', 'GSS_on_event', 'match_note']]
    .sort_values('GSS_on_event', ascending=False)
    .reset_index(drop=True)
)
ranked.index += 1

print('\nFull event ranking by GSS score (next trading day on or after event):\n')
print(ranked.to_string())

print(f'\nSaved → {PROCESSED_DIR / "events_with_gss.csv"}')
