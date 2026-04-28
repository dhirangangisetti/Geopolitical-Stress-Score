import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r'C:\Projects\P5-GeopoliticalStressScore\data\processed')

df = pd.read_csv(PROCESSED_DIR / 'events_with_gss.csv')

ranked = (
    df[['event_name', 'trading_date', 'category', 'GSS_on_event']]
    .sort_values('GSS_on_event', ascending=False)
    .reset_index(drop=True)
)

ranked.index += 1  # start ranking from 1
print('\nFull event ranking by GSS score on event date:\n')
print(ranked.to_string())
