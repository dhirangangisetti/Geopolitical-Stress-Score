import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Geopolitical Stress Score",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #21262d; }
  h1, h2, h3, h4                     { color: #e6edf3; }
  .metric-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px 18px; text-align: center;
  }
  .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 26px; font-weight: 700; margin-top: 4px; }
  .metric-sub   { font-size: 11px; color: #8b949e; margin-top: 2px; }
  .story-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 18px 22px; margin-bottom: 12px;
    color: #c9d1d9; line-height: 1.6; font-size: 14.5px;
  }
  .story-card b { color: #e6edf3; }
  .subtitle { color: #8b949e; font-size: 15px; margin-top: -6px; margin-bottom: 18px; }
  .section-hint { color: #8b949e; font-size: 13px; margin-bottom: 12px; }
  div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ASSET_MAP = {
    "Composite GSS":    None,
    "S&P 500":          "SP500",
    "Gold":             "Gold",
    "WTI Crude Oil":    "Oil_WTI",
    "VIX":              "VIX",
    "10Y Treasury":     "Treasury_10Y",
    "USD Index":        "USD_Index",
    "Bitcoin":          "Bitcoin",
    "Emerging Markets": "EM_Equities",
    "High Yield Bonds": "HighYield",
    "EUR / USD":        "EURUSD",
    "Japanese Yen":     "JPY_USD",
}

ASSET_REFERENCE = [
    ("S&P 500",          "-1 (flipped)",   "Equity sell-off"),
    ("Gold",             "+1",             "Safe-haven demand"),
    ("WTI Crude Oil",    "+1",             "Supply shock / energy stress"),
    ("VIX",              "+1 (raw level)", "Implied equity volatility spike"),
    ("10Y Treasury",     "-1 (flipped)",   "Yields falling -> flight to quality"),
    ("USD Index",        "+1",             "Dollar strength -> reserve demand"),
    ("Bitcoin",          "-1 (flipped)",   "Crypto risk-off"),
    ("Emerging Markets", "-1 (flipped)",   "EM equity stress"),
    ("High Yield Bonds", "-1 (flipped)",   "Credit spreads widening"),
    ("EUR / USD",        "-1 (flipped)",   "Euro weakness vs dollar"),
    ("Japanese Yen",     "-1 (flipped)",   "Yen weakness vs dollar"),
]

CATEGORY_COLORS = {
    "Military/Conflict":       "#f85149",
    "Financial Crisis":        "#ffa657",
    "Sanctions & Trade War":   "#d2a8ff",
    "Political/Institutional": "#79c0ff",
    "Systemic/Supply Chain":   "#e3b341",
    "Pandemic":                "#56d364",
    "Natural Disaster":        "#58a6ff",
    "Market Crash":            "#ff6b6b",
}
ALL_CATEGORIES = list(CATEGORY_COLORS.keys())


@st.cache_data
def load_data():
    def clean_index(df):
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
        return df
    gss     = clean_index(pd.read_csv(PROCESSED_DIR / "gss_scores.csv",  index_col=0, parse_dates=True))
    zscores = clean_index(pd.read_csv(PROCESSED_DIR / "zscores.csv",     index_col=0, parse_dates=True))
    events  = pd.read_csv(PROCESSED_DIR / "events_with_gss.csv", parse_dates=["event_date"])
    events  = events.dropna(subset=["trading_date", "GSS_on_event"])
    events["trading_date"] = pd.to_datetime(events["trading_date"])
    return gss, zscores, events

gss_df, zscores_df, events_df = load_data()


def score_color(v):
    if pd.isna(v):  return "#8b949e"
    if v >= 2.0:    return "#f85149"
    elif v >= 1.0:  return "#ffa657"
    elif v >= 0.5:  return "#e3b341"
    elif v >= 0:    return "#56d364"
    else:           return "#8b949e"


# ==============================================================================
# LAYER 1 - THE STORY
# ==============================================================================

st.markdown("# Geopolitical Stress Score")
st.markdown("<div class='subtitle'>A weight-free, cross-asset stress gauge covering 25 years of geopolitical shocks.</div>", unsafe_allow_html=True)

st.markdown("## The premise")
st.markdown("""
<div class='story-card'>
Market headlines are noisy. Financial media labels dozens of events each year as
"crises," but most do not move cross-asset risk in a measurable way - and a few
that no one was watching turn out to be the ones that matter.
<br><br>
The Geopolitical Stress Score (GSS) is an attempt to separate signal from noise
using only price data, no narrative weighting, and no subjective asset importance.
It expresses each trading day as a single number: positive means markets are under
stress, negative means conditions are benign relative to the prior trading year.
<br><br>
Because the score is built from rolling z-scores across eleven globally traded
assets - equities, bonds, commodities, FX, credit, and crypto - it captures stress
that shows up <i>across</i> markets, not just in one corner of the system.
</div>
""", unsafe_allow_html=True)

st.markdown("## What the data says")
top_events  = events_df.sort_values("GSS_on_event", ascending=False).head(5)
worst_event = top_events.iloc[0]

st.markdown(f"""
<div class='story-card'>
<b>The biggest cross-market shock since 2001 was the Lehman collapse</b>
(15 Sep 2008) with a composite GSS of
<b style='color:#f85149'>{worst_event["GSS_on_event"]:.2f}</b> - more than two
standard deviations above the prior year's cross-asset behaviour. COVID being
declared a pandemic in March 2020 was second at ~1.14.
<br><br>
<b>Several "canonical" geopolitical events barely registered on markets.</b>
The Iraq War (2003), Brexit (2016), the 6 January Capitol insurrection (2021),
and the 2024 US election all produced <i>negative</i> GSS readings - markets
had priced in the outcome or rallied on it. Headline severity does not equal
market stress.
<br><br>
<b>The score surfaces events that were not on the front page.</b> The Turkey
lira crisis in August 2018 and the US chip export controls on China in October
2022 both rank in the top five - not because of the headlines, but because of
how broadly they transmitted across equity, credit, and currency markets.
</div>
""", unsafe_allow_html=True)

st.markdown("### The score across 25 years")
composite = gss_df["GSS"].dropna()
hero = go.Figure()
hero.add_trace(go.Scatter(
    x=composite.index, y=composite.values, mode="lines",
    line=dict(color="#58a6ff", width=1.2), name="Composite GSS",
    hovertemplate="<b>%{x|%d %b %Y}</b><br>GSS: %{y:.3f}<extra></extra>",
))
for i, (_, row) in enumerate(top_events.iterrows()):
    ay = -55 if i % 2 == 0 else -30
    hero.add_annotation(
        x=row["trading_date"], y=row["GSS_on_event"],
        text=f"<b>{row['event_name']}</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
        arrowcolor="#8b949e", ax=0, ay=ay,
        font=dict(size=10, color="#e6edf3"),
        bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1,
    )
hero.add_hline(y=1.0, line=dict(color="rgba(255,166,87,0.4)", width=1, dash="dot"),
               annotation_text="Elevated", annotation_position="right",
               annotation=dict(font=dict(size=10, color="#ffa657")))
hero.add_hline(y=2.0, line=dict(color="rgba(248,81,73,0.5)", width=1, dash="dot"),
               annotation_text="Crisis", annotation_position="right",
               annotation=dict(font=dict(size=10, color="#f85149")))
hero.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(family="Segoe UI", color="#8b949e", size=12),
    height=360, margin=dict(l=10, r=90, t=20, b=10),
    hovermode="x unified", showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(title="Composite GSS", showgrid=True, gridcolor="#21262d",
               zeroline=True, zerolinecolor="#30363d"),
)
st.plotly_chart(hero, use_container_width=True)
st.caption("Full window, Composite GSS only, top-5 events labelled.")

st.markdown("## Methodology")
st.markdown("""
<div class='story-card'>
<b>Rolling 252-day baseline.</b> Each asset's daily return is converted into a
z-score using the mean and standard deviation of the prior 252 trading days
(about one trading year). This creates a rolling comparison window that adapts
to each era's volatility regime.<br><br>
<b>Direction-adjustment so positive always means stress.</b> An equity sell-off
and a gold rally are both stress signals. The S&amp;P 500, treasuries, and risk-on
assets are sign-flipped so positive = stress, negative = calm.<br><br>
<b>Equal-weighted composite.</b> The GSS is the simple average of whichever
assets have valid z-scores on a given day - minimum 6 of 11 required. No
judgment-weighted importance, no regime-specific weights, no machine learning.<br><br>
<b>Event matching.</b> Events are matched to the next available trading day
on or after the event date. Maximum lookahead is 10 trading days.
</div>
""", unsafe_allow_html=True)

st.markdown("## What each asset score means")
ref_df = pd.DataFrame(ASSET_REFERENCE, columns=["Asset", "Direction", "A high score means..."])
ref_df.index = ref_df.index + 1
st.dataframe(
    ref_df.style.set_properties(**{"background-color": "#161b22", "color": "#e6edf3", "border-color": "#30363d"}),
    use_container_width=True, height=420,
)
st.caption("Direction is applied before averaging. A +1 asset with a high z-score and a -1 asset with a low z-score both contribute positively to the GSS.")
st.divider()


# ==============================================================================
# LAYER 2 - EVENTS TABLE
# ==============================================================================

st.markdown("## Events table")
st.markdown("<div class='section-hint'>Every scored event in the database. Filter by category or search by name.</div>", unsafe_allow_html=True)

tf1, tf2, tf3 = st.columns([2, 2, 1])
with tf1:
    table_cats = st.multiselect("Category", options=ALL_CATEGORIES, default=ALL_CATEGORIES, key="table_cats")
with tf2:
    name_search = st.text_input("Search event name", value="", placeholder="e.g. Lehman, COVID, Turkey...", key="table_search")
with tf3:
    table_sort = st.selectbox("Sort", ["GSS score (high to low)", "Date - newest first", "Date - oldest first"], key="table_sort")

table_src = events_df.copy()
if table_cats:
    table_src = table_src[table_src["category"].isin(table_cats)]
else:
    table_src = table_src.iloc[0:0]
if name_search.strip():
    table_src = table_src[table_src["event_name"].str.contains(name_search.strip(), case=False, na=False)]

if table_src.empty:
    st.info("No events match the current filters.")
else:
    if table_sort == "GSS score (high to low)":
        table_src = table_src.sort_values("GSS_on_event", ascending=False)
    elif table_sort == "Date - newest first":
        table_src = table_src.sort_values("trading_date", ascending=False)
    else:
        table_src = table_src.sort_values("trading_date", ascending=True)

    table = table_src[["event_name", "event_date", "category", "GSS_on_event", "match_note"]].copy()
    table.columns = ["Event", "Date", "Category", "GSS Score", "Note"]
    table["Date"]      = pd.to_datetime(table["Date"]).dt.strftime("%d %b %Y")
    table["GSS Score"] = table["GSS Score"].round(3)
    table = table.reset_index(drop=True)
    table.index += 1

    def color_score(val):
        try:
            v = float(val)
            if v >= 2.0:   return "color: #f85149; font-weight: 700"
            elif v >= 1.0: return "color: #ffa657; font-weight: 700"
            elif v >= 0.5: return "color: #e3b341"
            elif v >= 0:   return "color: #56d364"
            else:          return "color: #8b949e"
        except Exception:
            return ""

    def color_category(val):
        return f"color: {CATEGORY_COLORS.get(val, '#8b949e')}"

    styled = (
        table.style
        .map(color_score,    subset=["GSS Score"])
        .map(color_category, subset=["Category"])
        .set_properties(**{"background-color": "#161b22", "color": "#e6edf3", "border-color": "#30363d"})
    )
    st.dataframe(styled, use_container_width=True, height=460)

st.divider()


# ==============================================================================
# LAYER 3 - INTERACTIVE CHART
# ==============================================================================

st.markdown("## Interactive chart")
st.markdown("<div class='section-hint'>Asset selector, date range, and category filter live in the sidebar. These controls affect this chart only.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Chart controls")
    st.caption("These controls drive the interactive chart in Section 3 only.")
    st.markdown("---")
    selected_asset = st.selectbox("Asset or composite", options=list(ASSET_MAP.keys()), index=0)
    st.markdown("**Event categories**")
    selected_categories = st.multiselect(
        label="Filter by category", options=ALL_CATEGORIES, default=ALL_CATEGORIES, label_visibility="collapsed"
    )
    st.markdown("---")
    show_events    = st.toggle("Show event markers",           value=True)
    show_eventline = st.toggle("Show vertical event lines",    value=False)
    show_reflines  = st.toggle("Show Elevated / Crisis lines", value=True)
    st.markdown("---")
    min_date   = gss_df.index.min().date()
    max_date   = gss_df.index.max().date()
    date_range = st.slider("Date range", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="YYYY")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#484f58; line-height:1.7'>"
        "Source: Yahoo Finance via yfinance<br>Rolling window: 252 trading days<br>"
        "Assets: 11 (equal weight)<br>Events: 31 scored (2001-2024)</div>",
        unsafe_allow_html=True,
    )

col_key      = ASSET_MAP[selected_asset]
start_dt     = pd.Timestamp(date_range[0])
end_dt       = pd.Timestamp(date_range[1])
is_composite = col_key is None

if is_composite:
    series       = gss_df["GSS"].loc[start_dt:end_dt].dropna()
    asset_counts = gss_df["assets_in_composite"].loc[start_dt:end_dt].reindex(series.index)
else:
    series       = zscores_df[col_key].loc[start_dt:end_dt].dropna()
    asset_counts = None

filtered_events = events_df[
    events_df["category"].isin(selected_categories) &
    (events_df["trading_date"] >= start_dt) &
    (events_df["trading_date"] <= end_dt)
].copy()

if not series.empty:
    cur        = series.iloc[-1]
    peak       = series.max()
    peak_date  = series.idxmax().strftime("%b %Y")
    mean       = series.mean()
    n_assets   = int(asset_counts.iloc[-1]) if is_composite and asset_counts is not None else "-"
    asset_note = f"{n_assets} of 11 assets" if is_composite else "Direction-adjusted z-score"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Current score",  cur,                  series.index[-1].strftime("%d %b %Y")),
        (c2, "Peak score",     peak,                 peak_date),
        (c3, "Mean score",     mean,                 "selected period"),
        (c4, "Events shown",   len(filtered_events), asset_note),
    ]:
        is_events_col = label == "Events shown"
        v_color   = "#79c0ff" if is_events_col else score_color(val)
        v_display = val if is_events_col else f"{val:.3f}"
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{v_color}">{v_display}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=series.index, y=series.values, mode="lines", name=selected_asset,
    line=dict(color="#58a6ff" if is_composite else "#79c0ff", width=1.8),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>" + selected_asset + ": %{y:.3f}<extra></extra>",
))

if show_reflines and not series.empty:
    fig.add_hline(y=1.0, line=dict(color="rgba(255,166,87,0.45)", width=1, dash="dot"),
                  annotation_text="Elevated", annotation_position="right",
                  annotation=dict(font=dict(size=10, color="#ffa657"), bgcolor="rgba(13,17,23,0.7)"))
    fig.add_hline(y=2.0, line=dict(color="rgba(248,81,73,0.55)", width=1, dash="dot"),
                  annotation_text="Crisis", annotation_position="right",
                  annotation=dict(font=dict(size=10, color="#f85149"), bgcolor="rgba(13,17,23,0.7)"))

if is_composite and asset_counts is not None and not series.empty:
    fig.add_trace(go.Scatter(
        x=asset_counts.index, y=asset_counts.values, mode="lines",
        name="Assets in composite",
        line=dict(color="rgba(255,255,255,0.08)", width=1),
        yaxis="y2", hovertemplate="Assets in composite: %{y}<extra></extra>", showlegend=False,
    ))

if show_events and not filtered_events.empty:
    series_dict = series.to_dict()
    for category in ALL_CATEGORIES:
        if category not in selected_categories:
            continue
        cat_ev = filtered_events[filtered_events["category"] == category]
        if cat_ev.empty:
            continue
        colour = CATEGORY_COLORS[category]
        dates, ys, hovers = [], [], []
        for _, row in cat_ev.iterrows():
            d = row["trading_date"]
            if d in series_dict:
                y = series_dict[d]
            else:
                idx = series.index.get_indexer([d], method="nearest")[0]
                y   = series.iloc[idx] if 0 <= idx < len(series) else np.nan
            note        = str(row.get("match_note", ""))
            closed_note = f"<br><i style='color:#8b949e'>{note}</i>" if "market closed" in note else ""
            dates.append(d)
            ys.append(y)
            hovers.append(
                f"<b>{row['event_name']}</b><br>"
                f"Date: {row['event_date'].strftime('%d %b %Y')}<br>"
                f"Category: {row['category']}<br>"
                f"GSS on event: {row['GSS_on_event']:.3f}{closed_note}"
            )
        fig.add_trace(go.Scatter(
            x=dates, y=ys, mode="markers", name=category,
            marker=dict(symbol="diamond", size=9, color=colour, line=dict(color="#0d1117", width=1)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers, legendgroup=category,
        ))
        if show_eventline:
            for d in dates:
                fig.add_vline(x=pd.Timestamp(d).timestamp() * 1000,
                              line=dict(color=colour, width=0.7, dash="dash"), opacity=0.3)

y_label = ("Composite GSS Score" if is_composite
           else f"{selected_asset} - Z-Score (direction-adjusted)")

layout_kwargs = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(family="Segoe UI", color="#8b949e", size=12),
    height=500, margin=dict(l=10, r=90, t=30, b=10),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1,
                font=dict(size=11), x=1.01, y=1, xanchor="left"),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
    yaxis=dict(title=dict(text=y_label, font=dict(size=11)),
               showgrid=True, gridcolor="#21262d",
               zeroline=True, zerolinecolor="#30363d", tickfont=dict(size=11)),
)
if is_composite:
    layout_kwargs["yaxis2"] = dict(
        overlaying="y", side="right", range=[0, 14], showgrid=False, showticklabels=False,
    )
fig.update_layout(**layout_kwargs)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Equal-weight average of available asset z-scores - positive = market stress."
    if is_composite else
    f"Rolling 252-day z-score - direction-adjusted so positive always = stress for {selected_asset}."
)
