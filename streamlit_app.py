# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present wiqilee

# streamlit_app.py — Crypto Sentiment Dashboard (UI) using sentiment.py core
# - Crypto-domain analysis + conclusion & recommendations (English)
# - Bigger, PDF-safe PNG export (no cropping, readable in print)
# - ETH dashed: white in UI, black in PDF
# - Centered "Not enough data..." placeholder
# - Collision-free labels on bars and time-series
# - Time-series lines rendered edge-to-edge in the app (slight pad only for PNG/PDF)
# - FIX: Explicit trace colors for PNG/PDF so lines always render
# - FIX: Force exported PNGs to RGB (remove alpha) so lines don't disappear in PDF

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image  # <<< for RGB-safe PNGs

from sentiment import (
    load_data as core_load_data,
    build_pdf_report as core_build_pdf,
    derive_crypto_takeaways,
)

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_CSV = "data/news_raw.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

st.set_page_config(
    page_title="Crypto Sentiment Dashboard",
    layout="wide",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="expanded",
)

LABEL_COLORS = {
    "BTC":   "#22c55e",
    "ETH":   "#ffffff",  # white in UI; black in PDF variant
    "SOL":   "#fde047",
    "OTHER": "#94a3b8",
}
LABEL_ORDER = ["BTC", "ETH", "SOL", "OTHER"]
SOURCE_COLORS = px.colors.qualitative.Set3 + px.colors.qualitative.Set2

# Legend on top to maximize horizontal plotting area
LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def fmtdt(ts) -> str:
    if pd.isna(ts):
        return "-"
    try:
        return pd.to_datetime(ts).tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")

def _force_rgb_png(path_png: str):
    """Ensure PNG has no alpha/palette so PDF embedding won't drop thin lines."""
    try:
        im = Image.open(path_png)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(path_png, format="PNG", optimize=True)
    except Exception as e:
        st.info(f"RGB convert skipped for {os.path.basename(path_png)}: {e}")

def ensure_png(fig: go.Figure, path_png: str, scale: float = 3.0):
    """
    Save a PNG copy with a large canvas and generous margins.
    IMPORTANT: Do not change styling here; the figure must already carry explicit colors/widths.
    """
    try:
        f = go.Figure(fig)  # clone
        f.update_layout(
            width=1300, height=780,
            margin=dict(l=130, r=110, t=110, b=110),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        f.update_traces(cliponaxis=False, opacity=1.0)
        f.write_image(path_png, scale=scale)  # requires kaleido
        _force_rgb_png(path_png)  # make PDF-safe
        return True
    except Exception as e:
        st.info(f"PNG export requires `kaleido` (pip install kaleido). Skipped: {e}")
        return False

def chip(label: str) -> str:
    color = LABEL_COLORS.get(label, "#9ca3af")
    dot = f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};margin-right:6px;vertical-align:middle'></span>"
    return f"{dot}{label}"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return core_load_data(path)

# -------------------------------------------------
# Charts
# -------------------------------------------------
def _pad_range(xmin: float, xmax: float) -> tuple[float, float]:
    span = xmax - xmin if np.isfinite(xmax - xmin) else 2.0
    pad = max(0.08, 0.06 * span)
    return xmin - pad, xmax + pad

def bar_sentiment_by_source(dff: pd.DataFrame):
    agg = dff.groupby("source", dropna=False)["compound"].mean().reset_index().sort_values("compound")
    cmap = {src: SOURCE_COLORS[i % len(SOURCE_COLORS)] for i, src in enumerate(agg["source"])}
    fig = px.bar(
        agg, x="compound", y="source", orientation="h",
        labels={"compound": "Avg VADER compound", "source": "Source"},
        text=agg["compound"].round(3).astype(str),
        color="source", color_discrete_map=cmap
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=12,
        marker_line_color="#0f172a",
        marker_line_width=0.6,
        showlegend=False,
        cliponaxis=False,
    )
    if len(agg):
        xmin, xmax = float(agg["compound"].min()), float(agg["compound"].max())
        xmin, xmax = _pad_range(xmin, xmax)
        fig.update_xaxes(range=[xmin, xmax], automargin=True, zeroline=True, zerolinewidth=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=22))
    return fig

def bar_sentiment_by_label(dff: pd.DataFrame):
    agg = dff.groupby("label", dropna=False)["compound"].mean().reset_index()
    agg["__ord"] = agg["label"].apply(lambda x: LABEL_ORDER.index(x) if x in LABEL_ORDER else 999)
    agg = agg.sort_values("__ord").drop(columns="__ord")

    vals = agg["compound"].astype(float).values
    text_pos = ["inside" if v < 0 else "outside" for v in vals]
    text_colors = ["#111111" if (v < 0 and lab in {"SOL", "OTHER"}) else None
                   for lab, v in zip(agg["label"], vals)]

    fig = px.bar(
        agg, x="compound", y="label", orientation="h",
        labels={"compound": "Avg VADER compound", "label": "Label"},
        text=agg["compound"].round(3).astype(str),
        color="label", color_discrete_map=LABEL_COLORS,
    )
    fig.update_traces(
        textposition=text_pos,
        textfont_size=12,
        marker_line_color="#0f172a",
        marker_line_width=1.0,
        showlegend=False,
        cliponaxis=False,
    )
    for i, c in enumerate(text_colors):
        if c:
            fig.data[i].textfont = dict(color=c, size=12)

    xmin, xmax = (float(np.nanmin(vals)) if len(vals) else -1.0,
                  float(np.nanmax(vals)) if len(vals) else  1.0)
    xmin, xmax = _pad_range(xmin, xmax)
    fig.update_xaxes(range=[xmin, xmax], automargin=True, zeroline=True, zerolinewidth=1)

    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=22))
    return fig, agg

def _apply_explicit_colors(fig: go.Figure, line_colors: dict, theme_for_pdf: bool):
    """Make every trace fully explicit (line+marker color/width/mode)."""
    for tr in fig.data:
        name = getattr(tr, "name", None)
        if not name:
            continue
        col = line_colors.get(name, "#222222")
        dash = "dot" if name == "ETH" else "solid"
        outline = "#000000" if theme_for_pdf else "#0f172a"
        tr.update(
            mode="lines+markers",
            line=dict(color=col, width=4.5, dash=dash),
            marker=dict(color=col, size=9, line=dict(width=1.2, color=outline)),
            opacity=1.0,
        )

def timeseries_by_label(dff: pd.DataFrame, theme_for_pdf: bool=False, show_end_labels: bool=False):
    if dff.empty:
        return px.line()
    ts = dff.copy()
    ts["date"] = pd.to_datetime(ts["publishedat"]).dt.tz_convert(None).dt.floor("D")
    g = ts.groupby(["date", "label"], dropna=False)["compound"].mean().reset_index()

    line_colors = LABEL_COLORS.copy()
    if theme_for_pdf:
        line_colors["ETH"] = "#111111"  # black in PDF

    fig = px.line(
        g, x="date", y="compound", color="label",
        color_discrete_map=line_colors, markers=True,
        labels={"date": "Date", "compound": "Avg VADER compound", "label": "Label"},
    )
    _apply_explicit_colors(fig, line_colors, theme_for_pdf)

    if show_end_labels:
        last_vals = g.sort_values("date").groupby("label").tail(1)
        yshift = {"BTC": 16, "ETH": -18, "SOL": 16, "OTHER": -10}
        for _, row in last_vals.iterrows():
            fig.add_annotation(
                x=row["date"], y=row["compound"],
                text=f"{row['label']} {row['compound']:.3f}",
                showarrow=False,
                yshift=yshift.get(row["label"], 0),
                font=dict(color="#111111" if theme_for_pdf else "#e5e7eb", size=12),
            )

    dmin, dmax = pd.to_datetime(g["date"].min()), pd.to_datetime(g["date"].max())
    span = (dmax - dmin) if pd.notna(dmax) and pd.notna(dmin) else pd.Timedelta(days=1)
    pad = max(pd.Timedelta(hours=6), span * 0.03)

    if theme_for_pdf:
        rmin, rmax = dmin - pad, dmax + pad
        fig.update_xaxes(type="date", range=[rmin, rmax], automargin=True,
                         title_standoff=20, tickformat="%b %d, %Y")
    else:
        fig.update_xaxes(type="date", range=[dmin, dmax], constrain="domain", automargin=True,
                         title_standoff=20, tickformat="%b %d, %Y")

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=LEGEND_TOP,
        yaxis=dict(range=[-1.05, 1.05], title_standoff=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def heatmap_corr_by_source(dff: pd.DataFrame, theme_for_pdf: bool=False):
    tmp = dff.copy()
    tmp["date"] = pd.to_datetime(tmp["publishedat"]).dt.tz_convert(None).dt.date
    daily = tmp.groupby(["date", "source"], dropna=False)["compound"].mean().reset_index()
    pivot = daily.pivot(index="date", columns="source", values="compound")

    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        text_color = "#e5e7eb" if not theme_for_pdf else "#111111"
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Not enough data for correlation\n(≥3 days & ≥2 sources)",
            showarrow=False, font=dict(color=text_color, size=16), align="center",
            bgcolor="rgba(0,0,0,0)" if not theme_for_pdf else "rgba(255,255,255,0)",
        )
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        fig.update_layout(
            height=420,
            margin=dict(l=80, r=80, t=60, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    corr = pivot.corr(min_periods=3)
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    labels=dict(color="corr"), aspect="auto")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=18),
                      legend=LEGEND_TOP)
    return fig

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Filters")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset", use_container_width=True):
            st.experimental_rerun()
    with c2:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.experimental_rerun()

df = load_data(DATA_CSV)

with st.sidebar.expander("Labels", expanded=True):
    labels_sel = st.multiselect("Filter by label", LABEL_ORDER, default=LABEL_ORDER)

with st.sidebar.expander("Source", expanded=True):
    sources_all = sorted(df["source"].astype(str).unique().tolist())
    sources_sel = st.multiselect("Filter by source", sources_all, default=sources_all)

with st.sidebar.expander("Date range (UTC)", expanded=False):
    min_dt = df["publishedat"].min().tz_convert(None)
    max_dt = df["publishedat"].max().tz_convert(None)
    date_from, date_to = st.date_input("Date range", value=(min_dt.date(), max_dt.date()))
    if isinstance(date_from, tuple):
        date_from, date_to = date_from

with st.sidebar.expander("Display options", expanded=True):
    show_end_ann = st.checkbox("Show end annotations (time-series)", value=True)
    show_index   = st.checkbox("Show table row index (UI)", value=False)  # <<< toggle

# -------------------------------------------------
# Apply filters
# -------------------------------------------------
dff = df.copy()
if labels_sel:
    dff = dff[dff["label"].isin(labels_sel)]
if sources_sel:
    dff = dff[dff["source"].isin(sources_sel)]
if date_from and date_to:
    mask_dt = (
        dff["publishedat"].dt.tz_convert(None).dt.date >= pd.to_datetime(date_from).date()
    ) & (
        dff["publishedat"].dt.tz_convert(None).dt.date <= pd.to_datetime(date_to).date()
    )
    dff = dff[mask_dt]

# -------------------------------------------------
# Header
# -------------------------------------------------
cA, cB, cC = st.columns([0.7, 0.2, 0.1])
with cA:
    st.title("Crypto Sentiment Dashboard")
with cB:
    st.caption(f"Rows: {len(df)} | File: `{DATA_CSV}`")
with cC:
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.caption(
    f"{len(dff)} results after filters "
    f"(UTC {fmtdt(dff['publishedat'].min())} ➜ {fmtdt(dff['publishedat'].max())})"
)

# -------------------------------------------------
# Table (HTML)
# -------------------------------------------------
st.subheader("📰 Latest Crypto News")
show_cols = ["publishedat", "channel", "source", "label", "title", "url", "compound"]
dshow = dff[show_cols].copy()
dshow["label"] = dshow["label"].apply(chip)
dshow["url"] = dshow["url"].apply(lambda u: f"<a href='{u}' target='_blank'>open</a>"
                                  if isinstance(u, str) and u.startswith("http") else "")

# build styler and optionally hide index (compat with old pandas)
styler = (dshow.style
          .format(na_rep="")
          .set_table_styles([
              dict(selector="th", props=[("text-align","left")]),
              dict(selector="td", props=[("text-align","left"), ("vertical-align","top")]),
          ]))
if not show_index:
    try:
        styler = styler.hide_index()        # pandas >= 2.1
    except Exception:
        try:
            styler = styler.hide(axis="index")  # pandas < 2.1
        except Exception:
            pass

html_tbl = styler.to_html(escape=False)
st.markdown(html_tbl, unsafe_allow_html=True)

# -------------------------------------------------
# Charts + PNG (for PDF)
# -------------------------------------------------
st.subheader("📊 Sentiment Score by News Source")
fig_src = bar_sentiment_by_source(dff)
st.plotly_chart(fig_src, use_container_width=True)
png_src = os.path.join(CHART_DIR, "sentiment_by_source.png")
ensure_png(fig_src, png_src)
st.caption("Higher bar = more positive mean VADER compound score (−1..+1).")

st.subheader("📊 Sentiment by Label (colored)")
fig_lbl, _ = bar_sentiment_by_label(dff)
st.plotly_chart(fig_lbl, use_container_width=True)
png_lbl = os.path.join(CHART_DIR, "sentiment_by_label.png")
ensure_png(fig_lbl, png_lbl)
st.caption("Average sentimental tone per label. BTC green, ETH dashed in PDF, SOL yellow, OTHER slate.")

st.subheader("⏱️ Time-series by Label (daily mean)")
fig_ts_ui = timeseries_by_label(dff, theme_for_pdf=False, show_end_labels=show_end_ann)
st.plotly_chart(fig_ts_ui, use_container_width=True)
fig_ts_pdf = timeseries_by_label(dff, theme_for_pdf=True, show_end_labels=show_end_ann)
png_ts = os.path.join(CHART_DIR, "timeseries_by_label.png")
ensure_png(fig_ts_pdf, png_ts)
st.caption("Daily averages by label; ETH dashed for visibility. Toggle end labels in the sidebar.")

st.subheader("🧩 Source Correlation Heatmap (daily mean)")
fig_hm_ui = heatmap_corr_by_source(dff, theme_for_pdf=False)
st.plotly_chart(fig_hm_ui, use_container_width=True)
fig_hm_pdf = heatmap_corr_by_source(dff, theme_for_pdf=True)
png_hm = os.path.join(CHART_DIR, "source_corr_heatmap.png")
ensure_png(fig_hm_pdf, png_hm)
st.caption("Needs ≥3 distinct dates and ≥2 sources to compute correlations.")

# -------------------------------------------------
# Summary + Domain analysis
# -------------------------------------------------
st.subheader("🧾 Summary (Research-ready)")
n = len(dff)
avg = float(dff["compound"].mean()) if n else float("nan")
median = float(dff["compound"].median()) if n else float("nan")
std = float(dff["compound"].std()) if n else float("nan")

pos_mask = dff["compound"] > 0.05
neu_mask = (dff["compound"] >= -0.05) & (dff["compound"] <= 0.05)
neg_mask = dff["compound"] < -0.05
pos_n, neu_n, neg_n = int(pos_mask.sum()), int(neu_mask.sum()), int(neg_mask.sum())

def pct(x): return 0.0 if n == 0 else 100.0 * x / n

m1, m2, m3, m4 = st.columns(4)
m1.metric("Articles (n)", f"{n}")
m2.metric("Avg / Median", f"{avg:.3f} / {median:.3f}")
m3.metric("Std dev", f"{std:.3f}")
m4.metric("Positive / Neutral / Negative", f"{pos_n} / {neu_n} / {neg_n}",
          help="Thresholds: positive > 0.05, neutral −0.05..0.05, negative < −0.05")

# Per-label mean table (UI) — hide index via toggle
agg_lbl = dff.groupby("label", dropna=False)["compound"].mean().reset_index() \
             .rename(columns={"compound":"avg_compound"})
agg_lbl["avg_compound"] = agg_lbl["avg_compound"].round(3)

st.markdown("**Per-label (mean compound)** — *Average sentiment per label*")
st.caption("How to read: **> 0.05 = Positive**, **< −0.05 = Negative**, in between = Neutral.")
st.dataframe(agg_lbl.sort_values("label"),
             use_container_width=True,
             hide_index=not show_index)

# Domain analysis (UI)
take = derive_crypto_takeaways(dff)
st.subheader("🔎 Crypto analysis")
st.markdown(take["analysis_md"])
st.subheader("🧭 Recommendations")
st.markdown(take["recommendations_md"])
st.subheader("✅ Conclusion")
st.write(take["conclusion_md"])

# -------------------------------------------------
# Downloads (CSV & PDF)
# -------------------------------------------------
st.subheader("⬇️ Downloads")
csv_bytes = dff.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv_bytes, "news_filtered.csv", "text/csv", use_container_width=True)

meta = {
    "n": n, "avg": avg, "median": median, "std": std,
    "pos_n": pos_n, "pos_pct": pct(pos_n),
    "neu_n": neu_n, "neu_pct": pct(neu_n),
    "neg_n": neg_n, "neg_pct": pct(neg_n),
    "data_file": DATA_CSV,
    "date_range": f"{fmtdt(dff['publishedat'].min())} – {fmtdt(dff['publishedat'].max())}" if n else "-",
    "insight_text": take["insight_text"],
    "analysis_md": take["analysis_md"],
    "recommendations_md": take["recommendations_md"],
    "conclusion_md": take["conclusion_md"],
}
pngs = {
    "by_source": os.path.join(CHART_DIR, "sentiment_by_source.png"),
    "by_label":  os.path.join(CHART_DIR, "sentiment_by_label.png"),
    "ts_label":  os.path.join(CHART_DIR, "timeseries_by_label.png"),
    "heatmap":   os.path.join(CHART_DIR, "source_corr_heatmap.png"),
}

if st.button("Build PDF report", type="primary", use_container_width=True):
    try:
        pdf_bytes = core_build_pdf(dff, agg_lbl, pngs, meta)
        st.download_button("Download PDF report", pdf_bytes,
                           "crypto_sentiment_report.pdf", "application/pdf", use_container_width=True)
    except Exception as e:
        st.error(f"PDF build failed: {e}")

st.caption(
    "Exported PNGs are converted to RGB (no alpha) and use explicit per-trace colors so lines are guaranteed to appear in the PDF. "
    "Charts use a large canvas and extra margins (no cropped/overlapping labels)."
)
