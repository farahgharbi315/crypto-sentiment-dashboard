# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present wiqilee


# sentiment.py — data loader, crypto-domain analysis, print-quality charts (matplotlib), and PDF (reportlab).
# No Streamlit imports here.

import os
import io
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

# ---------- ReportLab (PDF) ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlcolors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, KeepTogether, ListFlowable, ListItem
    )
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------- PIL ----------
try:
    from PIL import Image
except Exception:
    Image = None

# ---------- Matplotlib (print charts for PDF) ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


# =========================
# Config
# =========================
DATA_CSV = "data/news_raw.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

LABEL_COLORS = {
    "BTC":   "#22c55e",
    "ETH":   "#f8fafc",
    "SOL":   "#fde047",
    "OTHER": "#3b82f6",
}
LABEL_ORDER = ["BTC", "ETH", "SOL", "OTHER"]

LABEL_COLORS_BARS = {
    "BTC":   "#22c55e",
    "ETH":   "#64748b",
    "SOL":   "#fbbf24",
    "OTHER": "#3b82f6",
}


# =========================
# Data loading
# =========================
def load_data(path: str = DATA_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing file: {path}")

    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in low:
                return low[n]
        return None

    c_pub = pick("publishedat", "published", "publishdate", "date", "datetime", "time")
    c_chn = pick("channel", "source_channel", "provider")
    c_src = pick("source", "domain", "site")
    c_lbl = pick("label", "symbol", "ticker", "coin", "asset")
    c_cmp = pick("compound", "vader_compound", "sentiment", "score")
    c_tit = pick("title", "headline")
    c_desc = pick("description", "summary", "desc")
    c_url = pick("url", "link")

    ren = {}
    if c_pub and c_pub != "publishedat": ren[c_pub] = "publishedat"
    if c_chn and c_chn != "channel":     ren[c_chn] = "channel"
    if c_src and c_src != "source":      ren[c_src] = "source"
    if c_lbl and c_lbl != "label":       ren[c_lbl] = "label"
    if c_cmp and c_cmp != "compound":    ren[c_cmp] = "compound"
    if c_tit and c_tit != "title":       ren[c_tit] = "title"
    if c_desc and c_desc != "description": ren[c_desc] = "description"
    if c_url and c_url != "url":         ren[c_url] = "url"
    if ren:
        df = df.rename(columns=ren)

    need_min = ["publishedat", "source", "compound"]
    miss = [c for c in need_min if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing column(s): {miss}")

    pub = pd.to_datetime(df["publishedat"], errors="coerce")
    try:
        if pub.dt.tz is None:
            pub = pub.dt.tz_localize("UTC")
    except Exception:
        pub = pd.to_datetime(df["publishedat"], errors="coerce").dt.tz_localize("UTC")
    df["publishedat"] = pub

    if "channel" not in df.columns:
        df["channel"] = "newsapi"
    for col, default in [("title", ""), ("description", ""), ("url", "")]:
        if col not in df.columns:
            df[col] = default

    df["source"] = df["source"].astype(str)
    df["compound"] = pd.to_numeric(df["compound"], errors="coerce").clip(-1, 1)

    if "label" not in df.columns:
        df["label"] = infer_label_from_text(df["title"], df["description"])
    else:
        df["label"] = df["label"].astype(str)

    df["label_norm"] = df["label"].str.upper().str.strip()
    key_map = {"BTC":"BTC","BITCOIN":"BTC","ETH":"ETH","ETHEREUM":"ETH","SOL":"SOL","SOLANA":"SOL"}
    df["label_norm"] = df["label_norm"].map(lambda x: key_map.get(x, x))
    df.loc[~df["label_norm"].isin(LABEL_ORDER), "label_norm"] = "OTHER"
    df["label"] = df["label_norm"]
    df.drop(columns=["label_norm"], inplace=True, errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]

    return df.sort_values("publishedat", ascending=False).reset_index(drop=True)


def infer_label_from_text(title: pd.Series, desc: pd.Series) -> pd.Series:
    t = (title.fillna("") + " " + desc.fillna("")).str.lower()
    is_btc = t.str.contains(r"\bbtc\b|\bbitcoin\b", regex=True)
    is_eth = t.str.contains(r"\beth\b|\bethereum\b", regex=True)
    is_sol = t.str.contains(r"\bsol\b|\bsolana\b", regex=True)
    return pd.Series(
        np.where(is_btc, "BTC", np.where(is_eth, "ETH", np.where(is_sol, "SOL", "OTHER"))),
        index=t.index
    )


# =========================
# Analytics
# =========================
def sentiment_distribution_by_label(dff: pd.DataFrame) -> pd.DataFrame:
    cut = pd.cut(dff["compound"], bins=[-1, -0.05, 0.05, 1],
                 labels=["neg", "neu", "pos"], include_lowest=True)
    tmp = dff.assign(bin=cut)
    return (tmp.groupby(["label", "bin"]).size()
            .unstack("bin").fillna(0).astype(int)
            .reindex(LABEL_ORDER, fill_value=0))


def top_sources(dff: pd.DataFrame, k: int = 5) -> Tuple[pd.Series, pd.Series]:
    agg = dff.groupby("source")["compound"].mean().dropna()
    pos = agg.sort_values(ascending=False).head(k)
    neg = agg.sort_values(ascending=True).head(k)
    return pos, neg


def extreme_headlines(dff: pd.DataFrame, k: int = 3):
    cols = ["source", "label", "compound", "title", "url"]
    best = dff.sort_values("compound", ascending=False).head(k)[cols]
    worst = dff.sort_values("compound", ascending=True).head(k)[cols]
    return best, worst


def assess_data_quality(dff: pd.DataFrame) -> dict:
    if dff.empty:
        return {"n": 0, "unique_dates": 0, "unique_sources": 0,
                "span_hours": 0.0, "has_corr": False, "reason_corr": "No data"}
    ts = pd.to_datetime(dff["publishedat"]).dt.tz_convert(None)
    unique_dates = ts.dt.date.nunique()
    span_hours = (ts.max() - ts.min()).total_seconds() / 3600.0
    unique_sources = dff["source"].nunique()
    has_corr = unique_dates >= 3 and unique_sources >= 2
    reason = ""
    if not has_corr:
        parts = []
        if unique_dates < 3: parts.append(f"need ≥3 distinct days (have {unique_dates})")
        if unique_sources < 2: parts.append(f"need ≥2 sources (have {unique_sources})")
        reason = "; ".join(parts)
    return {"n": len(dff), "unique_dates": int(unique_dates),
            "unique_sources": int(unique_sources),
            "span_hours": float(round(span_hours, 2)),
            "has_corr": bool(has_corr), "reason_corr": reason}


# =========================
# Domain analysis (Crypto)
# =========================
def derive_crypto_takeaways(dff: pd.DataFrame) -> dict:
    if dff.empty:
        return {
            "insight_text": "No data available — cannot derive insights.",
            "analysis_md": "- Dataset is empty.",
            "recommendations_md": "- Ingest more sources and expand the date window.",
            "conclusion_md": "Insufficient evidence to form a view on crypto sentiment."
        }

    n = len(dff)
    avg = float(dff["compound"].mean())
    median = float(dff["compound"].median())

    agg_lbl = dff.groupby("label")["compound"].mean().reindex(LABEL_ORDER).dropna()
    top_label = agg_lbl.idxmax()
    low_label = agg_lbl.idxmin()

    tmp = dff.copy()
    tmp["date"] = pd.to_datetime(tmp["publishedat"]).dt.tz_convert(None).dt.date
    g = tmp.groupby(["date", "label"])["compound"].mean().reset_index()

    trends = {}
    MOMO_STRONG = 0.30
    for lab in LABEL_ORDER:
        gi = g[g["label"] == lab].sort_values("date")
        if len(gi) >= 2:
            delta = float(gi["compound"].iloc[-1] - gi["compound"].iloc[0])
            trends[lab] = delta
        else:
            trends[lab] = 0.0

    src_mean_sorted = dff.groupby("source")["compound"].mean().sort_values()
    worst_src = (src_mean_sorted.index[0], float(src_mean_sorted.iloc[0]))
    best_src  = (src_mean_sorted.index[-1], float(src_mean_sorted.iloc[-1]))

    insight_text = (
        f"Overall mean {avg:.3f} (median {median:.3f}); "
        f"{top_label} leads ({agg_lbl[top_label]:.3f}) while {low_label} lags ({agg_lbl[low_label]:.3f}). "
        f"Most positive source: {best_src[0]} ({best_src[1]:.3f}); most negative: {worst_src[0]} ({worst_src[1]:.3f})."
    )

    bullets = []
    for lab in LABEL_ORDER:
        dlt = trends.get(lab, 0.0)
        if abs(dlt) >= MOMO_STRONG:
            tag = "strong"
        elif abs(dlt) >= 0.05:
            tag = "mild"
        else:
            tag = "flat"
        arrows = "↑" if dlt > 0 else ("↓" if dlt < 0 else "→")
        bullets.append(f"- **{lab}** tone: **{arrows} {tag}** (Δ ≈ {dlt:+.2f}); mean ≈ {agg_lbl.get(lab, np.nan):.3f}.")

    bullets.append(f"- Label ranking (mean): **{top_label}** highest ({agg_lbl[top_label]:.3f}); "
                   f"**{low_label}** lowest ({agg_lbl[low_label]:.3f}).")

    bullets.append(f"- Sources: most positive **{best_src[0]} ({best_src[1]:.3f})** vs most negative "
                   f"**{worst_src[0]} ({worst_src[1]:.3f})** — be mindful of outlet bias when scanning headlines.")

    recs = []
    if trends.get(top_label, 0) > 0.05:
        recs.append(f"Monitor narratives around **{top_label}** (improving tone).")
    if trends.get(low_label, 0) < -0.05:
        recs.append(f"Be cautious of negative skew on **{low_label}**; avoid over-reacting to extreme headlines.")
    recs.append("Diversify news sources to reduce editorial bias in your watchlist.")
    recs.append("Use alerting on major catalysts (ETF, L2 launches, protocol upgrades) that can shift sentiment quickly.")
    recs.append("Treat sentiment as context, not a trading signal; corroborate with price/volume & on-chain flows.")

    conclusion = (
        f"In this window (n={n}), sentiment is {('constructive' if avg>0.05 else ('negative' if avg<-0.05 else 'mixed'))}. "
        f"{top_label} carries the most optimistic tone while {low_label} faces the heaviest skepticism. "
        f"Given source dispersion ({best_src[1]:.2f} vs {worst_src[1]:.2f}), readers should apply source-weighting and "
        f"validate narratives against market data."
    )

    return {
        "insight_text": insight_text,
        "analysis_md": "\n".join(bullets),
        "recommendations_md": "\n".join([f"- {x}" for x in recs]),
        "conclusion_md": conclusion
    }


# =========================
# Matplotlib charts (for PDF)
# =========================
def _mpl_base(figsize=(7.2, 4.2)):
    """Gray background + high DPI for print clarity."""
    if not MPL_OK:
        raise RuntimeError("Matplotlib not available.")
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    bg = "#f5f5f5"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.grid(True, color="#d1d5db", linewidth=0.9, alpha=1.0)
    ax.tick_params(colors="#111")
    for s in ax.spines.values():
        s.set_edgecolor("#9ca3af"); s.set_linewidth(0.9)
    return fig, ax


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    plt.close(fig)


def render_print_charts(df_filtered: pd.DataFrame, out_dir: Optional[str] = None) -> Dict[str, str]:
    """Build print-safe charts (PNG) for the PDF using Matplotlib."""
    if not MPL_OK:
        return {}
    out: Dict[str, str] = {}
    out_dir = out_dir or CHART_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 1) By Source
    try:
        agg = (df_filtered.groupby("source", dropna=False)["compound"].mean()
               .reset_index().sort_values("compound"))
        fig, ax = _mpl_base(figsize=(7.6, 4.2))
        bars = ax.barh(agg["source"], agg["compound"],
                       color="#60a5fa", edgecolor="#0f172a", linewidth=1.0)
        ax.set_xlabel("Avg VADER compound", color="#111")
        ax.set_ylabel("Source", color="#111")
        ax.set_xlim(min(-1, agg["compound"].min() - 0.05), max(1, agg["compound"].max() + 0.05))
        for b, v in zip(bars, agg["compound"]):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height()/2,
                    f"{v:.3f}", va="center", ha="left", color="#111", fontsize=10)
        path = os.path.join(out_dir, "print_sentiment_by_source.png")
        _save_fig(fig, path); out["by_source"] = path
    except Exception:
        pass

    # 2) By Label
    try:
        agg = (df_filtered.groupby("label", dropna=False)["compound"].mean()
               .reindex(LABEL_ORDER).reset_index().rename(columns={"compound": "avg"}))
        colors = [LABEL_COLORS_BARS.get(l, "#94a3b8") for l in agg["label"]]
        fig, ax = _mpl_base(figsize=(7.0, 3.8))
        bars = ax.barh(agg["label"], agg["avg"], color=colors, edgecolor="#0f172a", linewidth=1.1)
        ax.set_xlabel("Avg VADER compound", color="#111"); ax.set_ylabel("Label", color="#111")
        ax.set_xlim(-1.0, 1.0)
        for b, v in zip(bars, agg["avg"]):
            ax.text(b.get_width() + 0.02, b.get_y() + b.get_height()/2,
                    f"{v:.3f}", va="center", ha="left", color="#111", fontsize=10)
        path = os.path.join(out_dir, "print_sentiment_by_label.png")
        _save_fig(fig, path); out["by_label"] = path
    except Exception:
        pass

    # 3) Time-series
    try:
        if not df_filtered.empty:
            tmp = df_filtered.copy()
            tmp["date"] = pd.to_datetime(tmp["publishedat"]).dt.tz_convert(None).dt.date
            g = tmp.groupby(["date", "label"], dropna=False)["compound"].mean().reset_index()
            fig, ax = _mpl_base(figsize=(7.2, 4.2))
            line_colors = {"BTC":"#22c55e", "ETH":"#111111", "SOL":"#fbbf24", "OTHER":"#3b82f6"}
            MOMO_STRONG = 0.30
            for lab in LABEL_ORDER:
                gi = g[g["label"] == lab].sort_values("date")
                if gi.empty:
                    continue
                delta = float(gi["compound"].iloc[-1] - gi["compound"].iloc[0]) if len(gi) >= 2 else 0.0
                lw = 3.8 if abs(delta) < MOMO_STRONG else 5.2
                ls = "--" if lab == "ETH" else "-"
                ax.plot(
                    gi["date"], gi["compound"],
                    color=line_colors.get(lab, "#94a3b8"),
                    linewidth=lw, linestyle=ls, marker="o",
                    markeredgecolor="#0f172a", markeredgewidth=0.9, markersize=5.8,
                    label=lab
                )
                x_last = gi["date"].iloc[-1]; y_last = float(gi["compound"].iloc[-1])
                ax.annotate(
                    f"{lab} {y_last:.3f}", xy=(x_last, y_last), xytext=(6, 6),
                    textcoords="offset points", fontsize=10, color="#111",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#0f172a", lw=0.6)
                )
                if abs(delta) >= MOMO_STRONG:
                    idx = int(max(0, len(gi)*0.6 - 1))
                    ax.annotate(
                        "STRONG",
                        xy=(gi["date"].iloc[idx], float(gi["compound"].iloc[idx])),
                        xytext=(0, -14), textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fff7ed", ec="#fb923c", lw=0.8), color="#111"
                    )
            ax.set_xlabel("Date", color="#111"); ax.set_ylabel("Avg VADER compound", color="#111")
            ax.legend(frameon=False, fontsize=10); ax.set_ylim(-1.05, 1.05)
            path = os.path.join(out_dir, "print_timeseries_by_label.png")
            _save_fig(fig, path); out["ts_label"] = path
    except Exception:
        pass

    # 4) Heatmap
    try:
        tmp = df_filtered.copy()
        tmp["date"] = pd.to_datetime(tmp["publishedat"]).dt.tz_convert(None).dt.date
        daily = tmp.groupby(["date", "source"], dropna=False)["compound"].mean().reset_index()
        pivot = daily.pivot(index="date", columns="source", values="compound")
        fig, ax = _mpl_base(figsize=(7.2, 4.2))
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
            corr = pivot.corr(min_periods=3)
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9, color="#111")
            ax.set_yticklabels(corr.index, fontsize=9, color="#111")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=9, colors="#111")
        else:
            ax.imshow(np.ones((1, 1)), cmap="RdBu_r", vmin=-1, vmax=1)
            msg = "Not enough data for correlation\n(need ≥3 distinct dates and ≥2 sources)"
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#111",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#0f172a", lw=0.6))
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Source", color="#111"); ax.set_ylabel("Source", color="#111")
        path = os.path.join(out_dir, "print_source_corr_heatmap.png")
        _save_fig(fig, path); out["heatmap"] = path
    except Exception:
        pass

    return out


# =========================
# PDF helpers
# =========================
def _to_rgb_buffer(path: str):
    """Open PNG, force RGB (strip alpha), return BytesIO + size."""
    if not path or not os.path.exists(path) or Image is None:
        return None, None
    im = Image.open(path)
    w_px, h_px = im.size
    if im.mode != "RGB":
        im = im.convert("RGB")
    bio = io.BytesIO()
    im.save(bio, format="PNG", optimize=True)
    bio.seek(0)
    return bio, (w_px, h_px)

def _image_fit_block(path: str, max_w_cm: float = 17.4, max_h_cm: float = 11.2,
                     caption: Optional[str] = None, styles=None):
    """Fit image into a wide box (keep aspect)."""
    if not path or not os.path.exists(path):
        return []
    buf, size = _to_rgb_buffer(path) if REPORTLAB_OK else (None, None)
    if buf is None and Image is not None:
        img = Image.open(path); size = img.size
    if size is None:
        return []
    w_px, h_px = size
    max_w_pt = max_w_cm * cm; max_h_pt = max_h_cm * cm
    ratio = min(max_w_pt / w_px, max_h_pt / h_px)
    disp_w, disp_h = w_px * ratio, h_px * ratio

    elements = []
    rl_img = RLImage(buf if buf is not None else path, width=disp_w, height=disp_h)
    rl_img.hAlign = "CENTER"
    elements.append(rl_img)
    if caption and styles:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<font color='black'><i>{caption}</i></font>", styles["Small"]))
    elements.append(Spacer(1, 18))
    return [KeepTogether(elements)]


def _chart_block(title: str, path: str, styles,
                 max_w_cm: float = 17.4, max_h_cm: float = 11.2,
                 caption: Optional[str] = None):
    block = [Paragraph(f"<b>{title}</b>", styles["H2"]), Spacer(1, 2)]
    block += _image_fit_block(path, max_w_cm, max_h_cm, caption, styles)
    return block


def _add_page_numbers(canv: canvas.Canvas, doc):
    canv.setFont("Helvetica", 9)
    canv.drawRightString(doc.pagesize[0] - 1.8*cm, 1.1*cm, f"Page {canv.getPageNumber()}")


def _md_bullets_to_listflowable(md_text: str, styles):
    lines = [ln.strip() for ln in (md_text or "").splitlines()]
    items = []
    for ln in lines:
        if ln.startswith("- "):
            items.append(ListItem(Paragraph(ln[2:].strip(), styles["Small"]), leftIndent=12))
    if not items:
        return []
    return [KeepTogether(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=12))]


def _chart_explanations_md() -> str:
    return """
- **Sentiment Score by News Source** — Average sentiment per outlet; further right = more positive tone.
- **Sentiment by Label** — Average sentiment per crypto label (BTC/ETH/SOL/OTHER).
- **Time-series by Label** — Daily average sentiment by label. Up = improving tone; down = weakening tone.
- **Source Correlation Heatmap** — Similarity of outlets' sentiment moves (red = positively correlated, blue = negatively).
- **Summary (Research-ready)** — Mean/median, dispersion, and positive/neutral/negative counts.
- **Per-label (mean compound)** — > 0.05 ≈ positive; < −0.05 ≈ negative; in between = neutral.
""".strip()


# =========================
# PDF builder
# =========================
def build_pdf_report(df_filtered: pd.DataFrame,
                     per_label_tbl: pd.DataFrame,
                     png_paths: Dict[str, str],
                     meta: dict) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed. `pip install reportlab Pillow`")

    # Backfill PNGs with print versions (Matplotlib) if any Plotly PNG is missing
    need = {k: True for k in ["by_source", "by_label", "ts_label", "heatmap"]
            if not png_paths.get(k) or not os.path.exists(png_paths.get(k, ""))}
    if need and MPL_OK:
        mp = render_print_charts(df_filtered, out_dir=CHART_DIR)
        png_paths = {
            "by_source": mp.get("by_source", png_paths.get("by_source", "")),
            "by_label":  mp.get("by_label",  png_paths.get("by_label", "")),
            "ts_label":  mp.get("ts_label",  png_paths.get("ts_label", "")),
            "heatmap":   mp.get("heatmap",   png_paths.get("heatmap", "")),
        }

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.6*cm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6))

    s = []
    s.append(Paragraph("Crypto Sentiment Report", styles["H1"]))
    s.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    s.append(Paragraph(f"Data file: {meta.get('data_file','-')}", styles["Small"]))
    s.append(Paragraph(f"Rows: {meta.get('n','-')} | Date range: {meta.get('date_range','-')}", styles["Small"]))
    s.append(Spacer(1, 8))

    s.append(Paragraph("<b>How to read this</b>", styles["H2"]))
    s.append(Paragraph(
        "Headlines are scored with VADER compound (−1 = very negative, +1 = very positive). "
        "Thresholds: positive > 0.05, neutral −0.05..0.05, negative < −0.05. "
        "Bars further right indicate more positive tone. Label colors: BTC green, ETH dashed black in lines (for visibility), "
        "SOL yellow, OTHER blue.", styles["Small"]
    ))
    s.append(Spacer(1, 8))

    plain = meta.get("insight_text") or "—"
    try:
        if per_label_tbl is not None and not per_label_tbl.empty:
            top_row = per_label_tbl.sort_values("avg_compound", ascending=False).iloc[0]
            low_row = per_label_tbl.sort_values("avg_compound").iloc[0]
            plain += f" Highest label mean: {top_row['label']} ({float(top_row['avg_compound']):.3f}); " \
                     f"Lowest: {low_row['label']} ({float(low_row['avg_compound']):.3f})."
    except Exception:
        pass
    s.append(Paragraph(plain, styles["Small"]))
    s.append(Spacer(1, 8))

    k = [
        ["Articles (n)", str(meta.get("n", "-"))],
        ["Avg / Median", f"{meta.get('avg', float('nan')):.3f} / {meta.get('median', float('nan')):.3f}"],
        ["Std dev", f"{meta.get('std', float('nan')):.3f}"],
        ["Positive (>0.05)", f"{meta.get('pos_n','-')} ({meta.get('pos_pct',0):.1f}%)"],
        ["Neutral (−0.05..0.05)", f"{meta.get('neu_n','-')} ({meta.get('neu_pct',0):.1f}%)"],
        ["Negative (<−0.05)", f"{meta.get('neg_n','-')} ({meta.get('neg_pct',0):.1f}%)"],
    ]
    t = Table(k, hAlign="LEFT", colWidths=[7.0*cm, 8.0*cm])
    t.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25, rlcolors.grey),
        ("BACKGROUND",(0,0),(-1,0), rlcolors.whitesmoke),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,0),(1,-1),"RIGHT"),
    ]))
    s.append(t)
    s.append(Spacer(1, 8))

    s.append(Paragraph("<b>Per Label (mean compound)</b>", styles["H2"]))
    s.append(Paragraph(
        "Average sentiment per crypto label. Reading guide: "
        "<b>> 0.05 = Positive</b>, <b>< −0.05 = Negative</b>, in between = Neutral. "
        "Higher means a more optimistic tone for that label.", styles["Small"]
    ))
    lab_tbl = [["Label", "Avg compound"]]
    if per_label_tbl is not None and not per_label_tbl.empty:
        for _, r in per_label_tbl.sort_values("label").iterrows():
            lab_tbl.append([r["label"], f"{float(r['avg_compound']):.3f}"])
    t2 = Table(lab_tbl, hAlign="LEFT", colWidths=[7.0*cm, 8.0*cm])
    t2.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25, rlcolors.grey),
        ("BACKGROUND",(0,0),(-1,0), rlcolors.whitesmoke),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(1,-1),"RIGHT"),
    ]))
    s.append(t2)
    s.append(Spacer(1, 8))

    analysis_md = (meta.get("analysis_md") or "").strip()
    if analysis_md:
        s.append(Paragraph("<b>Crypto analysis</b>", styles["H2"]))
        s += _md_bullets_to_listflowable(analysis_md, styles)
        s.append(Spacer(1, 6))

    rec_md = (meta.get("recommendations_md") or "").strip()
    if rec_md:
        s.append(Paragraph("<b>Recommendations</b>", styles["H2"]))
        s += _md_bullets_to_listflowable(rec_md, styles)
        s.append(Spacer(1, 6))

    s.append(Paragraph("<b>What each chart shows</b>", styles["H2"]))
    s += _md_bullets_to_listflowable(_chart_explanations_md(), styles)

    s.append(PageBreak())

    chart_specs = [
        ("Sentiment by Source",  "by_source",
         "Average VADER per source (further right → more positive)."),
        ("Sentiment by Label",   "by_label",
         "Average VADER per label (BTC green, ETH dark gray in bars, SOL yellow, OTHER blue)."),
        ("Time-series by Label", "ts_label",
         "Daily mean by label; ETH shown as dashed black. Strong momentum is marked when |Δ| ≥ 0.30."),
        ("Source Correlation Heatmap", "heatmap",
         "Pearson correlation (daily means). Red = positive, blue = negative."),
    ]
    charts_added = 0
    for i, (title, key, cap) in enumerate(chart_specs):
        p = png_paths.get(key)
        if not (p and os.path.exists(p)):
            continue
        s += _chart_block(title, p, styles, max_w_cm=17.4, max_h_cm=11.2, caption=cap)
        charts_added += 1
        if i < len(chart_specs) - 1:
            s.append(PageBreak())

    concl = (meta.get("conclusion_md") or "").strip()
    if concl:
        if charts_added:
            s.append(PageBreak())
        s.append(Paragraph("<b>Conclusion</b>", styles["H2"]))
        s.append(Paragraph(concl, styles["Small"]))

    doc.build(s, onFirstPage=_add_page_numbers, onLaterPages=_add_page_numbers)
    return buf.getvalue()
