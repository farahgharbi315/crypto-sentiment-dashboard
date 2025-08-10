# Crypto Sentiment Dashboard

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
[![Made by wiqilee](https://img.shields.io/badge/made%20by-wiqilee-000000.svg?logo=github)](https://github.com/)

A Streamlit dashboard for analyzing crypto news sentiment (BTC / ETH / SOL / OTHER) using VADER.  
It includes interactive charts (Plotly), export-ready PNGs (RGB-safe), and a print-quality PDF report (ReportLab).

> **Disclaimer:** This project is for research and monitoring purposes only. It is **not** financial advice.

---

## ✨ Features

- **Interactive charts** — Source & label bar charts, daily time-series, and correlation heatmap.
- **Crypto domain insights** — Auto-generated summary, bullet-point analysis, recommendations, and conclusion.
- **Export-ready images** — Large-canvas PNGs with generous margins, **forced RGB** to avoid disappearing lines in PDFs.
- **One-click PDF** — A neatly paginated report with all charts and summary sections.
- **Powerful filters** — Label, source, and UTC date range.
- **Pandas compatibility** — Hides table index with a safe fallback for older pandas versions.

---

## 🚀 Quick start

```bash
# 1) (optional) create a virtual environment
python -m venv .venv
# activate it:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run the app
streamlit run streamlit_app.py
```

The app reads data from `data/news_raw.csv`.

---

## 📦 Requirements

- Python **3.10+** (recommended)
- `streamlit`, `pandas`, `numpy`
- `plotly`, `matplotlib`
- `kaleido` (PNG export backend for Plotly)
- `reportlab`, `Pillow` (PDF builder and image backend)

All pinned in `requirements.txt`.

---

## 🗂️ Data schema

The loader (`sentiment.py`) normalizes your columns automatically. Minimum required:

- `publishedAt` / `published` / `date` → normalized to `publishedat` (UTC timezone)
- `source` — outlet name/domain
- `compound` — VADER score in `[-1, +1]`

Optional:
- `channel`, `title`, `description`, `url`, `label` (if `label` is missing, it is inferred from text: **BTC/ETH/SOL**, else **OTHER**)

**Example row:**

```csv
publishedAt,source,compound,title,url,label
2025-08-01T10:30:00Z,CoinDesk,0.21,"BTC breaks range","https://example.com/article",BTC
```

> Tip: If your dataset is large or sensitive, keep it out of your repo and list `data/*.csv` in `.gitignore`.

---

## 📊 Charts & PDF exports

When you click **Build PDF report** in the UI, the app:

1. Saves high-resolution PNGs for each chart with large margins so labels never get cropped.
2. **Forces RGB** (removes alpha) on the PNGs to prevent thin lines from disappearing in PDFs.
3. Assembles a polished, paginated PDF using ReportLab — one chart per page, plus summaries.

**About the time-series edges**  
The PDF version uses a small time padding by default to avoid edge clipping. You can switch to fully edge-to-edge by editing the `timeseries_by_label(..., theme_for_pdf=True)` x-axis range in `streamlit_app.py`.

---

## 🧱 Project structure

```
.
├─ streamlit_app.py        # Streamlit UI (Plotly + PDF trigger)
├─ sentiment.py            # Data loader, domain analysis, print-safe charts (matplotlib), PDF builder
├─ requirements.txt
├─ data/
│  └─ news_raw.csv         # Your dataset (kept local by default)
├─ charts/                 # Generated PNGs (auto-created)
└─ LICENSE                 # MIT © 2025–present wiqilee
```

---

## ⚙️ Configuration

Adjust label colors/order in both modules if desired:

```python
LABEL_COLORS = {"BTC": "#22c55e", "ETH": "#ffffff", "SOL": "#fde047", "OTHER": "#94a3b8"}
LABEL_ORDER  = ["BTC", "ETH", "SOL", "OTHER"]
```

---

## 🛠️ Troubleshooting

- **`kaleido` export error**  
  Ensure `kaleido==0.2.1` is installed:
  ```bash
  pip install --upgrade kaleido
  ```

- **Time-series lines missing in PDFs**  
  Already handled by forcing RGB on exported PNGs. Make sure `Pillow` is installed.

- **Table index won’t hide (old pandas)**  
  The app falls back from `styler.hide_index()` to `styler.hide(axis="index")` automatically.

---

## 🗺️ Roadmap

- Automated news ingestion
- Alternative sentiment models beyond VADER
- One-click deploy to Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.  
Make sure your PR includes a clear description and, when possible, screenshots of UI changes.

---

## 📜 License

**MIT** © 2025–present **wiqilee** — see [`LICENSE`](./LICENSE) for details.
