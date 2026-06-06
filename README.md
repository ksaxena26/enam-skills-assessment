# Wisdom Investor — Intelligent Portfolio Management System

A quantitative trading engine with multi-timeframe feature engineering, risk-based position sizing, Optuna-driven parameter optimisation, and an interactive Streamlit dashboard.

---

## Folder Navigation

```
enam-skills-assessment/
├── leadership_note/            # Architecture notes for stakeholders
│   └── architecture_note.pdf  # Foundational architecture decisions (PDF)
│
├── chats/                      # Session history
│   └── chat_history.pdf        # Full development chat log (PDF)
│
├── data/                       # Market data
│   ├── ohlcv/                  # Raw OHLCV CSVs (one file per ticker)
│   └── features/               # Engineered feature tables
│       ├── features_d.csv      # Daily features  (primary dashboard input)
│       ├── features_w.csv      # Weekly features
│       └── features_m.csv      # Monthly features
│
├── trading_engine/             # Core engine (strategy, backtest, analytics)
├── src/                        # Supporting modules (engineering, data manager)
├── dashboard/                  # Streamlit dashboard (Wisdom Trader)
│   ├── app.py                  # Main application
│   ├── logo.svg                # Brand logo
│   └── requirements.txt        # Dashboard dependencies
│
├── context/                    # Reference documents and briefing notes
├── outputs/                    # Backtest output artefacts (trade logs, charts)
├── research/                   # Exploratory notebooks and scripts
└── main.py                     # End-to-end pipeline entry point
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r dashboard/requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard auto-loads `data/features/features_d.csv` on startup. A password gate protects access (configured in `dashboard/app.py`).

### 3. Run the full pipeline

```bash
python main.py
```

Fetches OHLCV data, runs feature engineering across daily / weekly / monthly timeframes, and writes feature CSVs to `data/features/`.

---

## Key Components

| Component | Location | Purpose |
|---|---|---|
| Feature engineering | `src/engineering.py` | RSI, ATR, Supertrend, CRSI, multi-timeframe aggregation |
| Backtest engine | `trading_engine/` | Event-driven, mark-to-market portfolio simulation |
| Parameter optimisation | `optuna_tuning.db` | Optuna TPE, 1000+ trials — best Trial #348 at 16.72% CAGR |
| Dashboard | `dashboard/app.py` | Daily signals, backtest runner, parameter tuning |
| Architecture note | `leadership_note/architecture_note.pdf` | Technical deep-dive for investment management leadership |

---

## Architecture Overview

See [`leadership_note/architecture_note.pdf`](leadership_note/architecture_note.pdf) for a detailed write-up covering:

- Data pipeline and multi-timeframe feature engineering
- Trading engine design and risk-based position sizing
- Optuna optimisation framework and best trial results
- Dashboard design and cloud deployment
- Data security, LLM usage, and AI agent integration

---

## Deployment

The dashboard can be hosted on:

- **Streamlit Community Cloud** — connect the GitHub repo and deploy in minutes (recommended for POC)
- **AWS EC2** — use `dashboard/ec2_setup.sh` for a tmux-managed server process
- **ngrok** — tunnel a local Streamlit instance for quick sharing

---

## Security Notes

- Secrets (broker credentials, Key Vault access) are never hardcoded; all runtime secrets are retrieved from Azure Key Vault.
- No proprietary OHLCV data or client trade history is included in this repository.
- Configuration parameters live in `config/config.py`, not in `.env` files.
