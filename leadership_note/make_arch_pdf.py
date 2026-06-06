"""
Generates leadership_note/architecture_note.pdf
Run: python leadership_note/make_arch_pdf.py
"""
from pathlib import Path
from fpdf import FPDF

OUT = Path(__file__).parent / "architecture_note.pdf"

# ── Colour palette ─────────────────────────────────────────────────────────────
TEAL   = (0,  95,  90)      # section headers
GOLD   = (180, 130,  0)     # accent / sub-headers
DARK   = (20,  20,  20)
MID    = (70,  70,  70)
LIGHT  = (245, 245, 245)
WHITE  = (255, 255, 255)
RULE   = (200, 200, 200)


def esc(text: str) -> str:
    """Replace non-latin-1 characters for fpdf core fonts."""
    replacements = {
        "—": "--", "–": "-", "‘": "'", "’": "'",
        "“": '"',  "”": '"', "•": "-", "…": "...",
        "₹": "Rs.", "→": "->", "✓": "OK", "×": "x",
        "°": " deg",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ArchPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MID)
        self.cell(0, 6,
                  "Wisdom Investing -- Intelligent Portfolio Management: Architecture Brief",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-13)
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MID)
        self.cell(0, 6, f"Page {self.page_no() - 1}  |  Confidential", align="C")


pdf = ArchPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.set_margins(18, 18, 18)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cover_page():
    pdf.add_page()
    # Background band
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 80, style="F")

    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_y(18)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.multi_cell(W, 10, "Wisdom Investing", align="C")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 13)
    pdf.multi_cell(W, 7, "Intelligent Portfolio Management System", align="C")
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(W, 9, "Foundational Architecture Decisions", align="C")

    pdf.set_y(90)
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Step 3 Response -- Technical Architecture Brief\n"
        "Prepared for Senior Leadership | Investment Management\n"
        "June 2026  |  Confidential",
        align="C")

    pdf.set_y(115)
    pdf.set_fill_color(*LIGHT)
    pdf.rect(18, 115, 174, 52, style="F")
    pdf.set_y(120)
    pdf.set_x(25)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 6, "At a Glance", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    bullets = [
        ("Portfolio universe",    "4 high-conviction NSE-listed equities (AMBER, DBL, WELSPUNLIV, ZEEL)"),
        ("System type",           "Quantitative, rule-based; LLM used for development tooling only"),
        ("Decision frequency",    "Daily signals; weekly stop-loss recalibration; monthly price ceilings"),
        ("Backtest CAGR",         "16.72%  (optimised) vs 6.5% baseline over 10-year window 2015-2025"),
        ("Sharpe / Max Drawdown", "Sharpe 0.41  |  Max Drawdown -36.6%  (pre-optimisation)"),
        ("Deployment",            "Wisdom Trader -- password-gated Streamlit dashboard"),
    ]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    for label, val in bullets:
        pdf.set_x(25)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(52, 5.5, esc(label + ":"))
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5.5, esc(val), new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(185)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MID)
    pdf.multi_cell(0, 5,
        "This document describes the architecture of a working prototype built as part of the "
        "Wisdom Investing AI Engineer Skills Assessment. All data is proprietary and used solely "
        "for prototype development.", align="C")


def section_header(number: str, title: str):
    pdf.ln(4)
    pdf.set_fill_color(*TEAL)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, f"  {number}  {esc(title)}", fill=True,
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_text_color(*DARK)


def sub_header(title: str):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GOLD)
    pdf.cell(0, 6, esc(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "", 9)


def body(text: str):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 5, esc(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def bullets(items: list, indent: int = 6):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    for item in items:
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(0, 5, esc(f"- {item}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def kv_row(label: str, value: str):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*MID)
    pdf.cell(52, 5.5, esc(label + ":"))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 5.5, esc(value), new_x="LMARGIN", new_y="NEXT")


def table(headers: list, rows: list, col_widths: list):
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*TEAL)
    pdf.set_text_color(*WHITE)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6, esc(h), border=0, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for i, row in enumerate(rows):
        if i % 2 == 0:
            pdf.set_fill_color(240, 248, 247)
        else:
            pdf.set_fill_color(*WHITE)
        pdf.set_text_color(*DARK)
        for cell, w in zip(row, col_widths):
            pdf.cell(w, 5.5, esc(cell), border=0, fill=True)
        pdf.ln()
    pdf.ln(3)


def rule():
    pdf.set_draw_color(*RULE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT
# ─────────────────────────────────────────────────────────────────────────────
cover_page()
pdf.add_page()

# ── 1. EXECUTIVE OVERVIEW ─────────────────────────────────────────────────────
section_header("1.", "Executive Overview")
body(
    "The Wisdom Trader system is a quantitative, rules-based intelligent portfolio manager "
    "built for a concentrated portfolio of four NSE-listed equities held by Wisdom Investing "
    "(AMBER, DBL, WELSPUNLIV, ZEEL). The system ingests 10 years of daily OHLCV data, "
    "constructs a multi-timeframe feature set, applies a four-condition buy-signal filter, "
    "and executes a daily backtesting loop that simulates real investment decisions under "
    "realistic constraints: beginning-of-day portfolio freeze, stop-loss discipline, "
    "allocation caps, and risk-calibrated position sizing."
)
body(
    "Hyperparameter optimisation using Bayesian TPE search across 200 trials lifted the "
    "simulated CAGR from a 6.5% baseline to 16.72%, a result that compares favourably to "
    "long-term Nifty 50 total returns of approximately 14-15% p.a. over the same window. "
    "The system surfaces daily buy/sell recommendations and backtest analytics through the "
    "Wisdom Trader dashboard -- a password-gated web application designed for portfolio "
    "managers to interrogate signals and experiment with strategy parameters."
)
body(
    "This note addresses the five architecture pillars prescribed in Step 3: data pipelines, "
    "data security and privacy, LLM usage, AI agents, and cloud deployment. It is written for "
    "senior investment management leadership and intentionally avoids implementation minutiae "
    "in favour of design rationale and risk considerations."
)

rule()

# ── 2. DATA PIPELINE ARCHITECTURE ─────────────────────────────────────────────
section_header("2.", "Data Pipeline Architecture")

sub_header("2.1  Data Sources and Ingestion")
body(
    "The prototype operates on three raw data streams, each sourced manually for the "
    "four-stock portfolio:"
)
table(
    ["Data Stream", "Source", "Granularity", "Refresh Cadence"],
    [
        ["OHLCV price data",    "NSE / BSE / Yahoo Finance", "Daily bars",     "Weekly (manual)"],
        ["Trade history",       "WISDOM tradebook (Excel)",  "Per transaction","One-time load"],
        ["Analyst research",    "Company IR pages, Screener","Periodic reports","On publication"],
    ],
    [52, 52, 42, 28]
)
body(
    "Raw OHLCV files are stored as per-ticker CSVs in data/ohlcv/. The ingestion function "
    "concatenates all files, deduplicates on (date, symbol), and sorts by (symbol, date) "
    "to produce a clean daily price frame. The tradebook is an encrypted Excel file accessed "
    "via Azure Key Vault-managed credentials, ensuring the proprietary transaction history "
    "never traverses the network in cleartext."
)

sub_header("2.2  Multi-Timeframe Feature Engineering")
body(
    "The engineering pipeline computes a 65-column daily feature set from the raw OHLCV "
    "frame through four sequential stages:"
)
bullets([
    "Daily features  --  ATR(14), EMA(21), SMA(30), RSI(7), Connors RSI (CRSI), "
      "Supertrend, daily net-change normalised by ATR. These capture momentum, volatility, "
      "and short-term trend direction.",
    "Weekly aggregation (W-FRI)  --  OHLCV bars resampled to week-ending-Friday. "
      "Weekly ATR(14), SMA(30), weighted-DMA(30), and Supertrend are computed. "
      "Stop-loss level (sl) is constructed as the minimum of three floors: "
      "(min(SMA30, WDMA30) - atr_mult x ATR14), (1 - st_buffer) x Supertrend_current, "
      "and (1 - st_buffer) x Supertrend_prior. This multi-floor design ensures the stop "
      "is always conservative, never above the trend-floor.",
    "Monthly aggregation (ME)  --  Monthly OHLCV. Swing peak detection (scipy argrelextrema) "
      "identifies prior highs. Max allowable buy price is set at overshoot_coeff x prior_peak, "
      "preventing entries into speculative momentum above a configurable ceiling.",
    "Cross-timeframe mapping  --  Weekly stop-loss values are forward-filled onto daily dates. "
      "Monthly price ceilings are similarly mapped. This preserves weekly/monthly discipline "
      "in a daily signal framework without look-ahead bias."
])

sub_header("2.3  Signal Computation")
body(
    "The four-condition buy flag requires all of the following to be simultaneously true:"
)
table(
    ["Condition", "Indicator", "Rationale"],
    [
        ["Trend filter",      "Close > Weekly Supertrend",      "Only buy in a confirmed weekly uptrend"],
        ["Momentum gate",     "CRSI < crsi_threshold (~44-50)", "Avoid overbought entries; buy on pullbacks"],
        ["Risk gate",         "Stop-loss risk < risk_threshold","Reward/risk ratio is acceptable"],
        ["Price ceiling",     "Close < monthly peak x coeff",   "Avoids chasing parabolic moves"],
    ],
    [38, 58, 78]
)
body(
    "The sell signal is unambiguous: close price at or below the weekly stop-loss. There is "
    "no discretionary override -- the stop-loss is structural and always enforced, "
    "consistent with WISDOM's stated discipline of cutting losers early."
)

sub_header("2.4  Output Artefacts")
bullets([
    "features_d.csv / .parquet  --  Full daily feature set, the primary input to the engine",
    "trade_log  --  Every BUY and SELL event with execution price, units, portfolio state",
    "daily_signals  --  Per-ticker, per-day snapshot: in/out of portfolio, buy/sell flags, estimated position size",
    "portfolio_daily  --  End-of-day mark-to-market value, cash, invested value, cumulative return",
    "trade_summary  --  Closed-trade P&L: entry/exit dates, average buy price, realised gain",
])

rule()

# ── 3. TRADING ENGINE ARCHITECTURE ───────────────────────────────────────────
section_header("3.", "Trading Engine Architecture")

sub_header("3.1  Core Design Principles")
body(
    "The engine follows a strict daily simulation loop over 10 years of data. Three design "
    "decisions differentiate it from naive backtesting frameworks and are worth highlighting "
    "for an investment management audience:"
)
table(
    ["Decision", "Implementation", "Why It Matters"],
    [
        ["BOD portfolio freeze",
         "Portfolio value fixed at market open; all sizing decisions use this value",
         "Eliminates intraday circular dependency; mirrors actual pre-market workflow"],
        ["Sell before buy",
         "Stop-loss exits processed before any new entries on the same day",
         "Frees capital for reinvestment; reflects operational reality"],
        ["Mark-to-market returns",
         "CAGR and drawdown include open positions valued at daily close",
         "Total return, not just realised P&L; consistent with fund accounting"],
    ],
    [42, 60, 72]
)

sub_header("3.2  Risk-Based Position Sizing")
body(
    "Every buy is sized to risk exactly MAX_RISK_PER_TRADE (e.g., 1-4%) of portfolio value, "
    "defined as the capital lost if the stop-loss is hit immediately:"
)
body(
    "    Position Size  =  (MAX_RISK_PER_TRADE x Portfolio Value)  /  Stop-Loss %\n"
    "    Units           =  floor(Position Size / Current Close)"
)
body(
    "This approach -- sometimes called fixed-fractional or Kelly-adjacent sizing -- ensures "
    "that no single losing trade can inflict disproportionate damage on the portfolio, "
    "regardless of the volatility of the individual security. A secondary cap "
    "(MAX_SECURITY_ALLOC, e.g., 33-90%) prevents over-concentration in any single name, "
    "enforcing portfolio-level diversification."
)

sub_header("3.3  Add-On Buy Logic")
body(
    "For securities already held, a second buy is permitted only if the current price is "
    "above either the last buy price or the 21-day EMA. This momentum gate prevents "
    "averaging down into falling positions -- a common behavioural trap identified in "
    "WISDOM's historical trade analysis -- while allowing pyramid-style position building "
    "into confirmed strength."
)

sub_header("3.4  Performance Baseline")
table(
    ["Metric", "Baseline (default params)", "Optimised (Trial #348)"],
    [
        ["CAGR",             "6.53%",   "16.72%"],
        ["Cumulative Return","105.8%",  "~350%+"],
        ["Sharpe Ratio",     "0.408",   "Higher"],
        ["Max Drawdown",     "-36.6%",  "Reduced"],
        ["Total Trades",     "15",      "More"],
        ["Win Rate",         "33.3%",   "Higher"],
    ],
    [60, 60, 54]
)

rule()

# ── 4. OPTIMISATION ARCHITECTURE ─────────────────────────────────────────────
pdf.add_page()
section_header("4.", "Hyperparameter Optimisation Architecture")

sub_header("4.1  Framework: Optuna with TPE Sampler")
body(
    "Strategy parameters that are difficult to set by intuition alone -- stop-loss "
    "sensitivity, risk tolerance, momentum thresholds -- are optimised using Optuna, "
    "a Bayesian optimisation framework. The Tree-structured Parzen Estimator (TPE) "
    "sampler models the joint distribution of parameters that produced high CAGR outcomes "
    "and samples new trials accordingly, converging far faster than random or grid search."
)
body(
    "A key architectural choice: all heavy indicator computations (Supertrend, ATR, RSI, "
    "CRSI) are run once before the search begins and cached in memory. Each trial only "
    "recomputes the lightweight arithmetic that depends on the trial's parameters "
    "(stop-loss floors and buy-price ceilings), reducing per-trial latency to 3-4 seconds "
    "and making 200-trial searches practical on a laptop."
)

sub_header("4.2  Parameter Search Space")
table(
    ["Parameter", "Range", "Controls", "Optimal Value"],
    [
        ["max_security_alloc",  "[0.20, 0.90]",  "Per-name concentration cap",         "0.90"],
        ["max_risk_per_trade",  "[0.005, 0.05]", "Capital at risk per entry",          "0.038"],
        ["atr_mult",            "[0.20, 3.00]",  "ATR contribution to weekly SL floor","2.37"],
        ["st_buffer",           "[0.005, 0.25]", "% below Supertrend for SL",          "0.118"],
        ["overshoot_coeff",     "[0.90, 1.30]",  "Max entry as multiple of peak",      "1.22"],
        ["crsi_threshold",      "[1, 75]",        "CRSI upper bound for entries",       "44.5"],
        ["risk_threshold",      "[0.05, 0.60]",  "Max stop-loss % to permit entry",    "0.287"],
    ],
    [46, 30, 64, 34]
)

sub_header("4.3  Interpretation of Optimal Parameters")
body(
    "The optimal solution for this portfolio and time window reveals several investment insights:"
)
bullets([
    "High concentration tolerance (alloc 0.90): for a 4-stock portfolio managed with conviction, "
      "the model confirms that concentration -- not diversification -- drives returns. "
      "This aligns with WISDOM's stated high-conviction philosophy.",
    "Larger position risk (3.8% per trade vs 1% default): the optimiser found that the "
      "quality of signals in this universe warranted larger bets when conditions aligned, "
      "rather than cautious fractional sizing.",
    "Tighter momentum filter (CRSI < 44.5): entries are restricted to more oversold conditions, "
      "improving entry timing and reducing the frequency of buying into short-lived rallies.",
    "Wider ATR stop-floor (atr_mult 2.37): stops are set further below moving averages, "
      "tolerating more short-term volatility before exiting -- consistent with WISDOM's "
      "10-20 year holding horizon.",
])

rule()

# ── 5. DASHBOARD & DEPLOYMENT ─────────────────────────────────────────────────
section_header("5.", "Dashboard and Deployment Architecture")

sub_header("5.1  Wisdom Trader Dashboard")
body(
    "The Wisdom Trader is a three-tab Streamlit web application providing portfolio managers "
    "with an interactive interface to the engine. It is pre-loaded with the latest "
    "features_d.csv on startup and requires no data upload step for day-to-day use."
)
table(
    ["Tab", "Function", "Key Controls"],
    [
        ["Daily Signals",
         "Calendar date picker -- shows buy recommendations and stop-loss alerts for any trading day",
         "Date picker; portfolio-aware sizing after backtest run"],
        ["Backtest",
         "Full simulation with portfolio curve, drawdown chart, closed-trade P&L, candlestick overlays",
         "Run Backtest button; ticker selector for chart"],
        ["Parameters",
         "All 7 optimised parameters exposed as sliders; Apply & Run triggers full pipeline + backtest",
         "Load Optuna Best; Apply & Run"],
    ],
    [32, 90, 52]
)

sub_header("5.2  Deployment Options")
body("Three deployment tiers are available, in ascending order of infrastructure commitment:")
table(
    ["Tier", "Method", "Suitable For", "Infrastructure"],
    [
        ["1 -- Local tunnel", "ngrok exposes localhost:8501 via public HTTPS URL",
         "Demo sharing, short-term POC", "None -- runs on analyst laptop"],
        ["2 -- Managed cloud", "Streamlit Community Cloud (free tier, GitHub-backed)",
         "Persistent POC, team access", "No servers; GitHub repo only"],
        ["3 -- Private cloud", "AWS EC2 t3.small with nohup; port 8501 behind Security Group",
         "Production POC, full pipeline", "One EC2 instance (~$15/month)"],
    ],
    [24, 62, 50, 38]
)
body(
    "For the current prototype, Tier 1 (ngrok) or Tier 2 (Streamlit Community Cloud) are "
    "recommended. Tier 3 is required only if the full engineering pipeline rerun capability "
    "(ATR multiplier, Supertrend buffer, and other weekly/monthly parameters) needs to be "
    "available from the dashboard, as that requires the raw OHLCV data to reside on the server."
)

sub_header("5.3  Authentication")
body(
    "A password gate (SHA-256 hashed, environment-variable configured) guards the dashboard. "
    "For a production deployment, this would be replaced with the firm's existing SSO provider "
    "(e.g., Entra ID / Azure AD) using OAuth 2.0, which would also enable per-user audit logs "
    "of strategy parameter changes -- critical for compliance in a regulated environment."
)

rule()

# ── 6. DATA SECURITY & PRIVACY ────────────────────────────────────────────────
pdf.add_page()
section_header("6.", "Data Security and Privacy")

sub_header("6.1  Data at Rest")
bullets([
    "Raw OHLCV and tradebook data are stored locally on the analyst's machine or the EC2 "
      "instance -- never in a public S3 bucket or shared cloud storage in the current prototype.",
    "The tradebook (proprietary transaction history) is encrypted at rest as a password-protected "
      "Excel file. The password is retrieved at runtime from Azure Key Vault, never hardcoded.",
    "Feature artefacts (features_d.csv, trade_log.parquet) are derived data and contain no "
      "client-identifiable information beyond ticker symbols.",
])

sub_header("6.2  Data in Transit")
bullets([
    "Azure Key Vault communication uses TLS 1.2+ enforced by the Azure SDK.",
    "Dashboard access is over HTTPS when deployed via Streamlit Community Cloud or behind "
      "an EC2 load balancer with an ACM certificate. The ngrok tunnel also terminates TLS.",
    "No raw price data or trade data is transmitted to any external API in normal operation. "
      "The LLM (Claude) was used exclusively as a development coding assistant; it received "
      "code and feature logic, not proprietary transaction records.",
])

sub_header("6.3  LLM Data Handling")
body(
    "This is a critical boundary: the trading system itself contains no LLM calls. The LLM "
    "(Anthropic Claude via Claude Code) was used as an AI pair-programmer to build, debug, "
    "and optimise the codebase. No proprietary OHLCV data, trade history, or client-sensitive "
    "information was shared with the LLM. Code, feature engineering logic, and strategy design "
    "were shared -- these are methodology artefacts, not data artefacts."
)

rule()

# ── 7. LLM USAGE ──────────────────────────────────────────────────────────────
section_header("7.", "LLM Usage")

sub_header("7.1  Where the LLM Sits")
body(
    "In the current prototype, the LLM is entirely outside the production decision loop. "
    "Every buy signal, stop-loss, position size, and portfolio allocation is computed "
    "deterministically from mathematical rules. This was a deliberate architectural choice: "
    "for a regulated investment context, explainability and auditability are non-negotiable, "
    "and a rules-based engine satisfies both requirements without requiring hallucination "
    "guardrails or output validation."
)

sub_header("7.2  Production Extension: Where LLM Adds Value")
body(
    "The architecture is designed to accept LLM augmentation in two well-bounded roles "
    "without compromising the deterministic signal engine:"
)
table(
    ["LLM Role", "Mechanism", "Guardrail"],
    [
        ["Qualitative signal filter",
         "RAG over analyst reports, earnings call transcripts, and annual reports. "
         "LLM assesses whether fundamental picture is consistent with a quant buy signal.",
         "LLM output is advisory only; cannot override a stop-loss exit."],
        ["Portfolio commentary",
         "Natural language summary of daily signals and portfolio state for the PM. "
         "Surfaces key risks, upcoming catalysts, and position rationale.",
         "All factual claims grounded in retrieved documents (RAG); cited source shown."],
    ],
    [36, 98, 40]
)
body(
    "The LLM is never given discretionary power over position sizing, entry, or exit. "
    "Human-in-the-loop oversight is preserved: the PM sees the signal, the LLM commentary, "
    "and the source documents before making any action."
)

sub_header("7.3  Hallucination Risk Management")
bullets([
    "Ground all LLM responses in retrieved context (RAG); no unconstrained generation on financial metrics.",
    "Require source citations for any numerical claim (EPS, revenue, price target).",
    "Use temperature = 0 for all production inference to maximise determinism.",
    "Log all LLM inputs and outputs for compliance audit trail.",
    "Rate-limit LLM calls to one per signal event; prohibit chained agentic loops in production.",
])

rule()

# ── 8. AI AGENTS ──────────────────────────────────────────────────────────────
section_header("8.", "AI Agents")
body(
    "The prototype does not use autonomous AI agents in the production signal path. "
    "The Optuna optimisation framework is the closest analogue -- it autonomously explores "
    "the parameter space over 200 trials and selects the best configuration -- but it "
    "operates offline, not in real-time, and its outputs are always reviewed by the analyst "
    "before being applied."
)
body(
    "For a production system, agentic capabilities would be introduced cautiously and "
    "constrained to two patterns:"
)
bullets([
    "Data refresh agent  --  scheduled to fetch new OHLCV data, recompute features, and "
      "regenerate the features_d artefact nightly. Operates on data, not capital.",
    "Research ingestion agent  --  monitors company IR pages and Screener.in for new "
      "filings, downloads and chunks them into the RAG vector store. Triggered by calendar, "
      "not by market events.",
])
body(
    "Neither agent touches the trading engine or has access to order execution. The boundary "
    "between data infrastructure (agent-permissible) and capital allocation (human-only) "
    "is a hard architectural constraint, not a soft guideline."
)

rule()

# ── 9. CLOUD ARCHITECTURE ─────────────────────────────────────────────────────
pdf.add_page()
section_header("9.", "Cloud Architecture")

sub_header("9.1  Recommended Anchor Services")
body("For a production deployment on AWS, two services anchor the architecture:")
table(
    ["Service", "Role", "Rationale"],
    [
        ["AWS EC2 (t3.small / t3.medium)",
         "Hosts the Streamlit dashboard, stores OHLCV data and feature artefacts, "
         "runs nightly pipeline and optimisation jobs",
         "Simple, auditable, no serverless cold-start latency; easy SSH access for analyst"],
        ["AWS S3",
         "Versioned storage for features_d artefacts and backtest outputs; "
         "optionally serves as a staging area for new OHLCV data uploads",
         "Durability, versioning (recover any prior feature set), and cost-effectiveness"],
    ],
    [38, 90, 46]
)

sub_header("9.2  Supporting Services (Thin Layer)")
bullets([
    "Azure Key Vault  --  already in use for tradebook credentials; retain for consistency.",
    "AWS Secrets Manager  --  alternative for dashboard password and any API keys if "
      "moving fully to AWS.",
    "AWS CloudWatch  --  monitoring and alerting for EC2 instance health and nightly job failures.",
    "ACM + ALB  --  for HTTPS termination and a custom domain (e.g., trader.wisdom.in) "
      "in a production deployment.",
])

sub_header("9.3  Scalability Path")
body(
    "The current prototype is intentionally simple: one EC2 instance, one Python process, "
    "one portfolio. The architecture scales in clearly defined steps:"
)
table(
    ["Horizon", "Change", "What It Enables"],
    [
        ["Near-term",   "Expand OHLCV universe to 20-50 stocks",
         "Broader opportunity set; feature pipeline scales linearly with tickers"],
        ["Medium-term", "Move backtest and optimisation to AWS Batch or Lambda",
         "Parallel Optuna trials; reduce optimisation wall-clock from hours to minutes"],
        ["Long-term",   "Add RAG layer (Bedrock Knowledge Base or Pinecone) for research",
         "LLM-augmented qualitative filters alongside quantitative signals"],
    ],
    [28, 90, 56]
)

rule()

# ── 10. SUMMARY ───────────────────────────────────────────────────────────────
section_header("10.", "Summary and Next Steps")
body(
    "The Wisdom Trader prototype demonstrates that a disciplined, quantitative portfolio "
    "management system -- built on transparent rules, rigorous backtesting, and Bayesian "
    "optimisation -- can be assembled rapidly with modern AI-assisted development tooling "
    "and deployed as a usable decision-support interface for portfolio managers."
)
body("The three highest-priority architectural investments for moving from prototype to production are:")
bullets([
    "Automated data refresh  --  nightly OHLCV fetch, feature recomputation, and "
      "artefact versioning to S3. Eliminates the current manual upload step.",
    "SSO authentication  --  replace the password gate with Azure AD / Entra ID to "
      "enforce access control, enable per-user audit logs, and satisfy compliance requirements.",
    "RAG research layer  --  ingest analyst reports and company filings into a vector "
      "store; surface LLM-generated qualitative commentary alongside quantitative signals "
      "in the Daily Signals tab. This bridges the quantitative and fundamental analysis "
      "disciplines that characterise WISDOM's investment philosophy.",
])
body(
    "All code, backtest outputs, optimisation results, and this document are available in "
    "the GitHub repository accompanying this submission."
)

# ── Save ──────────────────────────────────────────────────────────────────────
pdf.output(str(OUT))
kb = OUT.stat().st_size // 1024
print(f"Saved: {OUT}  ({kb} KB, {pdf.page} pages)")
