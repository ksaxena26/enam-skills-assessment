from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
FEAT_DIR = DATA_DIR / "features"
FIN_DIR = DATA_DIR / "financials"
TB_DIR = ROOT / "tradebook"
