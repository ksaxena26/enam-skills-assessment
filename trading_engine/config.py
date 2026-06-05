INITIAL_CAPITAL: float = 100_000_000   # Starting portfolio cash (INR or USD, unitless)
MAX_SECURITY_ALLOC: float = 0.33       # Max fraction of portfolio value in one security
MAX_RISK_PER_TRADE: float = 0.01       # Max capital at risk per new buy (1% of portfolio)
EXECUTION_PRICE_COL: str = "close"     # Price column used for fills
