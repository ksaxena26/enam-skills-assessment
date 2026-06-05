# import os
# import numpy as np
# import pandas as pd
from src.config.paths import FEAT_DIR
from src.data_manager import load_ohlc_data
from src.engineering import compute_features_d, compute_features_w, compute_features_m
from src.utils.utils import daily_to_weekly_transform, daily_to_monthly_transform

print("Loading data...")
price_d = load_ohlc_data()

print("Applying higher timeframe transformations...")
price_w = daily_to_weekly_transform(price_d)
price_w = price_w.drop(columns=['prev_close',])
price_m = daily_to_monthly_transform(price_d)
price_m = price_m.drop(columns=['prev_close',])
price_d = price_d.drop(columns=['prev_close',])

print("Computing features...")
features_d = compute_features_d(price_d)
features_w = compute_features_w(price_w)
features_m = compute_features_m(price_m)

print("Saving features...")
features_d.to_csv(FEAT_DIR / "features_d.csv", index=False)
features_w.to_csv(FEAT_DIR / "features_w.csv", index=False)
features_m.to_csv(FEAT_DIR / "features_m.csv", index=False)





