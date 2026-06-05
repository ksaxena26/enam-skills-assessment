# import os
# import numpy as np
import pandas as pd
from src.config.paths import FEAT_DIR
from src.data_manager import load_ohlc_data
from src.engineering import get_batches, compute_buy_range_u, compute_max_risk, compute_buy_flag
from src.engineering import compute_features_d, compute_features_w, compute_features_m
from src.engineering import map_weekly_to_daily, map_monthly_swings_to_daily
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
features_d = pd.DataFrame()
features_w = pd.DataFrame()
features_m = pd.DataFrame()

batches_d = get_batches(price_d)
batches_w = get_batches(price_w)
batches_m = get_batches(price_m)

for b in batches_d:
    features_d = pd.concat([features_d, compute_features_d(b)], ignore_index=True)

for b in batches_w:
    features_w = pd.concat([features_w, compute_features_w(b)], ignore_index=True)

for b in batches_m:
    features_m = pd.concat([features_m, compute_features_m(b)], ignore_index=True)

del batches_d, batches_w, batches_m

features_d = map_weekly_to_daily(features_d, features_w, ['supert', 'sl'])
features_d = map_monthly_swings_to_daily(features_d, features_m)
features_d = compute_buy_range_u(features_d, features_m)
features_d['risk'] = compute_max_risk(features_d, 'sl_w')
features_d['buy'] = compute_buy_flag(features_d)

benchmark_d = features_d[features_d['symbol'] == 'NIFTYBEES'].copy()
features_d = features_d[features_d['symbol'] != 'NIFTYBEES'].copy()

print("Saving features...")
features_d.to_csv(FEAT_DIR / "features_d.csv", index=False)
benchmark_d.to_csv(FEAT_DIR / "benchmark_d.csv", index=False)
features_w.to_csv(FEAT_DIR / "features_w.csv", index=False)
features_m.to_csv(FEAT_DIR / "features_m.csv", index=False)


