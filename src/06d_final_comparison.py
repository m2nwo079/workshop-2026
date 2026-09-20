"""
06d_final_comparison.py — Final Model Comparison

Purpose:
    The project's final scoreboard. Compares feature sets and text embeddings
    with the SAME cross-validation (5-fold, fixed seed) used throughout:

      Feature sets:
        v1  = base tabular features
        v2  = base + growth-trend features (net_profit / gross_revenue trend %)

      Text:
        none      = tabular only
        Qwen PCA-50 = best embedding from the embedding/PCA experiments

    So it reports up to four combinations plus a naive baseline, isolating the
    contribution of (a) trend features and (b) text, and showing they stack.

Key finding this script documents:
    trend and text carry DIFFERENT information, so their gains add up:
        v1 tabular            R2 ~0.40
        v1 + Qwen             R2 ~0.45   (text helps)
        v2 tabular (trend)    R2 ~0.45   (trend helps ~ as much as text)
        v2 + Qwen  (final)    R2 ~0.48   (both stack)

Inputs (local data/processed/):
    features_YYYY-MM-DD.csv                 (v1, from 02)
    features_v2_YYYY-MM-DD.csv              (v2, from 02b)
    text_embeddings_qwen_pca50_YYYY-MM-DD.csv   (from 04b)

Run:
    python 06d_final_comparison.py

Output:
    outputs/final_comparison.csv
"""

from pathlib import Path
import ast

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_LOG = "log_target"
TARGET_RAW = "annual_listing_multiple"
DROP_REDUNDANT = ["expense_ratio", "profit_margin"]

# Base tabular features (shared by v1 and v2)
BASE_NUMERIC = [
    "log_average_annual_net_profit", "log_average_annual_gross_revenue",
    "business_age_months", "net_margin", "hours_worked_per_week",
    "days_on_marketplace", "monetizations_count", "niches_count", "amazon_sku_count",
]
# Extra numeric features present only in v2
V2_EXTRA_NUMERIC = [
    "net_profit_trend_percent", "gross_revenue_trend_percent",
    "net_profit_trend_percent_missing", "gross_revenue_trend_percent_missing",
]
BOOL_FEATURES = [
    "has_trademark", "uses_pbn", "private_lender_approved",
    "patent_pending", "patented_design", "patented_utility",
]
CATEGORICAL_FEATURES = ["primary_monetization", "primary_niche", "country", "source"]

N_SPLITS = 5
RANDOM_STATE = 42


def first_of(x, key):
    try:
        lst = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(lst, list) and lst:
            item = lst[0]
            return item.get(key) if isinstance(item, dict) else item
    except Exception:
        pass
    return "Unknown"


def prep(df):
    df = df.copy()
    df["primary_monetization"] = df["monetizations"].apply(lambda x: first_of(x, "monetization"))
    df["primary_niche"] = df["niches"].apply(lambda x: first_of(x, "niche"))
    for c in DROP_REDUNDANT:
        if c in df.columns:
            df = df.drop(columns=c)
    return df


def numeric_features_for(df, use_trend: bool):
    """Base numeric, plus v2 trend extras only when use_trend is True.

    Note: the v1 CSV may still contain raw trend columns; we exclude them
    unless use_trend is set, so 'v1' is a genuine no-trend baseline.
    """
    feats = [c for c in BASE_NUMERIC if c in df.columns]
    if use_trend:
        feats += [c for c in V2_EXTRA_NUMERIC if c in df.columns]
    return feats


def make_models(numeric_block, cat):
    pre_ridge = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric_block),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    pre_gbr = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_block),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    return {
        "Ridge": Pipeline([("pre", pre_ridge), ("model", Ridge(alpha=1.0))]),
        "GradientBoosting": Pipeline([("pre", pre_gbr),
                                      ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
    }


def evaluate(df, text_cols, label, cv, results, use_trend):
    df = prep(df)
    num = numeric_features_for(df, use_trend)
    boo = [c for c in BOOL_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for c in num + boo + text_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_block = num + boo + text_cols
    X = df[numeric_block + cat].copy()
    y_log = df[TARGET_LOG].values
    y_raw = df[TARGET_RAW].values

    for name, pipe in make_models(numeric_block, cat).items():
        pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
        pred_raw = np.expm1(pred_log)
        row = {
            "condition": label, "model": name,
            "log_R2": r2_score(y_log, pred_log),
            "raw_MAE_multiple": mean_absolute_error(y_raw, pred_raw),
        }
        results.append(row)
        if name == "GradientBoosting":
            print(f"  {label:28s} GBR R2={row['log_R2']:.4f}  rawMAE={row['raw_MAE_multiple']:.3f}x")


def load_first(patterns):
    for pat in patterns:
        files = sorted(PROCESSED_DIR.glob(pat))
        if files:
            return files[-1]
    return None


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Locate files
    v1_path = load_first(["features_2*.csv"])          # v1: features_YYYY...
    v2_path = load_first(["features_v2_*.csv"])         # v2: features_v2_YYYY...
    qwen_path = load_first(["text_embeddings_qwen_pca50_*.csv", "text_embeddings_qwen_*.csv"])

    if v1_path is None or v2_path is None:
        raise FileNotFoundError("Need both features_*.csv (02) and features_v2_*.csv (02b).")
    print(f"v1 features : {v1_path.name}")
    print(f"v2 features : {v2_path.name}")
    print(f"qwen embed  : {qwen_path.name if qwen_path else 'NOT FOUND'}\n")

    v1 = pd.read_csv(v1_path)
    v2 = pd.read_csv(v2_path)
    qwen = pd.read_csv(qwen_path) if qwen_path else None

    results = []

    # Naive baseline (from v1's target)
    yr = v1[TARGET_RAW].values
    results.append({"condition": "Baseline (mean)", "model": "-",
                    "log_R2": np.nan,
                    "raw_MAE_multiple": mean_absolute_error(yr, np.full_like(yr, np.mean(yr), dtype=float))})
    print(f"  {'Baseline (mean)':28s}     rawMAE={results[-1]['raw_MAE_multiple']:.3f}x")

    # v1 tabular / v1 + text  (no trend features)
    evaluate(v1, [], "v1 tabular", cv, results, use_trend=False)
    if qwen is not None:
        m = v1.merge(qwen, on="id", how="inner")
        tc = [c for c in m.columns if c.startswith("text_emb_")]
        evaluate(m, tc, "v1 + Qwen", cv, results, use_trend=False)

    # v2 tabular (trend) / v2 + text  (with trend features)
    evaluate(v2, [], "v2 tabular (trend)", cv, results, use_trend=True)
    if qwen is not None:
        m = v2.merge(qwen, on="id", how="inner")
        tc = [c for c in m.columns if c.startswith("text_emb_")]
        evaluate(m, tc, "v2 + Qwen (final)", cv, results, use_trend=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUTS_DIR / "final_comparison.csv", index=False)

    print("\n=== Final scoreboard (GradientBoosting) ===")
    gbr = res_df[res_df.model == "GradientBoosting"]
    for _, r in gbr.iterrows():
        bar = "#" * int(max(0, r["log_R2"]) * 80)
        print(f"  {r['condition']:20s} R2={r['log_R2']:.4f}  {bar}")

    print(f"\nSaved: {OUTPUTS_DIR / 'final_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
