"""
06c_compare_pca.py — Compare PCA Dimensions for Qwen3 Embeddings

Purpose:
    Test whether keeping more PCA components improves multiple prediction.
    Compares tabular-only against Tabular + Qwen3 embeddings at several PCA
    sizes (50 / 100 / 200 / 300), using the SAME cross-validation as 05/06.

    Hypothesis: R2 may rise then plateau or DROP as dims grow, because 2,848
    rows can't support very high-dimensional text features (overfitting).

Inputs (local data/processed/):
    features_YYYY-MM-DD.csv
    text_embeddings_qwen_pca50_YYYY-MM-DD.csv
    text_embeddings_qwen_pca100_YYYY-MM-DD.csv
    text_embeddings_qwen_pca200_YYYY-MM-DD.csv
    text_embeddings_qwen_pca300_YYYY-MM-DD.csv

Run:
    python 06c_compare_pca.py

Output:
    outputs/pca_sweep_comparison.csv
"""

from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_LOG = "log_target"
TARGET_RAW = "annual_listing_multiple"
DROP_REDUNDANT = ["expense_ratio", "profit_margin"]

NUMERIC_FEATURES = [
    "log_average_annual_net_profit", "log_average_annual_gross_revenue",
    "business_age_months", "net_margin", "hours_worked_per_week",
    "days_on_marketplace", "monetizations_count", "niches_count", "amazon_sku_count",
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


def load_base() -> pd.DataFrame:
    feat_files = sorted(PROCESSED_DIR.glob("features_*.csv"))
    if not feat_files:
        raise FileNotFoundError("No features_*.csv. Run 02 first.")
    feat = pd.read_csv(feat_files[-1])
    feat["primary_monetization"] = feat["monetizations"].apply(lambda x: first_of(x, "monetization"))
    feat["primary_niche"] = feat["niches"].apply(lambda x: first_of(x, "niche"))
    for c in DROP_REDUNDANT:
        if c in feat.columns:
            feat = feat.drop(columns=c)
    return feat


def discover_pca_files() -> dict:
    """Find qwen PCA files and map {dim: path}."""
    files = sorted(PROCESSED_DIR.glob("text_embeddings_qwen_pca*_*.csv"))
    out = {}
    for f in files:
        m = re.search(r"pca(\d+)_", f.name)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def make_models(numeric_block, cat):
    pre_ridge = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric_block),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    ridge = Pipeline([("pre", pre_ridge), ("model", Ridge(alpha=1.0))])

    pre_gbr = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_block),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    gbr = Pipeline([("pre", pre_gbr),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    return {"Ridge": ridge, "GradientBoosting": gbr}


def eval_condition(df, num, boo, cat, text_cols, cv):
    numeric_block = num + boo + text_cols
    X = df[numeric_block + cat].copy()
    y_log = df[TARGET_LOG].values
    y_raw = df[TARGET_RAW].values
    rows = []
    for name, pipe in make_models(numeric_block, cat).items():
        pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
        pred_raw = np.expm1(pred_log)
        rows.append({
            "model": name,
            "log_R2": r2_score(y_log, pred_log),
            "raw_MAE_multiple": mean_absolute_error(y_raw, pred_raw),
        })
    return rows


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    feat = load_base()
    num = [c for c in NUMERIC_FEATURES if c in feat.columns]
    boo = [c for c in BOOL_FEATURES if c in feat.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in feat.columns]
    for c in num + boo:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # Tabular only
    for r in eval_condition(feat, num, boo, cat, [], cv):
        r["pca_dims"] = 0
        r["condition"] = "Tabular only"
        results.append(r)
        print(f"[Tabular only    ] {r['model']:16s} R2={r['log_R2']:.4f} rawMAE={r['raw_MAE_multiple']:.3f}x")

    # Each PCA dimension
    pca_files = discover_pca_files()
    if not pca_files:
        print("\nNo qwen PCA files found (text_embeddings_qwen_pca*_*.csv).")
        print("Run 04b in Colab and download the files first.")
    for dim, path in pca_files.items():
        emb = pd.read_csv(path)
        merged = feat.merge(emb, on="id", how="inner").copy()
        text_cols = [c for c in merged.columns if c.startswith("text_emb_")]
        for c in text_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        for r in eval_condition(merged, num, boo, cat, text_cols, cv):
            r["pca_dims"] = dim
            r["condition"] = f"Qwen PCA-{dim}"
            results.append(r)
            print(f"[Qwen PCA-{dim:<4d}    ] {r['model']:16s} R2={r['log_R2']:.4f} rawMAE={r['raw_MAE_multiple']:.3f}x")

    res_df = pd.DataFrame(results)[["condition", "pca_dims", "model", "log_R2", "raw_MAE_multiple"]]
    res_df.to_csv(OUTPUTS_DIR / "pca_sweep_comparison.csv", index=False)

    print("\n=== R2 by PCA dimension (GradientBoosting) ===")
    gbr = res_df[res_df.model == "GradientBoosting"].sort_values("pca_dims")
    for _, row in gbr.iterrows():
        bar = "#" * int(max(0, row["log_R2"]) * 80)
        print(f"  {row['condition']:16s} R2={row['log_R2']:.4f}  {bar}")

    print(f"\nSaved: {OUTPUTS_DIR / 'pca_sweep_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
