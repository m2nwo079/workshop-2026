"""
06b_compare_embeddings.py — Compare Embedding Models

Purpose:
    Compare THREE conditions with the SAME cross-validation (5-fold, fixed seed),
    to see whether text helps and whether a stronger embedding helps more:
      - Tabular only
      - Tabular + MiniLM text embeddings   (all-MiniLM-L6-v2, 384d -> PCA 50)
      - Tabular + Qwen3 text embeddings     (Qwen3-Embedding-0.6B, 1024d -> PCA 50)

    Runs Ridge and GradientBoosting for every condition.

Embedding files (auto-detected in data/processed/):
    text_embeddings_*.csv        -> MiniLM (no 'qwen' in name)
    text_embeddings_qwen_*.csv   -> Qwen3

Feature-selection carried from 03/05:
    - Drop perfect duplicates: expense_ratio, profit_margin (keep net_margin).
    - Price fields already excluded in 02 (leakage guard).

Run:
    python 06b_compare_embeddings.py

Output:
    outputs/embedding_comparison.csv
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Discover embedding files
# ---------------------------------------------------------------------------
def discover_embeddings() -> dict:
    """Return {label: path} for each embedding file found."""
    files = sorted(PROCESSED_DIR.glob("text_embeddings_*.csv"))
    sources = {}
    for f in files:
        label = "Qwen3-0.6B" if "qwen" in f.name.lower() else "MiniLM-L6"
        # keep the most recent for each label
        sources[label] = f
    return sources


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


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
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


def eval_condition(df, num, boo, cat, text_cols, y_log, y_raw, cv):
    numeric_block = num + boo + text_cols
    X = df[numeric_block + cat].copy()
    rows = []
    for name, pipe in make_models(numeric_block, cat).items():
        pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
        pred_raw = np.expm1(pred_log)
        rows.append({
            "model": name,
            "log_R2": r2_score(y_log, pred_log),
            "log_MAE": mean_absolute_error(y_log, pred_log),
            "raw_MAE_multiple": mean_absolute_error(y_raw, pred_raw),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    feat = load_base()
    emb_sources = discover_embeddings()
    print("Embedding files found:")
    for label, path in emb_sources.items():
        print(f"  {label:12s} <- {path.name}")
    print()

    num = [c for c in NUMERIC_FEATURES if c in feat.columns]
    boo = [c for c in BOOL_FEATURES if c in feat.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in feat.columns]
    for c in num + boo:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # --- Condition 1: Tabular only (base df, no text) ---
    y_log = feat[TARGET_LOG].values
    y_raw = feat[TARGET_RAW].values
    for r in eval_condition(feat, num, boo, cat, [], y_log, y_raw, cv):
        r["condition"] = "Tabular only"
        results.append(r)
        print(f"[Tabular only  ] {r['model']:16s} R2={r['log_R2']:.4f} rawMAE={r['raw_MAE_multiple']:.3f}x")

    # --- Conditions 2+: Tabular + each embedding ---
    for label, path in emb_sources.items():
        emb = pd.read_csv(path)
        merged = feat.merge(emb, on="id", how="inner").copy()
        text_cols = [c for c in merged.columns if c.startswith("text_emb_")]
        for c in text_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        yl = merged[TARGET_LOG].values
        yr = merged[TARGET_RAW].values
        for r in eval_condition(merged, num, boo, cat, text_cols, yl, yr, cv):
            r["condition"] = f"Tabular + {label}"
            results.append(r)
            print(f"[Tab + {label:10s}] {r['model']:16s} R2={r['log_R2']:.4f} rawMAE={r['raw_MAE_multiple']:.3f}x")

    res_df = pd.DataFrame(results)[["condition", "model", "log_R2", "log_MAE", "raw_MAE_multiple"]]
    res_df.to_csv(OUTPUTS_DIR / "embedding_comparison.csv", index=False)

    print("\n=== Comparison (pivot on GradientBoosting) ===")
    gbr = res_df[res_df.model == "GradientBoosting"].set_index("condition")
    print(gbr[["log_R2", "raw_MAE_multiple"]].round(4).to_string())

    print(f"\nSaved: {OUTPUTS_DIR / 'embedding_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
