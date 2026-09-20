"""
05_model_baseline.py — Tabular-only Baseline Models

Purpose:
    Predict the log listing multiple using ONLY tabular features (no text yet).
    Train two models and compare them fairly via cross-validation:
      - Ridge (regularized linear)  -> sensitive to multicollinearity
      - GradientBoosting (trees)    -> robust to multicollinearity
    Report metrics on the log target (MAE / RMSE) plus error in original units.

Feature-selection decisions (from 03_eda):
    - PERFECT redundancy (corr +/-1.00): net_margin, expense_ratio, profit_margin.
      Keep net_margin only; drop the other two. (Mathematically identical info.)
    - STRONG redundancy (corr 0.92): net_profit(log) vs gross_revenue(log).
      Kept both; Ridge uses regularization, GBR is robust. Interpretation handled in 07.

Leakage:
    Price fields were already dropped in 02. This script does not re-introduce them.

Run:
    python 05_model_baseline.py

Output:
    outputs/baseline_cv_results.csv     (metrics per model)
    outputs/baseline_predictions.csv    (out-of-fold predictions, for later error analysis)
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

# Feature-selection rule from EDA: keep net_margin, drop these perfect duplicates.
DROP_REDUNDANT = ["expense_ratio", "profit_margin"]

NUMERIC_FEATURES = [
    "log_average_annual_net_profit",
    "log_average_annual_gross_revenue",
    "business_age_months",
    "net_margin",               # kept; expense_ratio/profit_margin dropped
    "hours_worked_per_week",
    "days_on_marketplace",
    "monetizations_count",
    "niches_count",
    "amazon_sku_count",
]

BOOL_FEATURES = [
    "has_trademark",
    "uses_pbn",
    "private_lender_approved",
    "patent_pending",
    "patented_design",
    "patented_utility",
]

CATEGORICAL_FEATURES = [
    "primary_monetization",
    "primary_niche",
    "country",
    "source",
]

N_SPLITS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("features_*.csv"))
    if not files:
        raise FileNotFoundError("No processed features. Run 02_clean_features first.")
    df = pd.read_csv(files[-1])
    print(f"Loaded {files[-1].name}: {df.shape}")
    return df


def first_of(x, key):
    """Extract the first element's value from a stringified list of dicts/strings."""
    try:
        lst = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(lst, list) and lst:
            item = lst[0]
            return item.get(key) if isinstance(item, dict) else item
    except Exception:
        pass
    return "Unknown"


def prepare(df: pd.DataFrame):
    # Derive primary categorical values
    df["primary_monetization"] = df["monetizations"].apply(lambda x: first_of(x, "monetization"))
    df["primary_niche"] = df["niches"].apply(lambda x: first_of(x, "niche"))

    # Enforce redundancy drop (safety — even if columns exist, we won't select them)
    for c in DROP_REDUNDANT:
        if c in df.columns:
            df = df.drop(columns=c)

    # Keep only features that actually exist
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    boo = [c for c in BOOL_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Coerce numerics
    for c in num + boo:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[num + boo + cat].copy()
    y_log = df[TARGET_LOG].values
    y_raw = df[TARGET_RAW].values
    return X, y_log, y_raw, num, boo, cat, df


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def build_pipelines(num, boo, cat):
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

    # Ridge: needs scaling (linear, distance-sensitive)
    pre_ridge = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num + boo),
        ("cat", ohe, cat),
    ])
    ridge = Pipeline([("pre", pre_ridge), ("model", Ridge(alpha=1.0))])

    # GBR: trees don't need scaling
    pre_gbr = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num + boo),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    gbr = Pipeline([("pre", pre_gbr),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])

    return {"Ridge": ridge, "GradientBoosting": gbr}


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(models, X, y_log, y_raw):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []
    predictions = {"y_raw": y_raw, "y_log": y_log}

    # Naive baseline: predict the mean multiple
    mean_pred_raw = np.full_like(y_raw, np.mean(y_raw), dtype=float)
    results.append({
        "model": "Baseline (mean)",
        "log_MAE": np.nan, "log_RMSE": np.nan, "log_R2": np.nan,
        "raw_MAE_multiple": mean_absolute_error(y_raw, mean_pred_raw),
    })

    for name, pipe in models.items():
        pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
        pred_raw = np.expm1(pred_log)  # back to original units

        results.append({
            "model": name,
            "log_MAE": mean_absolute_error(y_log, pred_log),
            "log_RMSE": np.sqrt(mean_squared_error(y_log, pred_log)),
            "log_R2": r2_score(y_log, pred_log),
            "raw_MAE_multiple": mean_absolute_error(y_raw, pred_raw),
        })
        predictions[f"pred_{name}"] = pred_raw
        print(f"{name}: logMAE={results[-1]['log_MAE']:.4f}  "
              f"logRMSE={results[-1]['log_RMSE']:.4f}  "
              f"logR2={results[-1]['log_R2']:.4f}  "
              f"rawMAE={results[-1]['raw_MAE_multiple']:.3f}x")

    return pd.DataFrame(results), pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    X, y_log, y_raw, num, boo, cat, df = prepare(df)

    print(f"\nFeatures used: {len(num)} numeric, {len(boo)} boolean, {len(cat)} categorical")
    print(f"Numeric: {num}")
    print(f"Categorical: {cat}")
    print(f"Rows: {len(X)}\n")

    models = build_pipelines(num, boo, cat)
    results_df, preds_df = evaluate(models, X, y_log, y_raw)

    results_df.to_csv(OUTPUTS_DIR / "baseline_cv_results.csv", index=False)
    preds_df.to_csv(OUTPUTS_DIR / "baseline_predictions.csv", index=False)

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {OUTPUTS_DIR / 'baseline_cv_results.csv'}")
    print(f"Saved: {OUTPUTS_DIR / 'baseline_predictions.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
