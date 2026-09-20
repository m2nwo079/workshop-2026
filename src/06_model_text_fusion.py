"""
06_model_text_fusion.py — Tabular + Text Fusion

Purpose:
    Answer the project's central question:
    "Does adding text embeddings improve over the tabular-only baseline?"

    Compare TWO conditions using the SAME cross-validation setup as 05
    (same folds, same seed) so the comparison is fair:
      - Tabular only          (reproduces the 05 baseline features)
      - Tabular + Text (fused) (adds PCA-reduced text embeddings from 04)

    Both conditions run Ridge and GradientBoosting. Metrics: log MAE/RMSE/R2
    plus error in original multiple units.

Inputs (local data/processed/):
    features_YYYY-MM-DD.csv          (from 02)
    text_embeddings_YYYY-MM-DD.csv   (from 04, downloaded from Drive)

Feature-selection decisions carried over from 03/05:
    - Drop perfect duplicates: expense_ratio, profit_margin (keep net_margin).
    - Price fields already excluded in 02 (leakage guard).

Run:
    python 06_model_text_fusion.py

Output:
    outputs/fusion_cv_results.csv   (metrics: 2 models x 2 conditions + baseline)
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
# Config (kept identical to 05 where it matters)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_LOG = "log_target"
TARGET_RAW = "annual_listing_multiple"

DROP_REDUNDANT = ["expense_ratio", "profit_margin"]

NUMERIC_FEATURES = [
    "log_average_annual_net_profit",
    "log_average_annual_gross_revenue",
    "business_age_months",
    "net_margin",
    "hours_worked_per_week",
    "days_on_marketplace",
    "monetizations_count",
    "niches_count",
    "amazon_sku_count",
]
BOOL_FEATURES = [
    "has_trademark", "uses_pbn", "private_lender_approved",
    "patent_pending", "patented_design", "patented_utility",
]
CATEGORICAL_FEATURES = ["primary_monetization", "primary_niche", "country", "source"]

N_SPLITS = 5
RANDOM_STATE = 42   # same seed as 05 -> same folds -> fair comparison


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------
def load_merged() -> pd.DataFrame:
    feat_files = sorted(PROCESSED_DIR.glob("features_*.csv"))
    emb_files = sorted(PROCESSED_DIR.glob("text_embeddings_*.csv"))
    if not feat_files:
        raise FileNotFoundError("No features_*.csv. Run 02_clean_features first.")
    if not emb_files:
        raise FileNotFoundError(
            "No text_embeddings_*.csv in data/processed/. "
            "Run 04 in Colab and download the result here first."
        )

    feat = pd.read_csv(feat_files[-1])
    emb = pd.read_csv(emb_files[-1])
    print(f"Features: {feat_files[-1].name} {feat.shape}")
    print(f"Embeddings: {emb_files[-1].name} {emb.shape}")

    merged = feat.merge(emb, on="id", how="inner")
    print(f"Merged on id: {merged.shape}  ({len(merged)}/{len(feat)} rows matched)")
    if len(merged) < len(feat):
        print(f"  Note: {len(feat) - len(merged)} listings had no embedding and were dropped.")
    return merged


def first_of(x, key):
    try:
        lst = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(lst, list) and lst:
            item = lst[0]
            return item.get(key) if isinstance(item, dict) else item
    except Exception:
        pass
    return "Unknown"


def prepare(df: pd.DataFrame):
    df = df.copy()   # de-fragment after the merge to avoid PerformanceWarning
    df["primary_monetization"] = df["monetizations"].apply(lambda x: first_of(x, "monetization"))
    df["primary_niche"] = df["niches"].apply(lambda x: first_of(x, "niche"))
    for c in DROP_REDUNDANT:
        if c in df.columns:
            df = df.drop(columns=c)

    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    boo = [c for c in BOOL_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    text = [c for c in df.columns if c.startswith("text_emb_")]

    for c in num + boo + text:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    y_log = df[TARGET_LOG].values
    y_raw = df[TARGET_RAW].values
    return df, num, boo, cat, text, y_log, y_raw


# ---------------------------------------------------------------------------
# Pipelines (built per condition so the text block is included or not)
# ---------------------------------------------------------------------------
def make_models(num, boo, cat, text, use_text: bool):
    numeric_block = num + boo + (text if use_text else [])

    # Ridge (scaled)
    pre_ridge = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric_block),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    ridge = Pipeline([("pre", pre_ridge), ("model", Ridge(alpha=1.0))])

    # GradientBoosting (no scaling needed)
    pre_gbr = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_block),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    gbr = Pipeline([("pre", pre_gbr),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])

    return {"Ridge": ridge, "GradientBoosting": gbr}


# ---------------------------------------------------------------------------
# Evaluate both conditions
# ---------------------------------------------------------------------------
def run(df, num, boo, cat, text, y_log, y_raw):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # naive baseline
    mean_pred = np.full_like(y_raw, np.mean(y_raw), dtype=float)
    results.append({"condition": "-", "model": "Baseline (mean)",
                    "log_MAE": np.nan, "log_RMSE": np.nan, "log_R2": np.nan,
                    "raw_MAE_multiple": mean_absolute_error(y_raw, mean_pred)})

    for use_text in [False, True]:
        cond = "Tabular + Text" if use_text else "Tabular only"
        numeric_block = num + boo + (text if use_text else [])
        X = df[numeric_block + cat].copy()
        models = make_models(num, boo, cat, text, use_text)

        for name, pipe in models.items():
            pred_log = cross_val_predict(pipe, X, y_log, cv=cv)
            pred_raw = np.expm1(pred_log)
            row = {
                "condition": cond, "model": name,
                "log_MAE": mean_absolute_error(y_log, pred_log),
                "log_RMSE": np.sqrt(mean_squared_error(y_log, pred_log)),
                "log_R2": r2_score(y_log, pred_log),
                "raw_MAE_multiple": mean_absolute_error(y_raw, pred_raw),
            }
            results.append(row)
            print(f"[{cond:15s}] {name:16s} "
                  f"logR2={row['log_R2']:.4f}  rawMAE={row['raw_MAE_multiple']:.3f}x")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    df, num, boo, cat, text, y_log, y_raw = prepare(df)
    print(f"\nTabular: {len(num)} numeric + {len(boo)} bool + {len(cat)} categorical")
    print(f"Text embeddings: {len(text)} dims")
    print(f"Rows: {len(df)}\n")

    results = run(df, num, boo, cat, text, y_log, y_raw)

    results.to_csv(OUTPUTS_DIR / "fusion_cv_results.csv", index=False)

    print("\n=== Results (does text help?) ===")
    print(results.to_string(index=False))

    # Quick verdict per model
    print("\n=== Verdict ===")
    for name in ["Ridge", "GradientBoosting"]:
        try:
            tab = results[(results.model == name) & (results.condition == "Tabular only")].iloc[0]
            fus = results[(results.model == name) & (results.condition == "Tabular + Text")].iloc[0]
            dR2 = fus["log_R2"] - tab["log_R2"]
            dMAE = fus["raw_MAE_multiple"] - tab["raw_MAE_multiple"]
            better = "text HELPS" if dR2 > 0 else "text does NOT help"
            print(f"{name}: log_R2 {tab['log_R2']:.3f} -> {fus['log_R2']:.3f} "
                  f"(delta {dR2:+.3f}); rawMAE {tab['raw_MAE_multiple']:.3f} -> "
                  f"{fus['raw_MAE_multiple']:.3f} (delta {dMAE:+.3f}) => {better}")
        except IndexError:
            pass

    print(f"\nSaved: {OUTPUTS_DIR / 'fusion_cv_results.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
