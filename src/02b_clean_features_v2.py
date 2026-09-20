"""
02b_clean_features_v2.py — Cleaning & Feature Engineering (v2, with trend features)

Same as 02_clean_features.py, PLUS growth-trend features that 03/experiments
showed to be the single most useful addition:

    net_profit_trend_percent, gross_revenue_trend_percent

Experiment result (GBR, tabular only, 5-fold CV, same seed):
    base features         R2 0.398
    + trend               R2 0.442   <- big gain (+0.044)
    + traffic             R2 0.398   (no gain, dropped)
    + assets_count        R2 0.401   (negligible, dropped)

So this v2 adds ONLY the trend features (plus missing flags), and deliberately
does NOT add traffic/assets, which did not help and would only add overfitting risk.

Everything else (leakage guard, target log-transform, private text split) is
identical to 02.

Run:
    python 02b_clean_features_v2.py

Output:
    data/processed/features_v2_YYYY-MM-DD.csv     (feature table, includes trend)
    data/private/text_private_YYYY-MM-DD.csv      (unchanged; raw text stays private)
"""

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"

TARGET = "annual_listing_multiple"
TODAY = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

LEAKAGE_FIELDS = [
    "listing_price", "unpriced", "listing_multiple", "first_listing_price",
]

NUMERIC_FIELDS = [
    "average_annual_net_profit", "average_annual_gross_revenue",
    "average_annual_expenses", "hours_worked_per_week", "days_on_marketplace",
    "profit_margin", "amazon_sku_count", "amazon_parent_asin_count",
    "pricing_period_months",
]

# NEW in v2: growth-trend fields (the useful addition)
TREND_FIELDS = ["net_profit_trend_percent", "gross_revenue_trend_percent"]

BOOL_FIELDS = [
    "has_trademark", "uses_pbn", "private_lender_approved",
    "patent_pending", "patented_design", "patented_utility",
    "usa_made", "has_vat_registrations",
]

LIST_COUNT_FIELDS = ["monetizations", "niches", "opportunities", "work_required"]
TEXT_FIELDS = ["opportunities", "risks", "summary", "reason_for_sale", "seller_support"]
LOG_FINANCIAL_FIELDS = [
    "average_annual_net_profit", "average_annual_gross_revenue", "average_annual_expenses",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_latest(label: str) -> list:
    files = sorted(RAW_DIR.glob(f"listings_{label}_*.json"))
    if not files:
        raise FileNotFoundError(f"No '{label}' snapshot in {RAW_DIR}. Run 00_ingest first.")
    with open(files[-1], encoding="utf-8") as f:
        snap = json.load(f)
    print(f"[{label}] {files[-1].name} -> {snap.get('count')} listings")
    return snap["listings"]


def load_all() -> pd.DataFrame:
    fs = pd.DataFrame(load_latest("forsale"))
    fs["source"] = "For Sale"
    sd = pd.DataFrame(load_latest("sold"))
    sd["source"] = "Sold"
    df = pd.concat([fs, sd], ignore_index=True)
    print(f"Combined: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Clean & engineer
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Target -> numeric, drop missing / non-positive
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    print(f"Valid target rows: {len(df)} (dropped {before - len(df)})")

    # 2) Target log transform
    df["log_target"] = np.log1p(df[TARGET])

    # 3) Numeric coercion
    for c in NUMERIC_FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3b) NEW: trend fields -> numeric + missing flag
    #     ~43% are missing, so the flag lets the model know when trend is unknown.
    for c in TREND_FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[f"{c}_missing"] = df[c].isna().astype(int)

    # 4) Derived: business age in months (UTC)
    today = pd.Timestamp.now(tz="UTC")
    fmm = pd.to_datetime(df.get("first_made_money_at"), errors="coerce", utc=True)
    df["business_age_months"] = ((today - fmm).dt.days / 30.44).round(1)

    # 5) Derived: margins (guard against divide-by-zero)
    gross = df.get("average_annual_gross_revenue")
    if gross is not None:
        df["net_margin"] = np.where(gross > 0, df["average_annual_net_profit"] / gross, np.nan)
        df["expense_ratio"] = np.where(gross > 0, df["average_annual_expenses"] / gross, np.nan)

    # 6) Missing flags + log for financial scale fields
    for c in LOG_FINANCIAL_FIELDS:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype(int)
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

    # 7) Booleans -> 0/1
    for c in BOOL_FIELDS:
        if c in df.columns:
            df[c] = df[c].astype("boolean").astype("Int64")

    # 8) List fields -> counts
    for c in LIST_COUNT_FIELDS:
        if c in df.columns:
            df[f"{c}_count"] = df[c].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 9) Country fill
    for c in ["country", "country_of_business_registration"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    return df


def enforce_leakage_guard(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in LEAKAGE_FIELDS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Leakage guard: dropped {to_drop}")
    remaining = [c for c in LEAKAGE_FIELDS if c in df.columns]
    assert not remaining, f"Leakage fields still present: {remaining}"
    return df


def split_and_save(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    # Private text file (unchanged)
    text_cols = [c for c in TEXT_FIELDS if c in df.columns]
    id_cols = [c for c in ["id", "listing_number"] if c in df.columns]
    text_df = df[id_cols + text_cols].copy()
    text_path = PRIVATE_DIR / f"text_private_{TODAY}.csv"
    text_df.to_csv(text_path, index=False)
    print(f"Saved PRIVATE text: {text_path} (keep local, do not commit)")

    # Feature table (v2 filename so it sits beside the v1 features)
    drop_nested = [
        "sites", "metrics", "combined_site_metrics", "private_lenders",
        "data_providers", "assets_included", "work_required",
        "niche_image", "custom_niche_image_path", "seller_interview_link",
        "public_title",
    ]
    drop_cols = [c for c in (TEXT_FIELDS + drop_nested) if c in df.columns]
    feat_df = df.drop(columns=drop_cols)

    feat_path = PROCESSED_DIR / f"features_v2_{TODAY}.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Saved features v2: {feat_path}  shape={feat_df.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Date (UTC): {TODAY}")
    df = load_all()
    df = clean(df)
    df = enforce_leakage_guard(df)
    split_and_save(df)

    present_trend = [c for c in TREND_FIELDS if c in df.columns]
    print(f"\nTrend features added: {present_trend}")
    print("Done.")


if __name__ == "__main__":
    main()
