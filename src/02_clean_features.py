"""
02_clean_features.py — Cleaning & Feature Engineering

Purpose:
    Read the raw snapshots from data/raw/, clean types, build derived features,
    log-transform the target, and ENFORCE the price-field exclusion rule (leakage guard).
    Output a processed table to data/processed/ for later steps.

Target:
    annual_listing_multiple (listing multiple).
    Confirmed in 01: the sale multiple is NOT in the API response, so the
    listing multiple is the primary target. Distribution is right-skewed
    -> log transform by default.

Leakage guard (IMPORTANT):
    multiple ~= price / net_profit. Including any price field as a feature leaks
    the answer. Price-related fields are dropped here in code.

Run:
    python 02_clean_features.py

Output:
    data/processed/features_YYYY-MM-DD.csv     (numeric/boolean features + target)
    data/processed/text_private_YYYY-MM-DD.csv (raw text: opportunities/risks/summary)
                                               -> kept separate; goes to data/private-like handling.

Note on text:
    Raw text columns are written to a SEPARATE file so the main feature table
    can be shared while raw text stays private (copyright). Move/keep the text
    file only locally; do NOT commit it.
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

# Price-related / leakage fields to DROP (never used as features).
# multiple = price / net_profit, so price and the raw multiple leak the target.
LEAKAGE_FIELDS = [
    "listing_price",
    "unpriced",
    "listing_multiple",        # raw (non-annualized) multiple — same leakage family
    "first_listing_price",     # may not exist; dropped only if present
]

# Numeric fields to coerce to numbers
NUMERIC_FIELDS = [
    "average_annual_net_profit",
    "average_annual_gross_revenue",
    "average_annual_expenses",
    "hours_worked_per_week",
    "days_on_marketplace",
    "profit_margin",
    "amazon_sku_count",
    "amazon_parent_asin_count",
    "pricing_period_months",
]

# Boolean fields -> 0/1
BOOL_FIELDS = [
    "has_trademark",
    "uses_pbn",
    "private_lender_approved",
    "patent_pending",
    "patented_design",
    "patented_utility",
    "usa_made",
    "has_vat_registrations",
]

# List-type fields we turn into simple counts here (multi-hot comes in a later step)
LIST_COUNT_FIELDS = ["monetizations", "niches", "opportunities", "work_required"]

# Raw text fields -> written to a separate PRIVATE file (not shared)
TEXT_FIELDS = ["opportunities", "risks", "summary", "reason_for_sale", "seller_support"]

# Financial fields to log-transform (scale signal)
LOG_FINANCIAL_FIELDS = [
    "average_annual_net_profit",
    "average_annual_gross_revenue",
    "average_annual_expenses",
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

    # 2) Target log transform (right-skewed -> log)
    df["log_target"] = np.log1p(df[TARGET])

    # 3) Numeric coercion
    for c in NUMERIC_FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

    # 7) Booleans -> 0/1 (nullable Int)
    for c in BOOL_FIELDS:
        if c in df.columns:
            df[c] = df[c].astype("boolean").astype("Int64")

    # 8) List fields -> counts
    for c in LIST_COUNT_FIELDS:
        if c in df.columns:
            df[f"{c}_count"] = df[c].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 9) Country: fill missing with "Unknown" (74.5% coverage in 01)
    for c in ["country", "country_of_business_registration"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    return df


def enforce_leakage_guard(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all price/leakage fields. This is the hard rule from the project doc."""
    to_drop = [c for c in LEAKAGE_FIELDS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Leakage guard: dropped {to_drop}")
    # Safety assert: none of the leakage fields survive
    remaining = [c for c in LEAKAGE_FIELDS if c in df.columns]
    assert not remaining, f"Leakage fields still present: {remaining}"
    return df


def split_and_save(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Private text file (keep local, do NOT commit) ---
    text_cols = [c for c in TEXT_FIELDS if c in df.columns]
    id_cols = [c for c in ["id", "listing_number"] if c in df.columns]
    text_df = df[id_cols + text_cols].copy()
    text_path = PRIVATE_DIR / f"text_private_{TODAY}.csv"
    text_df.to_csv(text_path, index=False)
    print(f"Saved PRIVATE text: {text_path} (keep local, do not commit)")

    # --- Feature table (shareable): drop raw text + heavy nested fields ---
    drop_nested = [
        "sites", "metrics", "combined_site_metrics", "private_lenders",
        "data_providers", "assets_included", "work_required",
        "niche_image", "custom_niche_image_path", "seller_interview_link",
        "public_title",  # title can contain price hints; drop from features
    ]
    drop_cols = [c for c in (TEXT_FIELDS + drop_nested) if c in df.columns]
    feat_df = df.drop(columns=drop_cols)

    feat_path = PROCESSED_DIR / f"features_{TODAY}.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"Saved features: {feat_path}  shape={feat_df.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Date (UTC): {TODAY}")
    df = load_all()
    df = clean(df)
    df = enforce_leakage_guard(df)
    split_and_save(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
