# workshop-2026 — Predicting Online-Business Valuation Multiples

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-numeric_data-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-arrays-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBR%20%2B%20CV-F7931E?logo=scikitlearn&logoColor=white)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-MiniLM%20%2F%20Qwen3-EE4C2C?logo=huggingface&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-interpretability-1f77b4)
![Google Colab](https://img.shields.io/badge/Colab-GPU%20embeddings-F9AB00?logo=googlecolab&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
[![Live report](https://img.shields.io/badge/Live%20report-Open-7c2d3a)](https://m2nwo079.github.io/workshop-2026/)

A supervised regression pipeline that predicts the **valuation multiple** of
online businesses from their listing text and financial/operational metrics,
using the public [Empire Flippers API](https://api.empireflippers.com/).

**Central question:** *Beyond the level of profit, what drives the multiple up?*

**Live report:** https://m2nwo079.github.io/workshop-2026/

An interactive write-up of the full study is in [`index.html`](index.html).

---

## TL;DR — Key Findings

- **Two independent signals stack.** Growth-trend features and listing text each
  lift prediction by about the same amount, and because they carry *different*
  information, combining them adds up: cross-validated **R2 rose from 0.40 to 0.48**
  and mean error fell from **0.418 to 0.377** multiple points (GradientBoosting).
- **Text helps, and it's not an artifact.** Replacing the real embeddings with
  random noise of the same shape *lowered* performance — the gain comes from
  genuine text signal, not from adding columns.
- **Growth trend is the strongest single addition.** Adding profit/revenue trend
  raised the tabular baseline from 0.40 to 0.45 — as much as text did. Traffic
  volume (page views, unique users), by contrast, added nothing.
- **Bigger embeddings and more dimensions aren't automatically better.** Qwen3
  edged out MiniLM, but only slightly and at ~50x the compute; and R2 peaked at
  ~30-50 PCA dimensions, then *fell* as dimensions grew (overfitting on ~2,850 rows).
- **What lifts a multiple:** larger net profit, SaaS / recurring-revenue models,
  trademark ownership, business maturity. **What lowers it:** long time on the
  marketplace, eCommerce monetization.

---

## Final Scoreboard

Cross-validated (5-fold, fixed seed), GradientBoosting. Every row uses the same
folds, so differences are attributable to the feature set.

| Condition | log R2 | Mean error (multiple) |
|---|---|---|
| Baseline (predict mean) | — | 0.574 |
| v1 tabular | 0.400 | 0.418 |
| v1 + text (Qwen) | 0.447 | 0.399 |
| v2 tabular (+ trend) | 0.447 | 0.389 |
| **v2 + text (final)** | **0.482** | **0.377** |

---

## Problem Setup

| Item | Choice | Why |
|---|---|---|
| Target | `annual_listing_multiple` (listing multiple) | The sale multiple is **not** returned by the public API — verified by direct calls — so the listing multiple is the primary label. |
| Transform | `log1p(target)` | Target is right-skewed; log makes it near-symmetric. |
| Data | For Sale (183) + Sold (2,674) = 2,857 collected, **2,848 valid** after dropping 9 rows with a missing/non-positive target | Sold listings retain full financials/text, so they enlarge the training set even with the listing-multiple target. |
| Leakage guard | Drop `listing_price`, `unpriced`, `listing_multiple` | multiple ~= price / profit, so any price field leaks the answer. Enforced in code. |

---

## What the Experiments Showed

**Text contributes real signal.** Adding text embeddings lifted R2 from 0.40 to
0.45. A control that swapped the real embeddings for random noise of identical
shape *lowered* R2 to 0.38 — so the gain is from content, not column count.

**Embedding model: diminishing returns.** MiniLM-L6 (384-dim, ~90 MB, ~9 s) and
Qwen3-0.6B (1024-dim, ~1.2 GB, ~8 min) both helped; Qwen won by a hair
(0.447 vs 0.437 fused) at vastly higher cost. Bigger isn't freely better.

**PCA dimensions: an overfitting curve.** Sweeping the Qwen embedding through
PCA at 5/10/20/30/40/50/100/200/300 dims traced a clear inverted-U: too few
dims underfit, ~30-50 was best, and 100-300 *dropped* despite preserving more
variance (94.5% at 300). More retained variance did not mean better prediction.

**Growth trend beats traffic.** Of the unused fields, profit/revenue **trend**
was the single most useful addition (+0.05 R2, matching text). Traffic **volume**
added nothing — what a business earns matters, how many visitors it has doesn't.

**Trend and text are complementary.** Because trend is a quantitative growth
signal and text is a qualitative one, their gains stacked to the final 0.482.

---

## What Drives the Multiple (interpretation)

From SHAP on the best model:

**Pushes the multiple up:** higher annual net profit; SaaS / recurring revenue;
trademark ownership; older, more mature businesses.

**Pushes the multiple down:** more days on the marketplace (unsold listings get
discounted); eCommerce monetization.

**Error diagnostics:** error grows with business size (large businesses are
harder to price); SaaS has the highest per-segment error (high multiples, high
variance); residuals are roughly symmetric around zero.

---

## Pipeline

| Step | File | Runs on | What it does |
|---|---|---|---|
| 00 | `src/00_ingest.py` | local | Collect listings via the official API (1 req/sec, paginated, dated snapshots). |
| 01 | `notebooks/local/01_validate_schema.ipynb` | local | Field inventory, coverage, target check. |
| 02 | `src/02_clean_features.py` | local | Type casting, derived features, leakage guard, target log-transform. |
| 02b | `src/02b_clean_features_v2.py` | local | Adds growth-trend features (+ missing flags). |
| 03 | `notebooks/local/03_eda.ipynb` | local | Distributions, group differences, correlations. |
| 04 | `notebooks/colab/04_text_features.ipynb` | Colab (GPU) | MiniLM sentence embeddings + PCA. |
| 04 (qwen) | `notebooks/colab/04_text_features_qwen.ipynb` | Colab (GPU) | Qwen3-0.6B embeddings + PCA. |
| 04b | `notebooks/colab/04b_qwen_pca_sweep.ipynb` | Colab (GPU) | Qwen embeddings at several PCA dims (50-300). |
| 05 | `src/05_model_baseline.py` | local | Tabular-only baseline (Ridge + GradientBoosting). |
| 06 | `src/06_model_text_fusion.py` | local | Tabular + text; test whether text helps. |
| 06b | `src/06b_compare_embeddings.py` | local | MiniLM vs Qwen comparison. |
| 06c | `src/06c_compare_pca.py` | local | PCA-dimension sweep comparison. |
| 06d | `src/06d_final_comparison.py` | local | Final scoreboard across feature sets + text. |
| 07 | `notebooks/local/07_eval_interpret.ipynb` | local | Feature importance, SHAP, residuals, segment error. |

Heavy work (embeddings) runs in Colab on a free GPU; everything else runs locally.

---

## How to Reproduce

```bash
# 1. Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Collect + clean (local)
python src/00_ingest.py
python src/02_clean_features.py          # v1 features
python src/02b_clean_features_v2.py      # v2 features (with trend)

# 3. Text embeddings (Colab)
#    - Upload data/private/text_private_*.csv to Drive
#    - Run notebooks/colab/04b_qwen_pca_sweep.ipynb with a GPU runtime
#    - Download text_embeddings_qwen_pca*_*.csv into data/processed/

# 4. Model + compare (local)
python src/05_model_baseline.py
python src/06d_final_comparison.py       # the final scoreboard
#    then run notebooks/local/07_eval_interpret.ipynb
```

---

## Data & Ethics

- Data comes from the **official public API** — no scraping, no authentication,
  1 request/second. All timestamps are UTC.
- Listings can disappear from the API, so snapshots are dated and accumulated.
- **Raw listing text (opportunities / risks / summary) is kept private** and is
  never committed. Only derived numeric embeddings and aggregate results are shared.
- Not legal or financial advice; multiples here are model estimates, not appraisals.

## Repository Layout

```
workshop-2026/
├── src/                 # ingest, clean, model, comparison scripts (.py)
├── notebooks/
│   ├── local/           # schema, EDA, interpretation (.ipynb, VS Code)
│   └── colab/           # text embeddings (.ipynb, Colab GPU)
├── data/                # git-ignored (raw, processed, private)
├── outputs/             # git-ignored (model results)
├── index.html           # interactive report
├── requirements.txt
└── README.md
```

Data and outputs are git-ignored; the repository holds **code and results only**.
