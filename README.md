# Customer Churn & CLV Engine
**End-to-end machine learning system for churn prediction and customer lifetime value modelling on the Olist Brazilian E-Commerce dataset.**

---

## What This Project Does

This project answers two questions every e-commerce business needs to solve:

1. **Which customers are about to stop buying?** — XGBoost classifier with ROC-AUC 0.984
2. **How much future revenue will each customer generate?** — BG/NBD + Gamma-Gamma probabilistic CLV model

Results are served through an interactive Streamlit dashboard where business users can filter customers by risk tier, explore the CLV vs churn quadrant, and identify the highest-priority retention targets.

---

## Live Dashboard Preview

| KPI Cards | Churn Distribution | CLV vs Churn Quadrant |
|---|---|---|
| 93,357 customers scored | Risk tier breakdown | Retention targeting view |

> Run locally: `streamlit run app/streamlit_app.py`

---

## Key Technical Decisions (and why they matter)

### Churn label: BG/NBD P(alive), not a time window
Olist has a ~97% single-purchase rate. A naive "no purchase in 180 days" label flags nearly every customer as churned — making the ML problem degenerate. Instead, the **BG/NBD model** (Fader, Hardie & Lee 2005) computes P(alive) for each customer based on their individual purchase cadence. Customers with P(alive) < 50% are labelled churned. This is the industry-standard approach for non-contractual marketplaces.

### Feature design: behavioural signals only
XGBoost is trained exclusively on **behavioural features** — review scores, delivery times, category breadth, spend patterns, and payment behaviour. RFM inputs (recency, frequency, tenure) are intentionally excluded because they are the direct inputs to BG/NBD. Including them would let the model trivially replicate the probabilistic label, inflating AUC to 1.0 with no real insight.

### Leakage prevention caught twice
During development, AUC = 1.0 appeared twice — each time from a different leakage source. Both were identified and fixed. The final model achieves AUC = 0.984 on genuinely held-out data.

### Serialisation: JSON not pickle
The `lifetimes` library uses lambda closures internally, which break `joblib` serialisation. The CLV models are saved as JSON parameter dictionaries and reconstructed at inference time — a pattern that generalises to any library with non-picklable objects.

---

## Architecture

```
Raw CSVs (8 tables)
      │
      ▼
DuckDB  ◄──  dbt transformations (staging + mart layer)
      │
      ▼
Feature Engineering (SQL via DuckDB)
   ├── RFM features        → CLV models
   └── Behavioural features → Churn model
      │
      ▼
Training Pipeline (src/train.py)
   ├── BetaGeoFitter   (BG/NBD)        saved as JSON
   ├── GammaGammaFitter (Gamma-Gamma)  saved as JSON
   └── XGBClassifier + StandardScaler  saved as joblib Pipeline
      │
      ▼
Batch Predictions (src/predict.py)
   └── predictions.csv  (93,357 customers × 4 columns)
      │
      ▼
Streamlit Dashboard (app/streamlit_app.py)
```

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| Local warehouse | **DuckDB** | Zero-setup columnar OLAP; reads CSV natively; rivals cloud DW performance for this dataset size |
| Transformations | **dbt-duckdb** | SQL transformations with lineage, docs, and tests; same models can target Snowflake/BigQuery in production |
| CLV models | **lifetimes** | Reference implementation of Fader et al. BG/NBD and Gamma-Gamma papers |
| Churn model | **XGBoost** | Best-in-class for tabular data; native class imbalance handling via `scale_pos_weight` |
| Explainability | **SHAP** | Game-theory-based feature attribution; explains individual predictions, not just global importance |
| Dashboard | **Streamlit + Plotly** | Interactive charts (zoom/hover/filter) deployed with zero frontend code |
| Orchestration | **sklearn Pipeline** | Guarantees identical preprocessing at train and inference time |

---

## Model Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.984** |
| Average Precision | **0.811** |
| Total customers scored | 93,357 |
| High-Risk customers | 6,416 (6.9%) |
| Revenue at Risk | R$ 188,778 |
| Customers with CLV > 0 | 2,015 (2.2% — repeat buyers only) |

**Top feature by importance:** `n_distinct_categories` (60.7% gain) — customers who explore multiple product categories are significantly more likely to remain active.

---

## Project Structure

```
customer-churn-clv-engine/
├── app/
│   ├── streamlit_app.py          # Main dashboard
│   └── components/
│       ├── charts.py             # Plotly chart components
│       └── kpi_cards.py          # Top KPI metrics
├── dbt/
│   └── models/
│       ├── staging/              # Source cleaning
│       ├── intermediate/         # Customer-order aggregations
│       └── marts/                # RFM, churn labels, CLV features
├── src/
│   ├── ingest.py                 # CSV → DuckDB
│   ├── features.py               # RFM + behavioural SQL queries
│   ├── train.py                  # BG/NBD, Gamma-Gamma, XGBoost training
│   ├── evaluate.py               # ROC curve, SHAP, metrics export
│   └── predict.py                # Batch scoring → predictions.csv
├── notebooks/
│   └── 01_eda.ipynb              # Exploratory analysis
├── tests/                        # Unit tests
└── config.py                     # Central constants (paths, dates, hyperparams)
```

---

## Running Locally

**Prerequisites:** Python 3.11+, the Olist dataset CSVs placed in `data/raw/`

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest raw data into DuckDB
python src/ingest.py

# 3. Run dbt transformations
cd dbt && dbt run && cd ..

# 4. Train all models (~2 min)
python src/train.py

# 5. Generate predictions
python src/predict.py

# 6. Evaluate and save ROC / SHAP artefacts
python src/evaluate.py

# 7. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## What I Learned Building This

- **Domain knowledge changes everything.** A 97% single-purchase rate is not a data quality problem — it is the business reality of marketplaces. The modelling approach has to adapt to the domain.
- **AUC = 1.0 means something is wrong.** I hit this twice, from two different leakage sources. Diagnosing and fixing both sharpened my understanding of feature/label independence.
- **Library conventions matter.** The `lifetimes` library expects *repeat* purchase count (total − 1), not total count. This is buried in the paper, not the docs. Using raw counts silently corrupts every model parameter.
- **Separate scoring from serving.** Writing predictions to CSV and reading them in the dashboard decouples ML inference from the web layer — faster dashboards, simpler deployments, no ML dependencies at serve time.

---

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100k orders, 2016–2018, available on Kaggle under a CC BY-NC-SA 4.0 licence.
