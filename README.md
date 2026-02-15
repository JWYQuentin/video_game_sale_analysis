# Video Game Sales Analysis

**Analyzing 16,000+ video games to predict commercial hits, uncover regional market segments, and forecast regional sales distributions.**

This project applies a full data science pipeline — from exploratory analysis through supervised and unsupervised modeling — to a dataset of global video game sales spanning 1980–2016. The goal is to extract actionable insights about what drives commercial success in the gaming industry and how regional markets differ.

---

## Key Findings

- **Hit prediction is feasible with pre-release data alone.** Using only genre, publisher history, and platform characteristics, an XGBoost classifier achieves **0.85 AUC-ROC** and **0.62 F1-score** on identifying commercial hits — with publisher reputation (tier and historical hit rate) as the dominant predictor.

- **The global gaming market is fundamentally split into three segments:** NA-dominant titles (Sports/Shooters, 69% NA share), globally balanced Western hits (Action, 52% NA / 34% EU), and Japan-centric games (RPGs, 91% JP share). This was confirmed through K-Means clustering (k=3).

- **Japan is the most predictable market.** Regional share prediction models achieve R² = 0.44 for Japan vs. only 0.06 for North America — because Japan's preferences are highly systematic and genre-driven, while NA is the default market with high variance.

- **Platform choice is the single strongest signal for regional distribution.** Whether a game is on PC (strong EU signal) or a handheld console (strong JP signal) explains more regional variation than any other feature.

---

## Project Structure

```
video_game_sale_analysis/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── vgsales.csv
│   └── processed/                    # Cleaned & feature-engineered data
│       ├── vgsales_cleaned.csv
│       └── vgsales_featured.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Data cleaning & preprocessing
│   ├── 03_eda.ipynb                  # Exploratory data analysis (10 figures)
│   ├── 04_feature_engineering.ipynb  # 16 engineered features
│   ├── 05_classification.ipynb       # Phase 1: Hit prediction
│   ├── 06_clustering.ipynb           # Phase 2: Market segmentation
│   └── 07_regional_prediction.ipynb  # Phase 3: Regional share prediction
├── figures/                          # Saved EDA visualizations
├── model_data/                       # Train/test splits & scalers
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** VGChartz (via Kaggle)
- **Size:** 16,326 games after cleaning
- **Time span:** 1980–2016
- **Features:** Game name, platform, year, genre, publisher, and regional sales (NA, EU, JP, Other, Global)

---

## Methodology

### Data Cleaning & EDA
Cleaned raw data (handled missing values, standardized formats) and produced 10 publication-quality figures exploring sales distributions, market evolution over time, platform wars, genre trends, regional preferences, publisher dominance, and hit rates.

### Feature Engineering (16 New Features)
All features designed with **strict leakage prevention** — historical features use only past data via expanding windows with shift(1):

| Category | Features | Rationale |
|----------|----------|-----------|
| Publisher History | `Publisher_Hist_Hit_Rate`, `Publisher_Hist_Avg_Sales`, `Publisher_Experience` | Track record is the strongest predictor of future success |
| Market Context | `Genre_Trend`, `Market_Size_Prev_Year`, `Years_From_Peak` | Captures industry boom/bust cycles |
| Platform | `Platform_Age`, `Platform_Hist_Avg_Sales`, `Platform_Year_Competition` | Platform lifecycle and competitive density |
| Genre-Region | `Genre_NA_Affinity`, `Genre_EU_Affinity`, `Genre_JP_Affinity`, `Genre_Other_Affinity` | Historical regional preferences by genre |
| Derived | `Is_Sequel`, `Publisher_Genre_Spec`, `Decade_Num` | Title-based and interaction features |

**Train/test split:** Time-based (pre-2012 / 2012–2016) rather than random, simulating real-world prediction.

### Phase 1: Classification — Hit Prediction
Predicting whether a game will be a commercial hit (top 25% of sales).

| Model | AUC-ROC | F1-Score |
|-------|---------|----------|
| Logistic Regression | 0.845 | 0.586 |
| Random Forest | 0.853 | 0.621 |
| XGBoost | 0.834 | 0.568 |
| **XGBoost (Tuned)** | **0.845** | **0.625** |

Top predictive features: `Publisher_Tier_Indie` (strongest negative signal), `Publisher_Hist_Avg_Sales`, `Platform_Gen_Gen 7`, `Publisher_Hist_Hit_Rate`.

### Phase 2: Clustering — Market Segmentation
K-Means clustering (k=3) revealed three distinct market segments:

| Cluster | Games | NA Share | EU Share | JP Share | Top Genre | Hit Rate |
|---------|-------|----------|----------|----------|-----------|----------|
| NA-Dominant | 3,717 | 69.2% | 18.9% | 4.0% | Sports | 36.6% |
| Global Western | 5,265 | 52.2% | 33.5% | 4.1% | Action | 44.5% |
| Japan-Centric | 1,667 | 4.7% | 2.3% | 90.8% | Role-Playing | 23.4% |

### Phase 3: Regional Sales Share Prediction
Multi-output regression predicting what percentage of sales come from each region.

| Model | NA MAE | EU MAE | JP MAE | Overall MAE | Overall R² |
|-------|--------|--------|--------|-------------|------------|
| Baseline (Genre Avg) | 30.23 | 24.47 | 34.17 | 29.63 | -0.191 |
| Linear Regression | 25.39 | 22.62 | 31.80 | 26.60 | 0.040 |
| Random Forest | 23.79 | 19.52 | 24.70 | 22.67 | 0.177 |
| **XGBoost** | **22.03** | **19.31** | **23.33** | **21.56** | **0.255** |

Japan is the most predictable region (R² = 0.44) due to its systematic, genre-driven preferences. North America is the least predictable (R² = 0.06) because of its high variance as the default global market.

---

## How to Reproduce

```bash
# Clone the repository
git clone https://github.com/JWYQuentin/video_game_sale_analysis.git
cd video_game_sale_analysis

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (01 → 03 → 04 → 05 → 06 → 07)
```

---

## Future Improvements

- **Constrain regional predictions to sum to 100%** using compositional models (Dirichlet regression or softmax output)
- **Add cluster labels from Phase 2 as features** in the regional prediction model to connect the phases
- **Reframe regional prediction as classification** (dominant region) to handle the bimodal distribution better
- **Incorporate external data** such as review scores, marketing spend, and social media buzz to improve prediction accuracy

---

## Tools & Libraries

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
