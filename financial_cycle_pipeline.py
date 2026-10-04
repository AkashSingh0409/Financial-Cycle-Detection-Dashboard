# %% [markdown]
# # Financial Cycle Detection Using Data Analytics
# End-to-end pipeline: BIS-style data -> cleaning -> EDA -> PCA -> cycle labelling
# + t-test -> moving average / volatility -> ML models -> validation.
#
# Run in Jupyter / VS Code (each `# %%` is a cell) or as a script:
#     python financial_cycle_pipeline.py
#
# Requirements: pip install pandas numpy scipy scikit-learn matplotlib seaborn

# %% [markdown]
# ## 0. Setup & configuration

# %%
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# ---- Configuration ---------------------------------------------------------
# >>> TO USE YOUR OWN REAL BIS DATA <<<
# 1. Download a CSV from data.bis.org (long format: one row per country-quarter).
# 2. Set REAL_DATA_PATH to the file path, e.g. "bis_export.csv".
# 3. If your column names differ, edit COLUMN_MAP (BIS name -> required name).
#    Required columns: BORROWERS_CTY, Quarter, Credit_Value, CPI_Value,
#    Credit_to_GDP_Gap.  Quarter can look like "2000-Q1" or "2000-03-31".
# If your file is in BIS *wide* format (one column per quarter), call
# `wide_to_long()` below first.
REAL_DATA_PATH: Optional[str] = None   # e.g. "bis_export.csv"
COLUMN_MAP = {
    # "Reference area": "BORROWERS_CTY",
    # "TIME_PERIOD": "Quarter",
    # "Total credit": "Credit_Value",
    # "CPI": "CPI_Value",
    # "Gap": "Credit_to_GDP_Gap",
}
REQUIRED_COLS = ["BORROWERS_CTY", "Quarter", "Credit_Value", "CPI_Value", "Credit_to_GDP_Gap"]
NUMERIC_COLS = ["Credit_Value", "CPI_Value", "Credit_to_GDP_Gap"]
BOOM_THRESHOLD, CONTRACTION_THRESHOLD = 10.0, -10.0   # Basel III, percentage points

# %% [markdown]
# ## 1. Data sourcing (synthetic BIS-like data or your real CSV)

# %%
def generate_synthetic_bis_data(
    countries=("United States", "Japan", "India"),
    start="2000Q1",
    end="2025Q4",
    seed: int = RANDOM_STATE,
    inject_issues: bool = True,
) -> pd.DataFrame:
    """Create a realistic synthetic panel mimicking the BIS structure.

    Credit-to-GDP gap = slow sinusoidal financial cycle + AR(1) noise (large enough
    to cross the +/-10 pp Basel III thresholds). Credit growth is linked to the gap, so
    the Boom/Stable/Contraction labels correspond to genuinely different growth regimes.

    If `inject_issues` is True, duplicates, missing values and string placeholders
    ('..', 'NA') are injected so the cleaning phase has real work to do.
    """
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start, end, freq="Q")
    n = len(periods)
    t = np.arange(n)
    # (base ratio %, GDP growth/qtr, inflation/qtr, cycle amplitude, cycle length qtrs, phase)
    params = {
        "United States": (150, 0.012, 0.006, 14, 60, 0.0),
        "Japan":         (170, 0.004, 0.001, 16, 70, 1.5),
        "India":         (50,  0.017, 0.015, 13, 50, 3.0),
    }
    frames = []
    for c in countries:
        base, g, infl, amp, length, ph = params.get(
            c, (100, 0.01, 0.007, 12, 60, rng.uniform(0, 6)))
        noise = np.zeros(n)
        for i in range(1, n):                       # AR(1) noise
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 1.5)
        gap = amp * np.sin(2 * np.pi * t / length + ph) + noise
        ratio = base + 0.05 * t + gap
        gdp = 100 * np.cumprod(1 + g + rng.normal(0, 0.004, n))
        # Credit growth = GDP growth + a cycle-driven term (credit expands faster when the
        # gap is high, slower when it is low) + noise. The gap is a stylised cycle indicator.
        credit_growth = g + 0.0015 * gap + rng.normal(0, 0.004, n)
        credit = 1000 * base / 100 * np.cumprod(1 + credit_growth)
        cpi = 60 * np.cumprod(1 + infl + rng.normal(0, 0.002, n))
        frames.append(pd.DataFrame({
            "BORROWERS_CTY": c,
            "Quarter": [str(p) for p in periods],        # e.g. '2000Q1'
            "Credit_Value": credit.round(2),
            "CPI_Value": cpi.round(2),
            "Credit_to_GDP_Gap": gap.round(2),
            "Credit_to_GDP_Ratio": ratio.round(2),       # extra, used by dashboard
        }))
    df = pd.concat(frames, ignore_index=True)

    if inject_issues:
        df["Credit_Value"] = df["Credit_Value"].astype(object)
        for col in ["Credit_Value", "CPI_Value", "Credit_to_GDP_Gap"]:
            idx = rng.choice(df.index[5:], size=int(0.04 * len(df)), replace=False)
            df[col] = df[col].astype(object)
            df.loc[idx, col] = rng.choice([np.nan, "..", "NA"], size=len(idx))
        df = pd.concat([df, df.sample(8, random_state=seed)], ignore_index=True)  # duplicates
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)          # shuffle
    return df


def wide_to_long(df_wide: pd.DataFrame, id_cols: list, value_name: str) -> pd.DataFrame:
    """Convert BIS wide format (one column per quarter) to long (country-quarter) format."""
    time_cols = [c for c in df_wide.columns if c not in id_cols]
    return df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                        var_name="Quarter", value_name=value_name)


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load a real BIS CSV if `path` is given, otherwise synthetic data."""
    if path is None:
        print("[INFO] No REAL_DATA_PATH set -> using synthetic BIS-like data.")
        return generate_synthetic_bis_data()
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not read '{path}': {exc}") from exc
    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Found: {list(df.columns)}. "
            "Edit COLUMN_MAP to map your CSV headers.")
    print(f"[INFO] Loaded real data from {path}: {df.shape}")
    return df


raw_df = load_data(REAL_DATA_PATH)
print(raw_df.head(), "\n", raw_df.shape)

# %% [markdown]
# ## Phase 1: Data sourcing & cleaning

# %%
def parse_quarter(s: pd.Series) -> pd.Series:
    """Parse '2000Q1', '2000-Q1' or '2000-03-31' into quarter-end Timestamps."""
    cleaned = s.astype(str).str.strip().str.replace("-Q", "Q", regex=False)
    try:
        return pd.PeriodIndex(cleaned, freq="Q").to_timestamp(how="end").normalize()
    except Exception:
        return pd.PeriodIndex(pd.to_datetime(s), freq="Q").to_timestamp(how="end").normalize()


def clean_data(df: pd.DataFrame):
    """Remove duplicates, coerce to numeric, ffill/bfill within each country.

    Returns (clean_df, summary_df) where summary_df is the before/after report.
    NOTE: fill is applied *within each country* so one economy's values never leak
    into another's.
    """
    df = df.copy()
    df = df.replace(["..", "NA", "N/A", "-", ""], np.nan)
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    for col in NUMERIC_COLS + [c for c in ["Credit_to_GDP_Ratio"] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")   # strings -> NaN
    df["Date"] = parse_quarter(df["Quarter"])
    df = df.sort_values(["BORROWERS_CTY", "Date"]).reset_index(drop=True)

    missing_before = int(df.isna().sum().sum())
    num_cols = df.select_dtypes("number").columns.tolist()
    df[num_cols] = df.groupby("BORROWERS_CTY")[num_cols].transform(lambda s: s.ffill().bfill())
    # Safety net for entire-country-empty columns
    df[num_cols] = df[num_cols].ffill().bfill()
    missing_after = int(df.isna().sum().sum())

    reduction = 100 * (missing_before - missing_after) / missing_before if missing_before else 100.0
    summary = pd.DataFrame({
        "Metric": ["Duplicate rows removed", "Missing values BEFORE fill",
                   "Missing values AFTER fill", "Reduction (%)"],
        "Value": [n_dupes, missing_before, missing_after, round(reduction, 2)],
    })
    return df, summary


df, cleaning_summary = clean_data(raw_df)
print(cleaning_summary.to_string(index=False))
print("\nMissing per column BEFORE (after type coercion):")
print(raw_df.replace(["..", "NA", "N/A", "-", ""], np.nan).isna().sum())
print("\nMissing per column AFTER:")
print(df.isna().sum())

# Before/after bar chart
before = int(raw_df.replace(["..", "NA", "N/A", "-", ""], np.nan).isna().sum().sum())
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(["Before", "After"], [before, int(df.isna().sum().sum())], color=["tab:red", "tab:green"])
ax.set_title("Missing values before vs after cleaning"); ax.set_ylabel("Count")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 2: Exploratory data analysis

# %%
# Credit growth (QoQ %) per country -- needed for EDA and later phases
df["Credit_Growth"] = df.groupby("BORROWERS_CTY")["Credit_Value"].pct_change(fill_method=None) * 100

# 5-point summary (min, Q1, median, Q3, max) + mean/std
def five_point_summary(frame: pd.DataFrame, cols) -> pd.DataFrame:
    s = frame[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    return s[["min", "25%", "50%", "75%", "max", "mean", "std"]].round(2)

print(five_point_summary(df, NUMERIC_COLS + ["Credit_Growth"]))
print("\nBy country:")
print(df.groupby("BORROWERS_CTY")[["Credit_Value", "CPI_Value", "Credit_Growth"]]
        .describe(percentiles=[0.25, 0.5, 0.75]).T.round(2).to_string())

# %%
# Quarterly CPI correlation heatmap: countries x CPI series (pivot -> correlate across time)
cpi_wide = df.pivot_table(index="Date", columns="BORROWERS_CTY", values="CPI_Value")
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cpi_wide.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Quarterly CPI correlation across economies"); plt.tight_layout(); plt.show()

# Credit growth distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df.dropna(subset=["Credit_Growth"]), x="Credit_Growth",
             hue="BORROWERS_CTY", bins=40, kde=True, element="step")
plt.title("Distribution of quarterly credit growth (%)")
plt.xlabel("QoQ credit growth (%)"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 3: Feature engineering & dimensionality reduction (PCA)

# %%
def build_pca_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a correlated numeric feature set (levels, growth, lags, rolling stats)."""
    f = frame.copy()
    g = f.groupby("BORROWERS_CTY")
    f["CPI_Growth"] = g["CPI_Value"].pct_change(fill_method=None) * 100
    for lag in (1, 2, 4):
        f[f"Credit_lag{lag}"] = g["Credit_Value"].shift(lag)
        f[f"CPI_lag{lag}"] = g["CPI_Value"].shift(lag)
    f["Gap_lag1"] = g["Credit_to_GDP_Gap"].shift(1)
    f["Credit_MA4"] = g["Credit_Value"].transform(lambda s: s.rolling(4).mean())
    f["CPI_MA4"] = g["CPI_Value"].transform(lambda s: s.rolling(4).mean())
    cols = (["Credit_Value", "CPI_Value", "Credit_to_GDP_Gap", "Credit_Growth", "CPI_Growth",
             "Gap_lag1", "Credit_MA4", "CPI_MA4"]
            + [f"Credit_lag{l}" for l in (1, 2, 4)] + [f"CPI_lag{l}" for l in (1, 2, 4)])
    return f[cols].dropna()


X_feat = build_pca_features(df)
X_scaled = StandardScaler().fit_transform(X_feat)           # mean 0, std 1
pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_85 = int(np.argmax(cum_var >= 0.85) + 1)
X_pca = PCA(n_components=n_comp_85, random_state=RANDOM_STATE).fit_transform(X_scaled)

print(f"Original features: {X_feat.shape[1]}  ->  PCA components for >=85% variance: {n_comp_85}")
print("Cumulative explained variance:", np.round(cum_var[:n_comp_85 + 2], 3))

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
plt.axhline(0.85, color="red", ls="--", label="85% variance")
plt.xlabel("Number of principal components"); plt.ylabel("Cumulative explained variance")
plt.title("PCA - cumulative explained variance"); plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 4: Financial-cycle labelling & hypothesis testing

# %%
def label_cycles(frame: pd.DataFrame) -> pd.DataFrame:
    """Label each country-quarter via the Basel III +/-10 pp credit-to-GDP gap rule.

    Boom: gap > +10 | Stable: -10..+10 | Contraction: gap < -10
    (pd.cut with right=True puts exactly -10 in 'Contraction'; immaterial in practice.)
    """
    f = frame.copy()
    f["Credit_Growth"] = f.groupby("BORROWERS_CTY")["Credit_Value"].pct_change(fill_method=None) * 100
    f["Cycle"] = pd.cut(f["Credit_to_GDP_Gap"],
                        bins=[-np.inf, CONTRACTION_THRESHOLD, BOOM_THRESHOLD, np.inf],
                        labels=["Contraction", "Stable", "Boom"])
    return f


def run_ttest(frame: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Independent-samples t-test of credit growth: Boom vs Stable.

    H0: mean credit growth is equal in Boom and Stable phases.
    H1: the means differ.  Welch's version (equal_var=False) is used because the
    group variances are unlikely to be equal.
    """
    boom = frame.loc[frame.Cycle == "Boom", "Credit_Growth"].dropna()
    stable = frame.loc[frame.Cycle == "Stable", "Credit_Growth"].dropna()
    if len(boom) < 2 or len(stable) < 2:
        raise ValueError(f"Not enough observations (Boom={len(boom)}, Stable={len(stable)}).")
    t, p = ttest_ind(boom, stable, equal_var=False)
    return {"n_boom": len(boom), "n_stable": len(stable),
            "mean_boom": boom.mean(), "mean_stable": stable.mean(),
            "t_stat": t, "p_value": p,
            "decision": "Reject H0" if p < alpha else "Fail to reject H0"}


df = label_cycles(df)
print(df["Cycle"].value_counts())
print(pd.crosstab(df["BORROWERS_CTY"], df["Cycle"]))

res = run_ttest(df)
print("\n--- Independent-samples t-test (Boom vs Stable credit growth) ---")
for k, v in res.items():
    print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")

# Visual: gap with phases
fig, axes = plt.subplots(df.BORROWERS_CTY.nunique(), 1, figsize=(10, 7), sharex=True)
for ax, (c, g) in zip(np.atleast_1d(axes), df.groupby("BORROWERS_CTY")):
    ax.plot(g["Date"], g["Credit_to_GDP_Gap"], color="navy", lw=1.2)
    ax.axhline(BOOM_THRESHOLD, color="red", ls="--"); ax.axhline(CONTRACTION_THRESHOLD, color="green", ls="--")
    ax.fill_between(g["Date"], BOOM_THRESHOLD, g["Credit_to_GDP_Gap"],
                    where=g["Credit_to_GDP_Gap"] > BOOM_THRESHOLD, color="red", alpha=0.25)
    ax.fill_between(g["Date"], CONTRACTION_THRESHOLD, g["Credit_to_GDP_Gap"],
                    where=g["Credit_to_GDP_Gap"] < CONTRACTION_THRESHOLD, color="green", alpha=0.25)
    ax.set_title(f"{c}: credit-to-GDP gap (pp)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 5: Contemporary statistics (moving average & rolling volatility)

# %%
WINDOW = 4
g = df.groupby("BORROWERS_CTY")["Credit_Growth"]
df["Growth_MA4"] = g.transform(lambda s: s.rolling(WINDOW).mean())
df["Growth_RollStd4"] = g.transform(lambda s: s.rolling(WINDOW).std())

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for c, grp in df.groupby("BORROWERS_CTY"):
    axes[0].plot(grp["Date"], grp["Growth_MA4"], label=c)
    axes[1].plot(grp["Date"], grp["Growth_RollStd4"], label=c)
axes[0].set_title("4-quarter moving average of credit growth (%)")
axes[1].set_title("4-quarter rolling standard deviation of credit growth (volatility)")
axes[0].legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 6: ML predictive modelling (Linear Regression vs Decision Tree)

# %%
def build_model_frame(frame: pd.DataFrame, use_lags: bool = True) -> pd.DataFrame:
    """Target = NEXT quarter's credit level (per country). Features use only info
    available at time t (no leakage). Set use_lags=False for a time-index-only baseline."""
    m = frame.copy()
    g = m.groupby("BORROWERS_CTY")
    m["Target_NextCredit"] = g["Credit_Value"].shift(-1)
    m["Time_Index"] = (m["Date"].dt.year - m["Date"].dt.year.min()) * 4 + m["Date"].dt.quarter
    m["Country_Code"] = m["BORROWERS_CTY"].astype("category").cat.codes
    if use_lags:
        m["Credit_lag1"] = g["Credit_Value"].shift(1)
    m["Year"], m["Quarter_Num"], m["Month"] = m["Date"].dt.year, m["Date"].dt.quarter, m["Date"].dt.month
    return m.dropna(subset=["Target_NextCredit", "Credit_Value"] + (["Credit_lag1"] if use_lags else []))


def chronological_split(m: pd.DataFrame, test_size: float = 0.2):
    """80/20 split by TIME (last 20% of dates -> test), the correct choice for time series."""
    cutoff = m["Date"].quantile(1 - test_size)
    return m[m["Date"] <= cutoff], m[m["Date"] > cutoff]


MODEL_FEATURES = ["Time_Index", "Country_Code", "Credit_Value", "CPI_Value", "Credit_to_GDP_Gap", "Credit_lag1"]
model_df = build_model_frame(df, use_lags=True)
train, test = chronological_split(model_df)
X_train, y_train = train[MODEL_FEATURES], train["Target_NextCredit"]
X_test, y_test = test[MODEL_FEATURES], test["Target_NextCredit"]
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

models = {"Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)}
rows = []
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_test)
    rows.append({"Model": name,
                 "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                 "R2": r2_score(y_test, pred)})
results = pd.DataFrame(rows).round(3)
print("\n", results.to_string(index=False))

plt.figure(figsize=(8, 4))
plt.scatter(test["Date"], y_test, s=8, c="black", label="Actual")
for name, mdl in models.items():
    plt.scatter(test["Date"], mdl.predict(X_test), s=8, alpha=0.6, label=name)
plt.legend(); plt.title("Next-quarter credit: actual vs predicted (test set)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 7: Validation (5-fold CV) & feature importance

# %%
RF_FEATURES = ["Year", "Quarter_Num", "Month", "Country_Code", "CPI_Value", "Credit_to_GDP_Gap", "Credit_Value"]
rf_df = model_df.sort_values("Date")
X_rf, y_rf = rf_df[RF_FEATURES], rf_df["Target_NextCredit"]

rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
# TimeSeriesSplit = 5 expanding-window folds that never train on the future.
# (For plain shuffled folds use cv=5 instead -- but this can overstate performance on time series.)
cv_scores = cross_val_score(rf, X_rf, y_rf, cv=TimeSeriesSplit(n_splits=5), scoring="r2")
print("R2 per fold:", np.round(cv_scores, 3))
print(f"Mean CV R2: {cv_scores.mean():.3f}  (std {cv_scores.std():.3f})")

rf.fit(X_rf, y_rf)
importance = (pd.DataFrame({"Feature": RF_FEATURES, "Importance": rf.feature_importances_})
              .sort_values("Importance", ascending=False))
print(importance.round(4).to_string(index=False))

plt.figure(figsize=(8, 4))
sns.barplot(data=importance, x="Importance", y="Feature", color="steelblue")
plt.title("Random Forest feature importance"); plt.tight_layout(); plt.show()
print("\nPipeline complete.")
