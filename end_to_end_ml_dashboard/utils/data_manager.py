"""
utils/data_manager.py
=====================
Single module that handles:
  - Data loading  (CSV / Excel / sample)
  - Data cleaning (missing values, duplicates, outliers, encoding, feature engineering)
  - EDA helpers   (summary stats, correlation)
  - ML training   (classifiers, regressor, k-means)
  - Prediction    (single row, batch)
  - Forecasting   (ARIMA with linear fallback)
  - Insight generation (rule-based)
  - Streamlit session-state helpers
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_df() -> pd.DataFrame | None:
    return st.session_state.get("df")

def set_df(df: pd.DataFrame) -> None:
    st.session_state["df"] = df

def get_clean_df() -> pd.DataFrame | None:
    return st.session_state.get("clean_df")

def set_clean_df(df: pd.DataFrame) -> None:
    st.session_state["clean_df"] = df

def get_models() -> dict | None:
    return st.session_state.get("trained_models")

def set_models(models: dict) -> None:
    st.session_state["trained_models"] = models


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file from a Streamlit UploadedFile object.

    Supports: .csv, .xlsx, .xls

    Usage in your Streamlit page
    ----------------------------
    uploaded_file = st.file_uploader(
        "Upload your data", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file:
        try:
            df = load_uploaded_file(uploaded_file)
            set_df(df)
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        except ValueError as e:
            st.error(str(e))
    """
    name = uploaded_file.name.lower()

    # ── CSV ───────────────────────────────────────────────────────────────────
    if name.endswith(".csv"):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(
            "Could not decode the CSV file. "
            "Please save it as UTF-8 and try again."
        )

    # ── Modern Excel (.xlsx / .xlsm) ─────────────────────────────────────────
    elif name.endswith((".xlsx", ".xlsm")):
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            raise ValueError(
                f"Could not read the Excel file: {e}\n"
                "Make sure 'openpyxl' is installed:  pip install openpyxl"
            ) from e

    # ── Legacy Excel (.xls) ──────────────────────────────────────────────────
    elif name.endswith(".xls"):
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, engine="xlrd")
        except Exception as e:
            raise ValueError(
                f"Could not read the legacy .xls file: {e}\n"
                "Make sure 'xlrd' is installed:  pip install xlrd"
            ) from e

    else:
        raise ValueError(
            f"Unsupported file type: '{uploaded_file.name}'.\n"
            "Please upload a .csv, .xlsx, or .xls file."
        )


def generate_sample_data(n: int = 1200) -> pd.DataFrame:
    """
    Generate a synthetic dataset for demo purposes.
    """
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "date":          dates,
        "product":       np.random.choice(["Product A", "Product B", "Product C"], n),
        "region":        np.random.choice(["North", "South", "East", "West"], n),
        "units":         np.random.randint(5, 100, n),
        "revenue":       np.random.randint(100, 5000, n),
        "cost":          np.random.randint(50, 3000, n),
        "tenure_months": np.random.randint(1, 60, n),
        "churn":         np.random.choice([0, 1], n),
    })

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Auto-clean a DataFrame and return (cleaned_df, report_dict).

    Pipeline
    --------
    1. Drop fully-empty rows
    2. Impute missing values  (median for numeric, mode for categorical)
    3. Remove duplicate rows
    4. Cap outliers via IQR method
    5. Label-encode categorical columns  (adds *_enc columns)
    6. Feature-engineer date columns    (month, dayofweek, quarter)

    Returns
    -------
    cleaned_df : pd.DataFrame
    report     : dict with keys:
                   empty_rows_dropped, missing_values_filled, missing_by_column,
                   duplicates_removed, outliers_capped,
                   categorical_encoded, features_engineered, quality_score
    """
    from sklearn.preprocessing import LabelEncoder

    report = {}
    df = df.copy()

    # 1. Empty rows
    before = len(df)
    df.dropna(how="all", inplace=True)
    report["empty_rows_dropped"] = before - len(df)

    # 2. Missing values
    missing_before = df.isnull().sum().to_dict()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for c in num_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode()[0], inplace=True)

    report["missing_values_filled"] = int(
        sum(missing_before.values()) - df.isnull().sum().sum()
    )
    report["missing_by_column"] = {k: v for k, v in missing_before.items() if v > 0}

    # 3. Duplicates
    dups = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = dups

    # 4. Outliers (IQR cap)
    outlier_count = 0
    for c in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        outlier_count += int(mask.sum())
        df[c] = df[c].clip(lo, hi)
    report["outliers_capped"] = outlier_count

    # 5. Label-encode categoricals
    le = LabelEncoder()
    encoded = []
    for c in cat_cols:
        df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))
        encoded.append(c)
    report["categorical_encoded"] = encoded

    # 6. Date feature engineering
    date_cols = [c for c in df.columns if "date" in c.lower()]
    fe_cols   = []
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c])
            df[f"{c}_month"]     = df[c].dt.month
            df[f"{c}_dayofweek"] = df[c].dt.dayofweek
            df[f"{c}_quarter"]   = df[c].dt.quarter
            fe_cols += [f"{c}_month", f"{c}_dayofweek", f"{c}_quarter"]
        except Exception:
            pass
    report["features_engineered"] = fe_cols

    # Quality score
    total_issues = (
        report["missing_values_filled"]
        + report["duplicates_removed"]
        + report["outliers_capped"]
    )
    n_cells = len(df) * len(df.columns)
    report["quality_score"] = round(100 * (1 - total_issues / max(n_cells, 1)), 1)

    return df, report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDA HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T.reset_index().rename(columns={"index": "column"})

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).corr()

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_cat_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def get_date_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if "date" in c.lower() or str(df[c].dtype) == "datetime64[ns]"
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MACHINE LEARNING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def select_features(df: pd.DataFrame, target: str) -> list[str]:
    """Return numeric columns excluding the target and any *_enc shadow columns."""
    num = get_numeric_cols(df)
    return [c for c in num if c != target and not c.endswith("_enc")]


def train_all_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> dict:
    """
    Train Random Forest, Gradient Boosting, Logistic Regression, and
    Decision Tree on the given features and binary target.

    Returns
    -------
    dict keyed by model name, each value:
      { model, accuracy, f1, auc, cm, feature_importance }
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    CLASSIFIERS = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    }

    data = df[feature_cols + [target]].dropna()
    X, y = data[feature_cols].values, data[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = (
            pipe.predict_proba(X_te)[:, 1]
            if hasattr(pipe["clf"], "predict_proba") else None
        )

        fi    = None
        inner = pipe["clf"]
        if hasattr(inner, "feature_importances_"):
            fi = pd.Series(
                inner.feature_importances_, index=feature_cols
            ).sort_values(ascending=False)
        elif hasattr(inner, "coef_"):
            fi = pd.Series(
                np.abs(inner.coef_[0]), index=feature_cols
            ).sort_values(ascending=False)

        results[name] = {
            "model":              pipe,
            "accuracy":           round(accuracy_score(y_te, y_pred) * 100, 1),
            "f1":                 round(f1_score(y_te, y_pred, average="weighted") * 100, 1),
            "auc":                (
                round(roc_auc_score(y_te, y_prob) * 100, 1)
                if y_prob is not None else None
            ),
            "cm":                 confusion_matrix(y_te, y_pred),
            "feature_importance": fi,
        }

    return results


def train_regressor(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> dict:
    """Train a Linear Regression model and return metrics + pipeline."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    data = df[feature_cols + [target]].dropna()
    X, y = data[feature_cols].values, data[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    return {
        "model": pipe,
        "rmse":  round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 2),
        "r2":    round(r2_score(y_te, y_pred) * 100, 1),
    }


def run_kmeans(df: pd.DataFrame, feature_cols: list[str], k: int = 4) -> pd.DataFrame:
    """Cluster the dataset and return a copy with a 'cluster' column."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    data    = df[feature_cols].dropna()
    X_scaled = StandardScaler().fit_transform(data.values)
    labels  = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    out     = df.copy()
    out["cluster"] = np.nan
    out.loc[data.index, "cluster"] = labels.astype(int)
    return out


def get_best_model_name(results: dict) -> str:
    return max(results, key=lambda k: results[k]["accuracy"])


def predict_single(pipeline, feature_values: list[float]) -> tuple[int, float | None]:
    """
    Run a single-row prediction.

    Returns
    -------
    (predicted_class, probability_of_class_1)
    """
    X    = np.array(feature_values).reshape(1, -1)
    pred = int(pipeline.predict(X)[0])
    prob = (
        float(pipeline.predict_proba(X)[0, 1])
        if hasattr(pipeline["clf"], "predict_proba") else None
    )
    return pred, prob


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORECASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _linear_forecast(series: pd.Series, steps: int):
    y      = series.values
    x      = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    trend  = np.poly1d(coeffs)
    x_fut  = np.arange(len(y), len(y) + steps)
    fc     = trend(x_fut)
    std    = np.std(y - trend(x))
    return fc, fc - 1.96 * std, fc + 1.96 * std


def _arima_forecast(series: pd.Series, steps: int, order=(1, 1, 1)):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model  = ARIMA(series, order=order).fit()
        fc_obj = model.get_forecast(steps=steps)
        fc     = fc_obj.predicted_mean.values
        ci     = fc_obj.conf_int(alpha=0.05).values
        return fc, ci[:, 0], ci[:, 1]
    except Exception:
        return None


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "MS",
    steps: int = 12,
    method: str = "auto",
) -> dict:
    """
    Aggregate df to a regular time series and forecast ahead.

    Parameters
    ----------
    df        : DataFrame (cleaned preferred)
    date_col  : name of date/datetime column
    value_col : numeric column to aggregate and forecast
    freq      : pandas offset alias  ('MS' = month-start)
    steps     : number of future periods
    method    : 'auto' | 'arima' | 'linear'

    Returns
    -------
    dict with keys:
      historical_dates, historical_values,
      forecast_dates, forecast_values, lower, upper,
      mape, method_used
    OR  {'error': str}  on failure
    """
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    ts = (
        tmp.groupby(pd.Grouper(key=date_col, freq=freq))[value_col]
           .sum()
           .dropna()
    )

    if len(ts) < 6:
        return {"error": "Need at least 6 data points for forecasting."}

    future_dates = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]

    fc = lo = hi = None
    method_used = "Linear trend"

    if method in ("arima", "auto"):
        result = _arima_forecast(ts, steps)
        if result is not None:
            fc, lo, hi = result
            method_used = "ARIMA"

    if fc is None:
        fc, lo, hi = _linear_forecast(ts, steps)

    # Pseudo-holdout MAPE
    h        = max(2, len(ts) // 5)
    val_pred, _, _ = _linear_forecast(ts.iloc[:-h], h)
    val_true = ts.iloc[-h:].values
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((val_true - val_pred) / val_true)) * 100)
    mape = round(mape, 1) if not np.isnan(mape) else None

    return {
        "historical_dates":  ts.index.tolist(),
        "historical_values": ts.values.tolist(),
        "forecast_dates":    future_dates.tolist(),
        "forecast_values":   fc.tolist(),
        "lower":             lo.tolist(),
        "upper":             hi.tolist(),
        "mape":              mape,
        "method_used":       method_used,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSIGHT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITY_COLOR = {"high": "🔴", "medium": "🟡", "low": "🟢"}
CATEGORY_ICON  = {"risk": "⚠️", "opportunity": "💡", "trend": "📈", "info": "ℹ️"}
BORDER_COLOR   = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}


def generate_insights(df: pd.DataFrame, target: str | None = None) -> list[dict]:
    """
    Analyse a cleaned DataFrame and return a list of insight dicts:
      { title, description, priority, category }
    """
    insights = []
    num_cols = get_numeric_cols(df)
    rev_cols = [c for c in num_cols if "revenue" in c.lower() or "sales" in c.lower()]

    # ── Skewness ──────────────────────────────────────────────────────────────
    for col in num_cols:
        sk = df[col].skew()
        if abs(sk) > 2:
            direction = "right-skewed (long upper tail)" if sk > 0 else "left-skewed"
            insights.append({
                "title":       f"High skewness in '{col}'",
                "description": (
                    f"'{col}' is {direction} (skew = {sk:.1f}). "
                    "Log-transforming this feature may improve model performance."
                ),
                "priority": "medium", "category": "info",
            })

    # ── Pareto / revenue concentration ────────────────────────────────────────
    for rc in rev_cols[:1]:
        sv  = df[rc].sort_values(ascending=False)
        top = sv.iloc[: max(1, len(sv) // 5)].sum() / sv.sum()
        if top > 0.6:
            insights.append({
                "title":       f"Revenue concentration risk — '{rc}'",
                "description": (
                    f"Top 20% of rows account for {top:.0%} of total {rc}. "
                    "Losing just a few top accounts would materially impact revenue."
                ),
                "priority": "high", "category": "risk",
            })

    # ── Target / churn analysis ───────────────────────────────────────────────
    if target and target in df.columns:
        rate = df[target].mean()
        if rate > 0.2:
            insights.append({
                "title":       f"Elevated positive rate for '{target}' ({rate:.1%})",
                "description": (
                    f"A rate of {rate:.1%} is above the typical 5–15% industry benchmark. "
                    "Review top ML feature importances to understand key drivers."
                ),
                "priority": "high", "category": "risk",
            })

        corr  = df[num_cols].corr()[target].drop(target, errors="ignore")
        top_d = corr.abs().idxmax()
        if not pd.isna(top_d):
            insights.append({
                "title":       f"Strongest predictor of '{target}': '{top_d}'",
                "description": (
                    f"'{top_d}' has the highest correlation with '{target}' "
                    f"(r = {corr[top_d]:.2f}). Optimise this variable first."
                ),
                "priority": "high", "category": "opportunity",
            })

    # ── Segment performance ───────────────────────────────────────────────────
    cat_cols = get_cat_cols(df)
    for cc in cat_cols[:2]:
        for rc in rev_cols[:1]:
            grp   = df.groupby(cc)[rc].mean()
            best  = grp.idxmax()
            worst = grp.idxmin()
            ratio = grp[best] / grp[worst] if grp[worst] > 0 else np.nan
            if not np.isnan(ratio) and ratio > 1.5:
                insights.append({
                    "title":       f"'{best}' outperforms '{worst}' in {rc}",
                    "description": (
                        f"Within '{cc}', '{best}' generates {ratio:.1f}× the avg {rc} "
                        f"of '{worst}'. Consider reallocating resources to high-value segments."
                    ),
                    "priority": "medium", "category": "opportunity",
                })

    # ── Trend (first-half vs second-half) ─────────────────────────────────────
    for rc in rev_cols[:1]:
        vals = df[rc].values
        if len(vals) > 20:
            fh  = vals[: len(vals) // 2].mean()
            sh  = vals[len(vals) // 2 :].mean()
            pct = (sh - fh) / fh * 100 if fh else 0
            if abs(pct) > 10:
                direction = "increased" if pct > 0 else "decreased"
                insights.append({
                    "title":       f"{rc.title()} {direction} {abs(pct):.1f}% (first vs second half)",
                    "description": (
                        f"Average {rc} {direction} by {abs(pct):.1f}% between the "
                        "two halves of the dataset. Investigate seasonal, operational, "
                        "or market-driven causes."
                    ),
                    "priority": "medium", "category": "trend",
                })

    if not insights:
        insights.append({
            "title":       "Dataset looks healthy",
            "description": "No major anomalies or concentration risks detected.",
            "priority":    "low",
            "category":    "info",
        })

    return insights
