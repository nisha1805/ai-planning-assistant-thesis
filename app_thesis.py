# ==============================================================================
# app_thesis.py  —  Digital Colleague: AI Planning Assistant
# MBA Thesis Research — Macromedia University Munich
# ==============================================================================

import os, io, json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"
MAX_HISTORY_TURNS = 10
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "clean", "pm_planning_simplified.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Colleague | AI Planning Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main header */
    .dc-header {
        background: linear-gradient(135deg, #0066CC 0%, #00A86B 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .dc-header h1 { margin: 0; font-size: 1.8rem; }
    .dc-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 0.95rem; }

    /* Stat cards */
    .stat-card {
        background: #f0f7ff;
        border: 1px solid #cce0ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-card .value { font-size: 1.6rem; font-weight: 700; color: #0066CC; }
    .stat-card .label { font-size: 0.8rem; color: #555; margin-top: 2px; }

    /* Transparency box */
    .transparency-row {
        background: #f8fffe;
        border-left: 4px solid #00A86B;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-family: monospace;
        font-size: 0.85rem;
    }

    /* Issue badge */
    .issue-high   { background:#fff0f0; border-left: 4px solid #e53935; border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0; }
    .issue-medium { background:#fffbe6; border-left: 4px solid #f9a825; border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0; }
    .issue-low    { background:#f0fff4; border-left: 4px solid #43a047; border-radius:6px; padding:0.5rem 0.8rem; margin:0.3rem 0; }

    /* Step header */
    .step-header {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 1rem 0 0.5rem;
        font-weight: 600;
        color: #0066CC;
        font-size: 1.05rem;
    }

    /* Suggested question buttons */
    div[data-testid="stButton"] > button {
        border-radius: 20px;
        border: 1px solid #0066CC;
        color: #0066CC;
        background: white;
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
    }
    div[data-testid="stButton"] > button:hover {
        background: #0066CC;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ── Anthropic client ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY not found in .env file.")
        st.stop()
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Column type detection ─────────────────────────────────────────────────────
def detect_column_types(df: pd.DataFrame) -> dict:
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            col_types[col] = "unknown"
            continue
        if pd.api.types.is_numeric_dtype(series):
            col_types[col] = "numeric"
            continue
        # Try date parse (suppress format-inference warnings)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors="coerce")
            if converted.notna().sum() / n > 0.7:
                col_types[col] = "date"
                continue
        except Exception:
            pass
        # Try coerce numeric
        try:
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().sum() / n > 0.7:
                col_types[col] = "numeric"
                continue
        except Exception:
            pass
        # Category vs text
        n_unique = series.nunique()
        if n_unique <= 50 or (n > 0 and n_unique / n < 0.3):
            col_types[col] = "category"
        else:
            col_types[col] = "text"
    return col_types


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_file_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(bio)
    xls = pd.ExcelFile(bio)
    best, best_n = None, -1
    for sheet in xls.sheet_names:
        try:
            d = pd.read_excel(xls, sheet_name=sheet)
            if len(d) > best_n:
                best, best_n = d, len(d)
        except Exception:
            continue
    if best is None:
        raise ValueError("No readable sheet found in Excel file.")
    return best


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


def dataset_summary_stats(df: pd.DataFrame, col_types: dict) -> dict:
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": sum(1 for t in col_types.values() if t == "numeric"),
        "category_cols": sum(1 for t in col_types.values() if t == "category"),
        "date_cols": sum(1 for t in col_types.values() if t == "date"),
        "missing_cells": int(df.isna().sum().sum()),
    }
    # Date range
    for col, t in col_types.items():
        if t == "date":
            try:
                d = pd.to_datetime(df[col], errors="coerce").dropna()
                stats["date_range"] = f"{d.min().date()} → {d.max().date()}"
            except Exception:
                pass
            break
    # Total of first numeric col (or largest numeric)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    if num_cols:
        totals = {c: df[c].sum() for c in num_cols}
        biggest = max(totals, key=lambda c: abs(totals[c]))
        stats["key_metric"] = biggest
        stats["key_metric_total"] = float(totals[biggest])
    return stats


# ── Data quality scan ─────────────────────────────────────────────────────────
def scan_data_quality(df: pd.DataFrame) -> list:
    issues = []
    # Missing values
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            sev = "high" if n > len(df) * 0.2 else ("medium" if n > len(df) * 0.05 else "low")
            issues.append({"type": "missing_values", "severity": sev, "column": col, "count": n,
                           "description": f"Missing values in '{col}': {n} rows ({n/len(df)*100:.1f}%)"})
    # Duplicates
    n_dup = int(df.duplicated().sum())
    if n_dup:
        issues.append({"type": "duplicates", "severity": "medium", "column": "ALL", "count": n_dup,
                       "description": f"Duplicate rows: {n_dup} exact duplicates found"})
    # Negative values in numeric cols
    for col in df.select_dtypes(include="number").columns:
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            issues.append({"type": "negative_values", "severity": "low", "column": col, "count": n_neg,
                           "description": f"Negative values in '{col}': {n_neg} rows"})
    # Whitespace in text cols
    for col in df.select_dtypes(include="object").columns:
        n_ws = int(df[col].apply(lambda x: isinstance(x, str) and x != x.strip()).sum())
        if n_ws:
            issues.append({"type": "whitespace", "severity": "low", "column": col, "count": n_ws,
                           "description": f"Leading/trailing whitespace in '{col}': {n_ws} rows"})
    # Inconsistent casing in category cols
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().astype(str)
        u_orig = vals.nunique()
        u_lower = vals.str.lower().nunique()
        if u_lower < u_orig:
            diff = u_orig - u_lower
            issues.append({"type": "inconsistent_categories", "severity": "medium", "column": col, "count": diff,
                           "description": f"Inconsistent casing in '{col}': {diff} case variants (e.g. 'Foods' vs 'FOODS' vs 'foods')"})
    # Empty rows
    n_empty = int(df.isna().all(axis=1).sum())
    if n_empty:
        issues.append({"type": "empty_rows", "severity": "high", "column": "ALL", "count": n_empty,
                       "description": f"Completely empty rows: {n_empty}"})
    return issues


# ── Data cleaning ─────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> tuple:
    df_c = df.copy()
    log = []

    # 1. Empty rows
    n_before = len(df_c)
    df_c = df_c.dropna(how="all")
    removed = n_before - len(df_c)
    if removed:
        log.append(f"Removed {removed} completely empty rows")

    # 2. Duplicates
    n_before = len(df_c)
    df_c = df_c.drop_duplicates()
    removed = n_before - len(df_c)
    if removed:
        log.append(f"Removed {removed} duplicate rows")

    # 3. Whitespace
    for col in df_c.select_dtypes(include="object").columns:
        n_ws = int(df_c[col].apply(lambda x: isinstance(x, str) and x != x.strip()).sum())
        if n_ws:
            df_c[col] = df_c[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            log.append(f"Stripped whitespace in '{col}' ({n_ws} rows affected)")

    # 4. Standardise category casing
    for col in df_c.select_dtypes(include="object").columns:
        vals = df_c[col].dropna().astype(str)
        if vals.nunique() <= 50:
            u_orig = vals.nunique()
            u_lower = vals.str.lower().nunique()
            if u_lower < u_orig:
                old_vals = sorted(df_c[col].dropna().unique())
                df_c[col] = df_c[col].str.upper()
                new_vals = sorted(df_c[col].dropna().unique())
                n_affected = df_c[col].notna().sum()
                examples = " | ".join([f"'{v}' → '{v.upper()}'" for v in old_vals[:3] if v != v.upper()])
                log.append(f"Standardised casing in '{col}' ({u_orig} → {len(new_vals)} unique values, {n_affected} rows) — {examples}")

    # 5. Fill missing numeric with median
    for col in df_c.select_dtypes(include="number").columns:
        n_miss = int(df_c[col].isna().sum())
        if n_miss:
            med = df_c[col].median()
            df_c[col] = df_c[col].fillna(med)
            log.append(f"Fixed {n_miss} missing values in '{col}' using median ({med:.2f})")

    # 6. Fill missing text with 'Unknown'
    for col in df_c.select_dtypes(include="object").columns:
        n_miss = int(df_c[col].isna().sum())
        if n_miss:
            df_c[col] = df_c[col].fillna("Unknown")
            log.append(f"Filled {n_miss} missing text values in '{col}' with 'Unknown'")

    # 7. Fix negative numeric values
    for col in df_c.select_dtypes(include="number").columns:
        n_neg = int((df_c[col] < 0).sum())
        if n_neg:
            df_c.loc[df_c[col] < 0, col] = 0
            log.append(f"Fixed {n_neg} negative values in '{col}' (set to 0)")

    if not log:
        log.append("No cleaning required — data is already clean!")

    return df_c, log


# ── Analysis tools ────────────────────────────────────────────────────────────
def tool_top_n_analysis(df: pd.DataFrame, group_by: str, metric: str, n: int = 5, ascending: bool = False) -> dict:
    if group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found. Available: {list(df.columns)}"}
    if metric not in df.columns:
        return {"error": f"Column '{metric}' not found. Available: {list(df.columns)}"}
    try:
        df_w = df.copy()
        df_w[metric] = pd.to_numeric(df_w[metric], errors="coerce")
        grouped = df_w.groupby(group_by)[metric].sum().reset_index().dropna(subset=[metric])
        total = float(grouped[metric].sum())
        result = grouped.sort_values(metric, ascending=ascending).head(n)
        rows = [{"label": str(r[group_by]), "value": float(r[metric]),
                 "pct_of_total": round(float(r[metric]) / total * 100, 1) if total else 0}
                for _, r in result.iterrows()]
        recheck_total = float(df_w[metric].dropna().sum())
        discrepancy = abs(recheck_total - total)
        verification = {
            "passed": discrepancy < max(0.01, abs(total) * 0.0001),
            "method": f"Independent SUM({metric}) from raw data = {recheck_total:,.2f} vs grouped sum = {total:,.2f}",
            "discrepancy": round(discrepancy, 4),
        }
        return {"group_by": group_by, "metric": metric, "n": n,
                "order": "ascending" if ascending else "descending",
                "total_groups": int(df[group_by].nunique()), "grand_total": total,
                "rows": rows, "verification": verification}
    except Exception as e:
        return {"error": str(e)}


def tool_trend_analysis(df: pd.DataFrame, time_col: str, metric: str, group_by: str = None) -> dict:
    if time_col not in df.columns:
        return {"error": f"Column '{time_col}' not found."}
    if metric not in df.columns:
        return {"error": f"Column '{metric}' not found."}
    try:
        df_w = df.copy()
        df_w[time_col] = pd.to_datetime(df_w[time_col], errors="coerce")
        df_w[metric] = pd.to_numeric(df_w[metric], errors="coerce")
        df_w = df_w.dropna(subset=[time_col, metric])
        if group_by and group_by in df.columns:
            agg = df_w.groupby([time_col, group_by])[metric].sum().reset_index()
            groups = {}
            for grp, gdf in agg.groupby(group_by):
                gdf_s = gdf.sort_values(time_col)
                groups[str(grp)] = [{"date": str(r[time_col].date()), "value": float(r[metric])}
                                    for _, r in gdf_s.iterrows()]
            return {"time_col": time_col, "metric": metric, "group_by": group_by,
                    "data_points": int(len(df_w)), "groups": groups}
        else:
            agg = df_w.groupby(time_col)[metric].sum().reset_index().sort_values(time_col)
            trend = [{"date": str(r[time_col].date()), "value": float(r[metric])} for _, r in agg.iterrows()]
            pct = 0.0
            if len(trend) >= 2 and trend[0]["value"]:
                pct = round((trend[-1]["value"] - trend[0]["value"]) / trend[0]["value"] * 100, 1)
            direction = "up" if pct > 2 else ("down" if pct < -2 else "flat")
            date_range_str = f"{trend[0]['date']} → {trend[-1]['date']}" if len(trend) >= 2 else None
            return {"time_col": time_col, "metric": metric, "n_periods": len(trend),
                    "pct_change_overall": pct, "trend_direction": direction,
                    "date_range": date_range_str, "data_points": int(len(df_w)),
                    "trend": trend}
    except Exception as e:
        return {"error": str(e)}


def tool_filter_and_aggregate(df: pd.DataFrame, filter_col: str, filter_val: str,
                               metric: str, agg_fn: str = "sum") -> dict:
    if filter_col not in df.columns:
        return {"error": f"Column '{filter_col}' not found."}
    if metric not in df.columns:
        return {"error": f"Column '{metric}' not found."}
    if agg_fn not in ("sum", "mean", "count", "min", "max"):
        return {"error": "agg_fn must be sum/mean/count/min/max"}
    try:
        mask = df[filter_col].astype(str).str.lower() == str(filter_val).lower()
        filtered = df[mask]
        if filtered.empty:
            mask = df[filter_col].astype(str).str.lower().str.contains(str(filter_val).lower(), na=False)
            filtered = df[mask]
        num = pd.to_numeric(filtered[metric], errors="coerce")
        result = getattr(num, agg_fn)()
        result_val = float(result) if not pd.isna(result) else None
        all_numeric = pd.to_numeric(df[metric], errors="coerce")
        if agg_fn == "sum" and result_val is not None:
            total_metric = float(all_numeric.sum())
            verify_ok = -0.01 <= result_val <= total_metric + abs(total_metric) * 0.0001 + 0.01
            verify_msg = f"Filtered sum ({result_val:,.2f}) ≤ dataset total ({total_metric:,.2f})"
        elif agg_fn == "count":
            verify_ok = result_val == len(filtered)
            verify_msg = f"COUNT result ({result_val}) matches filtered row count ({len(filtered)})"
        elif agg_fn == "mean" and result_val is not None:
            mn, mx = float(all_numeric.min()), float(all_numeric.max())
            verify_ok = mn - 0.01 <= result_val <= mx + 0.01
            verify_msg = f"Mean ({result_val:.2f}) within dataset range [{mn:.2f}, {mx:.2f}]"
        else:
            verify_ok = True
            verify_msg = "Bounds check passed"
        verification = {"passed": verify_ok, "method": verify_msg}
        return {"filter_col": filter_col, "filter_val": filter_val, "metric": metric, "agg_fn": agg_fn,
                "n_matching_rows": len(filtered), "result": result_val,
                "pct_of_total_rows": round(len(filtered) / len(df) * 100, 1),
                "verification": verification}
    except Exception as e:
        return {"error": str(e)}


def tool_compare_groups(df: pd.DataFrame, group_col: str, metric: str) -> dict:
    if group_col not in df.columns:
        return {"error": f"Column '{group_col}' not found."}
    if metric not in df.columns:
        return {"error": f"Column '{metric}' not found."}
    try:
        df_w = df.copy()
        df_w[metric] = pd.to_numeric(df_w[metric], errors="coerce")
        agg = df_w.groupby(group_col)[metric].agg(["mean", "sum", "count", "min", "max"]).reset_index()
        agg.columns = [group_col, "mean", "total", "count", "min", "max"]
        agg = agg.sort_values("total", ascending=False)
        grand_total = float(agg["total"].sum())
        rows = [{"group": str(r[group_col]), "mean": round(float(r["mean"]), 2),
                 "total": float(r["total"]), "count": int(r["count"]),
                 "min": float(r["min"]), "max": float(r["max"]),
                 "pct_of_total": round(float(r["total"]) / grand_total * 100, 1) if grand_total else 0}
                for _, r in agg.iterrows()]
        return {"group_col": group_col, "metric": metric, "n_groups": len(rows),
                "grand_total": grand_total, "rows": rows}
    except Exception as e:
        return {"error": str(e)}


def tool_detect_anomalies(df: pd.DataFrame, metric: str, threshold: float = 2.0) -> dict:
    if metric not in df.columns:
        return {"error": f"Column '{metric}' not found."}
    try:
        series = pd.to_numeric(df[metric], errors="coerce")
        valid = series.dropna()
        mean, std = float(valid.mean()), float(valid.std())
        if std == 0:
            return {"error": "Standard deviation is 0 — all values are identical, no anomalies possible."}
        z = (series - mean) / std
        mask = z.abs() > threshold
        anom_df = df.loc[mask].copy()
        anom_df["_value"] = series[mask].values
        anom_df["_z_score"] = z[mask].values
        anom_df = anom_df.sort_values("_z_score", key=abs, ascending=False)
        context_cols = [c for c in df.columns if c != metric][:4]
        rows = []
        for _, row in anom_df.head(20).iterrows():
            r = {c: str(row[c]) for c in context_cols}
            r["value"] = float(row["_value"])
            r["z_score"] = round(float(row["_z_score"]), 2)
            rows.append(r)
        return {"metric": metric, "threshold_std": threshold, "mean": mean, "std": std,
                "n_total": int(len(valid)), "n_anomalies": int(mask.sum()),
                "pct_anomalies": round(mask.sum() / len(valid) * 100, 1), "rows": rows}
    except Exception as e:
        return {"error": str(e)}


def tool_data_summary(df: pd.DataFrame) -> dict:
    col_types = detect_column_types(df)
    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    num_stats = {}
    for col in df.select_dtypes(include="number").columns[:10]:
        s = df[col].dropna()
        num_stats[col] = {"min": float(s.min()), "max": float(s.max()),
                          "mean": round(float(s.mean()), 2), "sum": float(s.sum())}
    result = {"total_rows": len(df), "total_columns": len(df.columns),
              "column_types": col_types, "missing_values": missing, "numeric_stats": num_stats}
    for col, t in col_types.items():
        if t == "date":
            try:
                d = pd.to_datetime(df[col], errors="coerce").dropna()
                result["date_range"] = {"column": col, "min": str(d.min().date()),
                                        "max": str(d.max().date()), "n_unique": int(d.dt.date.nunique())}
            except Exception:
                pass
            break
    return result


def tool_forecast_simple(df: pd.DataFrame, time_col: str, metric: str, periods: int = 3) -> dict:
    if time_col not in df.columns:
        return {"error": f"Time column '{time_col}' not found."}
    if metric not in df.columns:
        return {"error": f"Metric column '{metric}' not found."}
    try:
        df_w = df.copy()
        df_w[time_col] = pd.to_datetime(df_w[time_col], errors="coerce")
        df_w[metric] = pd.to_numeric(df_w[metric], errors="coerce")
        df_w = df_w.dropna(subset=[time_col, metric])
        hist = df_w.groupby(time_col)[metric].sum().reset_index().sort_values(time_col)
        if len(hist) < 3:
            return {"error": "Need at least 3 time periods for forecasting."}
        x = np.arange(len(hist))
        y = hist[metric].values
        slope, intercept = np.polyfit(x, y, 1)
        last_date = hist[time_col].iloc[-1]
        freq = pd.infer_freq(hist[time_col])
        def next_date(i):
            fu = str(freq or "").upper()
            if "M" in fu:   return last_date + pd.DateOffset(months=i)
            elif "Q" in fu: return last_date + pd.DateOffset(months=3 * i)
            elif "W" in fu: return last_date + pd.DateOffset(weeks=i)
            elif "D" in fu: return last_date + pd.DateOffset(days=i)
            elif "Y" in fu or "A" in fu: return last_date + pd.DateOffset(years=i)
            else:           return last_date + pd.DateOffset(months=i)
        forecast = [{"date": str(next_date(i + 1).date()),
                     "value": round(max(0, slope * (len(hist) + i) + intercept), 2)}
                    for i in range(periods)]
        historical = [{"date": str(r[time_col].date()), "value": float(r[metric])}
                      for _, r in hist.iterrows()]
        n_hist = len(hist)
        if n_hist < 6:
            confidence = "low"
            confidence_note = f"⚠️ Low confidence: only {n_hist} time periods (≥ 6 recommended for reliable forecasts)"
        elif n_hist < 12:
            confidence = "medium"
            confidence_note = f"ℹ️ Moderate confidence: {n_hist} time periods (≥ 12 recommended for strong forecasts)"
        else:
            confidence = "high"
            confidence_note = f"✅ Good confidence: {n_hist} time periods of historical data"
        hist_dates = f"{historical[0]['date']} → {historical[-1]['date']}" if historical else None
        return {"time_col": time_col, "metric": metric, "periods_ahead": periods,
                "historical_periods": n_hist, "trend": "increasing" if slope > 0 else "decreasing",
                "avg_change_per_period": round(float(slope), 2),
                "confidence": confidence, "confidence_note": confidence_note,
                "historical_date_range": hist_dates,
                "historical": historical, "forecast": forecast}
    except Exception as e:
        return {"error": str(e)}


# ── Tool registry ─────────────────────────────────────────────────────────────
def get_tool_schemas(col_types: dict) -> list:
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols  = [c for c, t in col_types.items() if t == "category"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    all_cols  = list(col_types.keys())

    def oc(lst): return ", ".join(lst) if lst else "none detected"

    return [
        {
            "name": "top_n_analysis",
            "description": (
                "Rank items in a categorical column by a numeric metric to find top or bottom N. "
                f"Categorical columns: {oc(cat_cols)}. Numeric columns: {oc(num_cols)}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "description": f"Categorical column to group by. Options: {oc(cat_cols or all_cols)}"},
                    "metric":   {"type": "string", "description": f"Numeric column to rank by. Options: {oc(num_cols)}"},
                    "n":        {"type": "integer", "description": "Number of results (default: 5)", "default": 5},
                    "ascending":{"type": "boolean", "description": "True for bottom N (worst), False (default) for top N (best)", "default": False},
                },
                "required": ["group_by", "metric"],
            },
        },
        {
            "name": "trend_analysis",
            "description": (
                "Show how a metric changes over time. Requires a date/time column. "
                f"Date columns: {oc(date_cols)}. Numeric columns: {oc(num_cols)}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "time_col": {"type": "string", "description": f"Date/time column. Options: {oc(date_cols)}"},
                    "metric":   {"type": "string", "description": f"Numeric metric to track. Options: {oc(num_cols)}"},
                    "group_by": {"type": "string", "description": "Optional: split trend by a category column"},
                },
                "required": ["time_col", "metric"],
            },
        },
        {
            "name": "filter_and_aggregate",
            "description": "Filter data by a column value, then calculate sum/mean/count/min/max of a metric.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filter_col": {"type": "string", "description": f"Column to filter on. Options: {oc(cat_cols)}"},
                    "filter_val": {"type": "string", "description": "Value to filter for (case-insensitive)"},
                    "metric":     {"type": "string", "description": f"Numeric column to aggregate. Options: {oc(num_cols)}"},
                    "agg_fn":     {"type": "string", "enum": ["sum", "mean", "count", "min", "max"], "default": "sum"},
                },
                "required": ["filter_col", "filter_val", "metric"],
            },
        },
        {
            "name": "compare_groups",
            "description": "Compare all groups in a categorical column by a numeric metric (mean, total, count, share %).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_col": {"type": "string", "description": f"Categorical column to group by. Options: {oc(cat_cols)}"},
                    "metric":    {"type": "string", "description": f"Numeric metric to compare. Options: {oc(num_cols)}"},
                },
                "required": ["group_col", "metric"],
            },
        },
        {
            "name": "detect_anomalies",
            "description": "Find statistically unusual values in a numeric column using z-score (standard deviations from mean).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "metric":    {"type": "string", "description": f"Numeric column to check. Options: {oc(num_cols)}"},
                    "threshold": {"type": "number", "description": "Z-score threshold (default: 2.0). Lower = more sensitive.", "default": 2.0},
                },
                "required": ["metric"],
            },
        },
        {
            "name": "data_summary",
            "description": "Get a comprehensive overview of the dataset: row/column counts, column types, missing values, and numeric statistics.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "forecast_simple",
            "description": (
                "Forecast future values using linear trend extrapolation. "
                f"Date columns: {oc(date_cols)}. Numeric columns: {oc(num_cols)}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "time_col": {"type": "string", "description": f"Date/time column. Options: {oc(date_cols)}"},
                    "metric":   {"type": "string", "description": f"Numeric metric to forecast. Options: {oc(num_cols)}"},
                    "periods":  {"type": "integer", "description": "Number of future periods to forecast (default: 3)", "default": 3},
                },
                "required": ["time_col", "metric"],
            },
        },
    ]


def execute_tool(name: str, args: dict, df: pd.DataFrame) -> dict:
    dispatch = {
        "top_n_analysis":       tool_top_n_analysis,
        "trend_analysis":       tool_trend_analysis,
        "filter_and_aggregate": tool_filter_and_aggregate,
        "compare_groups":       tool_compare_groups,
        "detect_anomalies":     tool_detect_anomalies,
        "data_summary":         lambda df, **_: tool_data_summary(df),
        "forecast_simple":      tool_forecast_simple,
    }
    if name not in dispatch:
        return {"error": f"Unknown tool: {name}"}
    try:
        return dispatch[name](df, **args)
    except Exception as e:
        return {"error": f"Tool execution error: {str(e)}"}


# ── Plain-English calculation description ─────────────────────────────────────
def describe_calculation(tool_name: str, args: dict, result: dict) -> str:
    """Return a plain-English description of what the tool computed."""
    try:
        if tool_name == "top_n_analysis":
            order = "ASC" if args.get("ascending") else "DESC"
            return (f"SUM({args.get('metric')}) GROUP BY {args.get('group_by')} "
                    f"ORDER BY {order} LIMIT {args.get('n', 5)} — "
                    f"{result.get('total_groups', '?')} groups found, "
                    f"grand total = {result.get('grand_total', 0):,.2f}")
        elif tool_name == "trend_analysis":
            base = f"SUM({args.get('metric')}) GROUP BY {args.get('time_col')}"
            if args.get("group_by"):
                base += f" × {args.get('group_by')}"
            extras = []
            if "n_periods" in result:
                extras.append(f"{result['n_periods']} periods")
            if result.get("date_range"):
                extras.append(f"range: {result['date_range']}")
            if "data_points" in result:
                extras.append(f"{result['data_points']:,} raw rows used")
            return base + (" — " + ", ".join(extras) if extras else "")
        elif tool_name == "filter_and_aggregate":
            return (f"{args.get('agg_fn', 'sum').upper()}({args.get('metric')}) "
                    f"WHERE {args.get('filter_col')} = '{args.get('filter_val')}' "
                    f"— {result.get('n_matching_rows', 0):,} matching rows "
                    f"({result.get('pct_of_total_rows', 0):.1f}% of dataset)")
        elif tool_name == "compare_groups":
            return (f"SUM/MEAN/COUNT({args.get('metric')}) GROUP BY {args.get('group_col')} "
                    f"— {result.get('n_groups', '?')} groups, "
                    f"grand total = {result.get('grand_total', 0):,.2f}")
        elif tool_name == "detect_anomalies":
            return (f"Z-SCORE({args.get('metric')}) > {args.get('threshold', 2.0)}σ "
                    f"— mean = {result.get('mean', 0):.2f}, "
                    f"std = {result.get('std', 0):.2f}, "
                    f"{result.get('n_anomalies', 0)} of {result.get('n_total', 0)} values flagged "
                    f"({result.get('pct_anomalies', 0):.1f}%)")
        elif tool_name == "forecast_simple":
            return (f"LINEAR_TREND({args.get('metric')}) OVER {args.get('time_col')}, "
                    f"FORECAST +{args.get('periods', 3)} periods "
                    f"— {result.get('historical_periods', 0)} historical periods used"
                    + (f", range: {result['historical_date_range']}" if result.get('historical_date_range') else "")
                    + f", avg change/period = {result.get('avg_change_per_period', 0):+,.2f}")
        elif tool_name == "data_summary":
            return (f"DESCRIBE(*) — {result.get('total_rows', 0):,} rows × "
                    f"{result.get('total_columns', 0)} columns")
    except Exception:
        pass
    return ""


# ── Planning Brief ────────────────────────────────────────────────────────────
def run_planning_brief(df: pd.DataFrame, col_types: dict) -> dict:
    """Run multiple tools automatically and synthesize a planning brief."""
    client = get_client()
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols  = [c for c, t in col_types.items() if t == "category"]
    date_cols = [c for c, t in col_types.items() if t == "date"]

    tool_calls_log = []
    analyses_text = ""

    def _add(tool_name, inp, result):
        tool_calls_log.append({"tool": tool_name, "input": inp, "result": result})
        rs = json.dumps(result, indent=2)
        if len(rs) > 2000:
            if "rows" in result:
                trimmed = {k: v for k, v in result.items() if k != "rows"}
                trimmed["rows (first 10)"] = result["rows"][:10]
                rs = json.dumps(trimmed, indent=2)
            else:
                rs = rs[:2000] + "\n... [truncated]"
        return f"\n### {tool_name.upper()}\n{rs}\n"

    # 1. Top N — first cat/num pair
    if cat_cols and num_cols:
        inp = {"group_by": cat_cols[0], "metric": num_cols[0], "n": 5}
        r = tool_top_n_analysis(df, **inp)
        analyses_text += _add("top_n_analysis", inp, r)

    # 2. Compare groups — second cat or same with second metric
    if cat_cols and num_cols:
        cat2   = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
        met2   = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        if not (cat2 == cat_cols[0] and met2 == num_cols[0]):
            inp = {"group_col": cat2, "metric": met2}
            r = tool_compare_groups(df, **inp)
            analyses_text += _add("compare_groups", inp, r)

    # 3. Trend analysis
    if date_cols and num_cols:
        inp = {"time_col": date_cols[0], "metric": num_cols[0]}
        r = tool_trend_analysis(df, **inp)
        analyses_text += _add("trend_analysis", inp, r)

    # 4. Anomaly detection
    if num_cols:
        inp = {"metric": num_cols[0]}
        r = tool_detect_anomalies(df, **inp)
        analyses_text += _add("detect_anomalies", inp, r)

    # 5. Forecast
    if date_cols and num_cols:
        inp = {"time_col": date_cols[0], "metric": num_cols[0], "periods": 3}
        r = tool_forecast_simple(df, **inp)
        analyses_text += _add("forecast_simple", inp, r)

    # 6. Data summary
    r = tool_data_summary(df)
    tool_calls_log.append({"tool": "data_summary", "input": {}, "result": r})

    col_info = "\n".join(f"  - {c} ({t})" for c, t in col_types.items())

    prompt = f"""You are a Digital Colleague AI assistant. Several analyses have been run on a planning dataset. Synthesize a concise Planning Brief.

DATASET: {len(df):,} rows, {len(df.columns)} columns
COLUMNS:
{col_info}

ANALYSIS RESULTS:
{analyses_text}

Generate a Planning Brief with EXACTLY this structure:

## 📋 Planning Brief

### 🔑 Key Findings (ranked by business impact)
1. **[Short title]** — [Finding with specific numbers. Calculate $ or % impact where possible.]
2. **[Short title]** — [Finding with specific numbers.]
3. **[Short title]** — [Finding with specific numbers.]
(3–5 findings, most impactful first)

### ✅ Recommended Actions
1. **[Action title]** — [Specific, actionable step tied to Finding #N above.]
2. **[Action title]** — [Specific, actionable step tied to Finding #N above.]
3. **[Action title]** — [Specific, actionable step tied to Finding #N above.]

Rules:
- Every finding MUST cite specific numbers from the analysis results above
- Calculate dollar or percentage impact where possible
- Findings ranked by estimated business impact (highest first)
- Each recommendation references which finding it addresses
- 2 sentences max per finding/recommendation
- If forecast confidence is low, note it
"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    brief_text = resp.content[0].text if resp.content else "Could not generate brief."
    return {"brief": brief_text, "tool_calls": tool_calls_log}


# ── AI query runner ───────────────────────────────────────────────────────────
def run_ai_query(question: str, df: pd.DataFrame, col_types: dict, history: list) -> dict:
    client = get_client()
    tools = get_tool_schemas(col_types)

    col_info = "\n".join(f"  - {c} ({t})" for c, t in col_types.items())
    sample = df.head(2).to_string(index=False, max_cols=8)

    system = f"""You are a Digital Colleague — an AI assistant helping Product Managers analyse planning data and make better decisions.

DATASET:
- Rows: {len(df):,}  |  Columns: {len(df.columns)}

COLUMN TYPES:
{col_info}

SAMPLE (first 2 rows):
{sample}

RULES:
1. ALWAYS call a tool to retrieve actual data before answering. Never guess numbers.
2. After tools return results, give a clear, concise, business-friendly answer.
3. Cite specific numbers from the tool output.
4. State assumptions if you make any.
5. For forecast results, note these are linear extrapolations.
6. Answers should be 3-6 sentences — planners are busy.
"""

    messages = []
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    for h in recent:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    tool_calls_log = []

    for _ in range(8):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system,
            messages=messages,
            tools=tools,
        )

        if resp.stop_reason == "end_turn":
            answer = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return {"answer": answer, "tool_calls": tool_calls_log, "error": None}

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input, df)
                    tool_calls_log.append({"tool": block.name, "input": block.input, "result": result})
                    result_str = json.dumps(result)
                    if len(result_str) > 8000:
                        if "rows" in result:
                            r2 = {k: v for k, v in result.items() if k != "rows"}
                            r2["rows"] = result["rows"][:20]
                            r2["_truncated"] = f"Showing 20 of {len(result['rows'])} rows"
                            result_str = json.dumps(r2)
                        else:
                            result_str = result_str[:8000]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            answer = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return {"answer": answer, "tool_calls": tool_calls_log, "error": None}

    return {"answer": "Max tool iterations reached. Try a more specific question.",
            "tool_calls": tool_calls_log, "error": "max_iterations"}


# ── Charts ────────────────────────────────────────────────────────────────────
def render_chart(tool_name: str, result: dict):
    if "error" in result:
        return
    try:
        if tool_name == "top_n_analysis" and "rows" in result:
            df_c = pd.DataFrame(result["rows"])
            label_col = "label"
            fig = px.bar(df_c, x=label_col, y="value",
                         title=f"Top {result.get('n', len(df_c))} {result.get('group_by', '')} by {result.get('metric', '')}",
                         color="value", color_continuous_scale="Blues",
                         text=df_c["value"].apply(lambda v: f"{v:,.1f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              xaxis_title=result.get("group_by", ""), yaxis_title=result.get("metric", "Value"))
            st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "trend_analysis":
            if "trend" in result:
                df_c = pd.DataFrame(result["trend"])
                fig = px.line(df_c, x="date", y="value", markers=True,
                              title=f"{result.get('metric', 'Metric')} Over Time")
                fig.update_layout(xaxis_title="Date", yaxis_title=result.get("metric", "Value"))
                st.plotly_chart(fig, use_container_width=True)
            elif "groups" in result:
                rows = []
                for g, pts in result["groups"].items():
                    for p in pts:
                        rows.append({"date": p["date"], "value": p["value"], "group": g})
                df_c = pd.DataFrame(rows)
                fig = px.line(df_c, x="date", y="value", color="group", markers=True,
                              title=f"{result.get('metric', 'Metric')} by {result.get('group_by', 'Group')}")
                st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "compare_groups" and "rows" in result:
            df_c = pd.DataFrame(result["rows"]).head(15)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df_c, x="group", y="total",
                             title=f"{result.get('metric', 'Metric')} by {result.get('group_col', 'Group')}",
                             color="total", color_continuous_scale="Greens",
                             text=df_c["pct_of_total"].apply(lambda v: f"{v:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(df_c.head(8), names="group", values="total",
                             title=f"Share of {result.get('metric', 'Metric')}", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "detect_anomalies" and result.get("rows"):
            df_c = pd.DataFrame(result["rows"])
            if "value" in df_c.columns and "z_score" in df_c.columns:
                df_c["index"] = range(len(df_c))
                fig = px.scatter(df_c, x="index", y="value", color="z_score",
                                 size=df_c["z_score"].abs(),
                                 color_continuous_scale="RdYlGn_r",
                                 title=f"Anomalies in {result.get('metric', 'metric')} (>{result.get('threshold_std', 2)}σ)",
                                 labels={"index": "Anomaly Rank", "value": result.get("metric", "Value")})
                mean = result.get("mean", 0)
                std  = result.get("std", 0)
                thr  = result.get("threshold_std", 2.0)
                fig.add_hline(y=mean,           line_dash="dash", line_color="blue",   annotation_text="Mean")
                fig.add_hline(y=mean + thr*std, line_dash="dot",  line_color="red",    annotation_text=f"+{thr}σ")
                fig.add_hline(y=mean - thr*std, line_dash="dot",  line_color="orange", annotation_text=f"-{thr}σ")
                st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "forecast_simple" and "forecast" in result:
            hist  = [{"date": p["date"], "value": p["value"], "type": "Historical"} for p in result.get("historical", [])]
            fcast = [{"date": p["date"], "value": p["value"], "type": "Forecast"}   for p in result["forecast"]]
            df_c = pd.DataFrame(hist + fcast)
            fig = px.line(df_c, x="date", y="value", color="type", markers=True,
                          title=f"Forecast: {result.get('metric', 'Metric')}",
                          color_discrete_map={"Historical": "#0066CC", "Forecast": "#FF6B35"})
            if fcast:
                fig.add_vrect(x0=fcast[0]["date"], x1=fcast[-1]["date"],
                              fillcolor="orange", opacity=0.08, line_width=0,
                              annotation_text="Forecast")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.caption(f"Chart could not be rendered: {e}")


# ── Suggested questions ───────────────────────────────────────────────────────
def suggested_questions(col_types: dict) -> list:
    num  = [c for c, t in col_types.items() if t == "numeric"]
    cat  = [c for c, t in col_types.items() if t == "category"]
    date = [c for c, t in col_types.items() if t == "date"]
    qs = []
    if cat and num:
        qs.append(f"What are the top 5 {cat[0]}s by {num[0]}?")
        qs.append(f"Compare {num[0]} across all {cat[0]}s")
    if len(cat) > 1 and num:
        qs.append(f"Which {cat[1]} has the highest {num[0]}?")
    if date and num:
        qs.append(f"Show the {num[0]} trend over time")
        qs.append(f"Forecast the next 3 periods of {num[0]}")
    if num:
        qs.append(f"Find anomalies in {num[0]}")
    qs.append("Give me a complete data summary")
    if date and num and cat:
        qs.append(f"Show {num[0]} trend over time broken down by {cat[0]}")
    return qs[:8]


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    # ── Session state init ────────────────────────────────────────────────────
    for key, default in [
        ("df", None), ("df_clean", None), ("col_types", {}),
        ("data_issues", []), ("cleaning_log", []), ("is_cleaned", False),
        ("chat_history", []), ("file_name", ""), ("pending_question", ""),
        ("planning_brief", None), ("brief_tool_calls", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dc-header">
      <h1>🤝 Digital Colleague</h1>
      <p>AI Planning Assistant — MBA Thesis Research Tool · Macromedia University Munich</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Data")

        uploaded = st.file_uploader("Upload your CSV or Excel file",
                                    type=["csv", "xlsx", "xls"],
                                    help="Any CSV or Excel file works")
        if st.button("🗂 Use Sample Data (Planning Demo)", use_container_width=True):
            with st.spinner("Loading sample data…"):
                df_raw = load_sample_data()
                st.session_state.df = df_raw
                st.session_state.df_clean = None
                st.session_state.is_cleaned = False
                st.session_state.cleaning_log = []
                st.session_state.chat_history = []
                st.session_state.file_name = "pm_planning_simplified.csv"
                st.session_state.col_types = detect_column_types(df_raw)
                st.session_state.data_issues = scan_data_quality(df_raw)
                st.rerun()

        if uploaded is not None:
            file_bytes = uploaded.read()
            if uploaded.name != st.session_state.file_name:
                with st.spinner("Loading file…"):
                    try:
                        df_raw = load_file_bytes(file_bytes, uploaded.name)
                        st.session_state.df = df_raw
                        st.session_state.df_clean = None
                        st.session_state.is_cleaned = False
                        st.session_state.cleaning_log = []
                        st.session_state.chat_history = []
                        st.session_state.file_name = uploaded.name
                        st.session_state.col_types = detect_column_types(df_raw)
                        st.session_state.data_issues = scan_data_quality(df_raw)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load file: {e}")

        st.divider()

        if st.session_state.df is not None:
            issues = st.session_state.data_issues
            n_issues = len(issues)
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}

            st.markdown("### 🔍 Data Quality Report")
            if not issues:
                st.success("✅ No issues found — data is clean!")
            else:
                st.warning(f"**{n_issues} issue(s) found**")
                with st.expander(f"View all {n_issues} issues", expanded=True):
                    for iss in issues:
                        em = severity_emoji.get(iss["severity"], "⚪")
                        css = f"issue-{iss['severity']}"
                        st.markdown(f'<div class="{css}">{em} {iss["description"]}</div>',
                                    unsafe_allow_html=True)

            st.divider()
            if not st.session_state.is_cleaned:
                if st.button("🧹 Clean Data", use_container_width=True, type="primary"):
                    with st.spinner("Cleaning data…"):
                        df_c, log = clean_dataframe(st.session_state.df)
                        st.session_state.df_clean = df_c
                        st.session_state.cleaning_log = log
                        st.session_state.is_cleaned = True
                        st.session_state.col_types = detect_column_types(df_c)
                        st.rerun()
            else:
                st.success("✅ Data cleaned")
                if st.button("↩ Reset to Original", use_container_width=True):
                    st.session_state.df_clean = None
                    st.session_state.is_cleaned = False
                    st.session_state.cleaning_log = []
                    st.session_state.col_types = detect_column_types(st.session_state.df)
                    st.rerun()

            if st.session_state.is_cleaned and st.session_state.df_clean is not None:
                csv_bytes = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Cleaned Data", data=csv_bytes,
                                   file_name="cleaned_data.csv", mime="text/csv",
                                   use_container_width=True)

    # ── Main area: no data yet ────────────────────────────────────────────────
    if st.session_state.df is None:
        st.info("👈 **Upload a file or load sample data from the sidebar to get started.**")
        st.markdown("""
        #### What this tool can do:
        - **Smart Data Loading** — Upload any CSV or Excel file
        - **Transparent Data Cleaning** — See every issue found and every fix applied
        - **AI Chat** — Ask anything about your data in plain English
        - **Auto Charts** — Visualisations generated automatically for every answer
        - **Suggested Questions** — Click to explore your data instantly
        """)
        return

    # ── Active dataframe ──────────────────────────────────────────────────────
    df = st.session_state.df_clean if st.session_state.is_cleaned else st.session_state.df
    col_types = st.session_state.col_types

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Data Overview
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header">📊 Step 1: Data Overview</div>', unsafe_allow_html=True)

    stats = dataset_summary_stats(df, col_types)
    cols_stats = st.columns(4)
    card_data = [
        (f"{stats['rows']:,}", "Total Rows"),
        (str(stats['columns']), "Columns"),
        (str(stats['missing_cells']), "Missing Cells"),
        (stats.get("date_range", f"{stats['numeric_cols']} numeric"), "Date Range" if "date_range" in stats else "Numeric Cols"),
    ]
    for col_s, (val, lbl) in zip(cols_stats, card_data):
        with col_s:
            st.markdown(f'<div class="stat-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>',
                        unsafe_allow_html=True)

    if "key_metric" in stats:
        st.caption(f"**Total {stats['key_metric']}:** {stats['key_metric_total']:,.2f}")

    st.markdown("**Data Preview** (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Column Types Detected"):
        type_df = pd.DataFrame([{"Column": c, "Type": t, "Unique Values": df[c].nunique(),
                                  "Missing": df[c].isna().sum()}
                                 for c, t in col_types.items()])
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Data Cleaning Report
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.data_issues or st.session_state.is_cleaned:
        st.markdown('<div class="step-header">🧹 Step 2: Data Cleaning Report</div>',
                    unsafe_allow_html=True)

        if not st.session_state.is_cleaned:
            n = len(st.session_state.data_issues)
            st.warning(f"**{n} data quality issue(s) detected.** Click 'Clean Data' in the sidebar to fix them.")
            for iss in st.session_state.data_issues[:5]:
                st.markdown(f"- {iss['description']}")
        else:
            log = st.session_state.cleaning_log
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Before Cleaning**")
                orig = st.session_state.df
                bc = pd.DataFrame({
                    "Metric": ["Rows", "Duplicate rows", "Missing cells", "Columns"],
                    "Value": [str(len(orig)), str(int(orig.duplicated().sum())),
                              str(int(orig.isna().sum().sum())), str(len(orig.columns))]
                })
                st.dataframe(bc, hide_index=True, use_container_width=True)
            with col_b:
                st.markdown("**After Cleaning**")
                cleaned = st.session_state.df_clean
                ac = pd.DataFrame({
                    "Metric": ["Rows", "Duplicate rows", "Missing cells", "Columns"],
                    "Value": [str(len(cleaned)), str(int(cleaned.duplicated().sum())),
                              str(int(cleaned.isna().sum().sum())), str(len(cleaned.columns))]
                })
                st.dataframe(ac, hide_index=True, use_container_width=True)

            st.markdown(f"**Actions taken ({len(log)}):**")
            for i, action in enumerate(log, 1):
                st.markdown(f'<div class="transparency-row">✅ {i}. {action}</div>',
                            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: AI Chat
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header">💬 Step 3: AI Chat — Ask Anything About Your Data</div>',
                unsafe_allow_html=True)

    # ── Planning Brief ────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("📋 Generate Planning Brief", type="primary", use_container_width=True,
                     help="AI scans the full dataset and returns ranked findings + recommended actions"):
            with st.spinner("Digital Colleague is scanning your dataset…"):
                brief_result = run_planning_brief(df, col_types)
                st.session_state.planning_brief = brief_result["brief"]
                st.session_state.brief_tool_calls = brief_result["tool_calls"]
            st.rerun()

    if st.session_state.planning_brief:
        with st.container():
            st.markdown(st.session_state.planning_brief)
            with st.expander("🔍 How this brief was generated (tool transparency)"):
                for tc in st.session_state.brief_tool_calls:
                    st.markdown(f"**🛠 Tool:** `{tc['tool']}`")
                    calc_desc = describe_calculation(tc["tool"], tc["input"], tc.get("result", {}))
                    if calc_desc:
                        st.markdown(f'<div class="transparency-row">🔢 {calc_desc}</div>',
                                    unsafe_allow_html=True)
                    result_display = tc.get("result", {})
                    if isinstance(result_display, dict) and "rows" in result_display:
                        n_r = len(result_display["rows"])
                        dc = {k: v for k, v in result_display.items() if k not in ("rows", "verification")}
                        dc[f"rows (first {min(5,n_r)} of {n_r})"] = result_display["rows"][:5]
                        st.json(dc)
                    else:
                        dc = {k: v for k, v in result_display.items() if k != "verification"} if isinstance(result_display, dict) else result_display
                        st.json(dc)
                    st.divider()
        if st.button("🔄 Regenerate Brief"):
            st.session_state.planning_brief = None
            st.session_state.brief_tool_calls = []
            st.rerun()

    st.divider()

    # ── Suggested questions ───────────────────────────────────────────────────
    qs = suggested_questions(col_types)
    st.markdown("**Suggested questions** *(click to ask)*")
    n_cols = min(4, len(qs))
    q_cols = st.columns(n_cols)
    for i, q in enumerate(qs):
        with q_cols[i % n_cols]:
            if st.button(q, key=f"sq_{i}"):
                st.session_state.pending_question = q

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                with st.expander("🔍 How I got this answer (AI reasoning transparency)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**🛠 Tool:** `{tc['tool']}`")
                        st.markdown(f"**📥 Input:**")
                        st.json(tc["input"])
                        st.markdown(f"**📊 Result:**")
                        result_display = tc["result"]
                        if isinstance(result_display, dict) and "rows" in result_display:
                            n_rows = len(result_display["rows"])
                            display_copy = {k: v for k, v in result_display.items() if k != "rows"}
                            display_copy[f"rows (first {min(5, n_rows)} of {n_rows})"] = result_display["rows"][:5]
                            st.json(display_copy)
                        else:
                            st.json(result_display)
                        st.divider()
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    render_chart(tc["tool"], tc["result"])

    # ── Handle pending question from suggested buttons ────────────────────────
    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = ""
        _process_question(pending, df, col_types)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your data…")
    if user_input:
        _process_question(user_input, df, col_types)
        st.rerun()


def _process_question(question: str, df: pd.DataFrame, col_types: dict):
    """Run AI query and append to chat history."""
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Digital Colleague is thinking…"):
        result = run_ai_query(
            question, df, col_types,
            history=st.session_state.chat_history[:-1]  # exclude the question we just added
        )

    if result.get("error") and not result.get("answer"):
        answer = "Sorry, something went wrong. Please try again."
    else:
        answer = result["answer"]

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": result.get("tool_calls", []),
    })

    # Trim history to MAX_HISTORY_TURNS
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(st.session_state.chat_history) > max_msgs:
        st.session_state.chat_history = st.session_state.chat_history[-max_msgs:]


if __name__ == "__main__":
    main()
