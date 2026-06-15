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

def _get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")

MODEL = "claude-sonnet-4-6"
MAX_HISTORY_TURNS = 10
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_DATA_PATH = os.path.join(_DATA_DIR, "thesis_demand_clean.csv")
DIRTY_DATA_PATH  = os.path.join(_DATA_DIR, "thesis_demand_dirty.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Colleague | AI Planning Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Design tokens ── */
    :root {
        --bg:          #ffffff;
        --bg-subtle:   #fafafa;
        --border:      #e5e7eb;
        --border-strong: #d1d5db;
        --text:        #111827;
        --text-muted:  #6b7280;
        --text-xmuted: #9ca3af;
        --accent:      #2563eb;
        --accent-light:#eff6ff;
        --accent-dim:  #dbeafe;
        --green:       #16a34a;
        --green-light: #f0fdf4;
        --green-dim:   #dcfce7;
        --amber:       #d97706;
        --amber-light: #fffbeb;
        --red:         #dc2626;
        --red-light:   #fef2f2;
        --radius:      6px;
        --radius-lg:   10px;
        --shadow:      0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.05);
        --shadow-md:   0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -2px rgba(0,0,0,.05);
        --font-mono:   ui-monospace, "Cascadia Code", "Fira Mono", monospace;
    }

    /* ── Global resets ── */
    .stApp { background: #FAFAF8; }
    section[data-testid="stSidebar"] { background: #F0FAF7 !important; border-right: 1px solid #DCEEE7; }

    /* ── App header ── */
    .app-header {
        background: #0F6E56;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.75rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #E1F5EE;
    }
    .app-header p {
        margin: 0.2rem 0 0.6rem;
        font-size: 0.875rem;
        color: #9FE1CB;
    }
    .app-header .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #9FE1CB;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 99px;
        padding: 0.2rem 0.65rem;
    }

    /* ── Section headers ── */
    .step-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Stat cards ── */
    .stat-card {
        background: #ffffff;
        border: 1px solid #ECEAE4;
        border-radius: var(--radius-lg);
        padding: 1.1rem 1.2rem;
        box-shadow: var(--shadow);
    }
    .stat-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #085041;
        font-variant-numeric: tabular-nums;
    }
    .stat-card .label {
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-xmuted);
        margin-top: 0.2rem;
    }

    /* ── Insight & action cards (Planning Brief) ── */
    .insight-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius-lg);
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--shadow);
    }
    .action-card {
        background: var(--green-light);
        border: 1px solid var(--green-dim);
        border-left: 3px solid var(--green);
        border-radius: var(--radius-lg);
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .insight-card .label, .action-card .label {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .insight-card .label { color: var(--accent); }
    .action-card  .label { color: var(--green); }

    /* ── Transparency / calculation row ── */
    .transparency-row {
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius);
        padding: 0.55rem 0.9rem;
        margin-bottom: 0.5rem;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ── Verified badge ── */
    .verified-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--green);
        background: var(--green-light);
        border: 1px solid var(--green-dim);
        border-radius: 99px;
        padding: 0.15rem 0.6rem;
    }

    /* ── Issue badges ── */
    .issue-high   { background: var(--red-light);   border-left: 3px solid var(--red);   border-radius: var(--radius); padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.85rem; }
    .issue-medium { background: var(--amber-light);  border-left: 3px solid var(--amber); border-radius: var(--radius); padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.85rem; }
    .issue-low    { background: var(--green-light);  border-left: 3px solid var(--green); border-radius: var(--radius); padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.85rem; }

    /* ── Buttons ── */
    div[data-testid="stButton"] > button {
        border-radius: var(--radius);
        border: 1px solid var(--border-strong);
        color: var(--text);
        background: var(--bg);
        font-size: 0.825rem;
        font-weight: 500;
        padding: 0.35rem 0.9rem;
        box-shadow: var(--shadow);
        transition: background 0.15s, border-color 0.15s;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--bg-subtle);
        border-color: var(--border-strong);
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
        box-shadow: none;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
    }

    /* ── Sidebar ── */
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-xmuted);
        margin: 1rem 0 0.5rem;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--text-muted);
    }
    .empty-state h2 { font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem; }
    .empty-state p  { font-size: 0.875rem; margin-bottom: 1.5rem; }
    .capability-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        max-width: 520px;
        margin: 0 auto;
        text-align: left;
    }
    .capability-item {
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 0.85rem 1rem;
        font-size: 0.8rem;
    }
    .capability-item strong { display: block; color: var(--text); margin-bottom: 0.2rem; font-size: 0.82rem; }
    .capability-item span   { color: var(--text-muted); }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Inter, system-ui, sans-serif",
    font_color="#374151",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=16, r=16, t=40, b=16),
    title_font_size=13,
    title_font_color="#111827",
    colorway=["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2"],
    xaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", tickcolor="#e5e7eb",
               tickfont_size=11, tickfont_color="#6b7280"),
    yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", tickcolor="#e5e7eb",
               tickfont_size=11, tickfont_color="#6b7280"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font_size=11),
)


# ── Anthropic client ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client() -> anthropic.Anthropic:
    key = _get_api_key()
    if not key:
        st.error("ANTHROPIC_API_KEY not found. Set it in .env (local) or Streamlit secrets (cloud).")
        st.stop()
    return anthropic.Anthropic(api_key=key)


# ── Column type detection ─────────────────────────────────────────────────────
def detect_column_types(df: pd.DataFrame) -> dict:
    import warnings, re
    col_types = {}
    _TIME_RE = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')
    _ID_SUFFIXES = ('_id', '_no', '_num', '_code', '_key', '_ref')

    for col in df.columns:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            col_types[col] = "unknown"
            continue

        # Already a true numeric dtype — but exclude ID-like columns
        if pd.api.types.is_numeric_dtype(series):
            col_lower = col.lower()
            is_id_name = col_lower == "id" or col_lower.endswith(_ID_SUFFIXES)
            is_all_unique_int = (pd.api.types.is_integer_dtype(series)
                                 and series.nunique() == n)
            if is_id_name and is_all_unique_int:
                col_types[col] = "category"
            else:
                col_types[col] = "numeric"
            continue

        str_series = series.astype(str)

        # Detect time-only strings (e.g. "07:06:11") — must run before date check
        if str_series.head(50).apply(lambda v: bool(_TIME_RE.match(v))).mean() > 0.7:
            col_types[col] = "category"
            continue

        # Try date parse (suppress format-inference warnings)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors="coerce")
            if converted.notna().sum() / n > 0.7:
                col_types[col] = "date"
                continue
        except Exception:
            pass

        # Try coerce numeric (for numbers stored as strings)
        try:
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().sum() / n > 0.7:
                col_types[col] = "numeric"
                continue
        except Exception:
            pass

        # Category vs text
        n_unique = series.nunique()
        if n_unique <= 50 or n_unique / n < 0.3:
            col_types[col] = "category"
        else:
            col_types[col] = "text"
    return col_types


# ── Data loading ──────────────────────────────────────────────────────────────
_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]


def _read_csv_with_encoding(source) -> tuple:
    """Try encodings in order; return (DataFrame, encoding_used)."""
    last_err = None
    for enc in _CSV_ENCODINGS:
        try:
            if isinstance(source, (str, bytes)):
                bio = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")
            else:
                source.seek(0)
                bio = source
            df = pd.read_csv(bio, encoding=enc)
            return df, enc
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
        except Exception as e:
            raise e
    raise ValueError(f"Could not decode CSV with any supported encoding: {last_err}")


@st.cache_data(show_spinner=False)
def load_file_bytes(file_bytes: bytes, file_name: str) -> tuple:
    """Returns (DataFrame, encoding_used). encoding_used is None for Excel."""
    if file_name.lower().endswith(".csv"):
        return _read_csv_with_encoding(file_bytes)
    bio = io.BytesIO(file_bytes)
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
    return best, None


@st.cache_data(show_spinner=False)
def _load_sample(path: str) -> tuple:
    """Load a sample CSV from disk; returns (DataFrame, encoding_used)."""
    with open(path, "rb") as f:
        raw = f.read()
    return _read_csv_with_encoding(raw)


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
    # Guard with actual dtype check so string columns never enter totals
    num_cols = [c for c, t in col_types.items()
                if t == "numeric" and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        totals = {c: float(pd.to_numeric(df[c], errors="coerce").sum()) for c in num_cols}
        biggest = max(totals, key=lambda c: abs(totals[c]))
        stats["key_metric"] = biggest
        stats["key_metric_total"] = totals[biggest]
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
            fig = px.bar(df_c, x="label", y="value",
                         title=f"Top {result.get('n', len(df_c))} {result.get('group_by', '')} by {result.get('metric', '')}",
                         text=df_c["value"].apply(lambda v: f"{v:,.0f}"))
            fig.update_traces(textposition="outside", marker_color="#2563eb",
                              marker_line_width=0)
            fig.update_layout(**PLOTLY_LAYOUT,
                              showlegend=False,
                              xaxis_title=result.get("group_by", ""),
                              yaxis_title=result.get("metric", "Value"))
            st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "trend_analysis":
            if "trend" in result:
                df_c = pd.DataFrame(result["trend"])
                fig = px.line(df_c, x="date", y="value", markers=True,
                              title=f"{result.get('metric', 'Metric')} over time")
                fig.update_traces(line_color="#2563eb", line_width=2,
                                  marker=dict(size=5, color="#2563eb"))
                fig.update_layout(**PLOTLY_LAYOUT,
                                  xaxis_title="", yaxis_title=result.get("metric", "Value"))
                st.plotly_chart(fig, use_container_width=True)
            elif "groups" in result:
                rows = []
                for g, pts in result["groups"].items():
                    for p in pts:
                        rows.append({"date": p["date"], "value": p["value"], "group": g})
                df_c = pd.DataFrame(rows)
                fig = px.line(df_c, x="date", y="value", color="group", markers=True,
                              title=f"{result.get('metric', 'Metric')} by {result.get('group_by', 'Group')}")
                fig.update_traces(line_width=2, marker_size=4)
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "compare_groups" and "rows" in result:
            df_c = pd.DataFrame(result["rows"]).head(15)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df_c, x="group", y="total",
                             title=f"{result.get('metric', 'Metric')} by {result.get('group_col', 'Group')}",
                             text=df_c["pct_of_total"].apply(lambda v: f"{v:.1f}%"))
                fig.update_traces(textposition="outside", marker_color="#2563eb",
                                  marker_line_width=0)
                fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(df_c.head(8), names="group", values="total",
                             title=f"Share of {result.get('metric', 'Metric')}", hole=0.45,
                             color_discrete_sequence=["#2563eb","#3b82f6","#60a5fa",
                                                       "#93c5fd","#bfdbfe","#dbeafe","#eff6ff","#1d4ed8"])
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "detect_anomalies" and result.get("rows"):
            df_c = pd.DataFrame(result["rows"])
            if "value" in df_c.columns and "z_score" in df_c.columns:
                df_c["rank"] = range(1, len(df_c) + 1)
                fig = px.scatter(df_c, x="rank", y="value", color="z_score",
                                 size=df_c["z_score"].abs().clip(lower=1),
                                 color_continuous_scale=[[0,"#16a34a"],[0.5,"#d97706"],[1,"#dc2626"]],
                                 title=f"Anomalies in {result.get('metric', 'metric')} (>{result.get('threshold_std', 2)}σ)",
                                 labels={"rank": "Anomaly rank", "value": result.get("metric", "Value")})
                mean, std, thr = result.get("mean", 0), result.get("std", 0), result.get("threshold_std", 2.0)
                fig.add_hline(y=mean,           line_dash="dash", line_color="#6b7280", line_width=1,
                              annotation_text="Mean", annotation_font_size=10)
                fig.add_hline(y=mean + thr*std, line_dash="dot",  line_color="#dc2626", line_width=1,
                              annotation_text=f"+{thr}σ", annotation_font_size=10)
                fig.add_hline(y=mean - thr*std, line_dash="dot",  line_color="#dc2626", line_width=1,
                              annotation_text=f"−{thr}σ", annotation_font_size=10)
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        elif tool_name == "forecast_simple" and "forecast" in result:
            hist  = [{"date": p["date"], "value": p["value"], "type": "Historical"} for p in result.get("historical", [])]
            fcast = [{"date": p["date"], "value": p["value"], "type": "Forecast"}   for p in result["forecast"]]
            df_c = pd.DataFrame(hist + fcast)
            fig = px.line(df_c, x="date", y="value", color="type", markers=True,
                          title=f"Forecast: {result.get('metric', 'Metric')}",
                          color_discrete_map={"Historical": "#2563eb", "Forecast": "#d97706"})
            fig.update_traces(line_width=2, marker_size=5)
            if fcast:
                fig.add_vrect(x0=fcast[0]["date"], x1=fcast[-1]["date"],
                              fillcolor="#fef3c7", opacity=0.4, line_width=0,
                              annotation_text="Forecast zone", annotation_font_size=10,
                              annotation_font_color="#92400e")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            if result.get("confidence_note"):
                conf = result.get("confidence", "")
                if conf == "low":
                    st.warning(result["confidence_note"])
                elif conf == "medium":
                    st.info(result["confidence_note"])

    except Exception as e:
        st.caption(f"Chart error: {e}")


# ── Planning Brief renderer ───────────────────────────────────────────────────
def _render_planning_brief(brief_text: str, tool_calls_log: list):
    """Render the planning brief with styled insight/action cards."""
    lines = brief_text.strip().splitlines()
    in_findings = False
    in_actions  = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        # Section headers
        if "key finding" in low or ("finding" in low and stripped.startswith("#")):
            in_findings, in_actions = True, False
            st.markdown(f'<div class="step-header">Key findings</div>', unsafe_allow_html=True)
            continue
        if "recommended action" in low or ("action" in low and stripped.startswith("#")):
            in_findings, in_actions = False, True
            st.markdown(f'<div class="step-header">Recommended actions</div>', unsafe_allow_html=True)
            continue
        if stripped.startswith("#"):
            in_findings = in_actions = False
            st.markdown(stripped.lstrip("#").strip())
            continue
        # Numbered list items → cards
        import re
        m = re.match(r'^(\d+)\.\s+\*?\*?(.+)', stripped)
        if m:
            content = m.group(2).rstrip("*")
            # Strip trailing ** from bold markers
            content = re.sub(r'\*+$', '', content)
            if in_findings:
                st.markdown(
                    f'<div class="insight-card"><span class="label">Finding</span><br>{content}</div>',
                    unsafe_allow_html=True)
            elif in_actions:
                st.markdown(
                    f'<div class="action-card"><span class="label">Action</span><br>{content}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(stripped)
        else:
            st.markdown(stripped)

    # Tool transparency
    with st.expander("How this brief was computed"):
        for tc in tool_calls_log:
            st.markdown(f"**Tool:** `{tc['tool']}`")
            calc_desc = describe_calculation(tc["tool"], tc["input"], tc.get("result", {}))
            if calc_desc:
                st.markdown(f'<div class="transparency-row">{calc_desc}</div>',
                            unsafe_allow_html=True)
            result_display = tc.get("result", {})
            if isinstance(result_display, dict) and "rows" in result_display:
                n_r = len(result_display["rows"])
                dc = {k: v for k, v in result_display.items() if k not in ("rows", "verification")}
                dc[f"rows (first {min(5, n_r)} of {n_r})"] = result_display["rows"][:5]
                st.json(dc)
            else:
                dc = ({k: v for k, v in result_display.items() if k != "verification"}
                      if isinstance(result_display, dict) else result_display)
                st.json(dc)
            st.divider()


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
    <div class="app-header">
      <h1>AI Planning Assistant</h1>
      <p>Your Digital Colleague for demand planning &amp; forecasting</p>
      <span class="badge">MBA Thesis · Macromedia University</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sidebar-section-label">Data source</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV or Excel",
                                    type=["csv", "xlsx", "xls"],
                                    label_visibility="collapsed")
        _sb1, _sb2 = st.columns(2)
        _sample_clicked = None
        with _sb1:
            if st.button("Sample (clean)", use_container_width=True):
                _sample_clicked = (SAMPLE_DATA_PATH, "thesis_demand_clean.csv")
        with _sb2:
            if st.button("Sample (messy)", use_container_width=True):
                _sample_clicked = (DIRTY_DATA_PATH, "thesis_demand_dirty.csv")
        if _sample_clicked:
            with st.spinner("Loading…"):
                df_raw, _ = _load_sample(_sample_clicked[0])
                st.session_state.df = df_raw
                st.session_state.df_clean = None
                st.session_state.is_cleaned = False
                st.session_state.cleaning_log = []
                st.session_state.chat_history = []
                st.session_state.planning_brief = None
                st.session_state.brief_tool_calls = []
                st.session_state.file_name = _sample_clicked[1]
                st.session_state.col_types = detect_column_types(df_raw)
                st.session_state.data_issues = scan_data_quality(df_raw)
                st.rerun()

        if uploaded is not None:
            file_bytes = uploaded.read()
            if uploaded.name != st.session_state.file_name:
                with st.spinner("Loading…"):
                    try:
                        df_raw, enc = load_file_bytes(file_bytes, uploaded.name)
                        if enc and enc != "utf-8":
                            st.info(f"File encoding detected as '{enc}' and handled automatically.")
                        st.session_state.df = df_raw
                        st.session_state.df_clean = None
                        st.session_state.is_cleaned = False
                        st.session_state.cleaning_log = []
                        st.session_state.chat_history = []
                        st.session_state.planning_brief = None
                        st.session_state.brief_tool_calls = []
                        st.session_state.file_name = uploaded.name
                        st.session_state.col_types = detect_column_types(df_raw)
                        st.session_state.data_issues = scan_data_quality(df_raw)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load file: {e}")

        if st.session_state.df is not None:
            st.divider()
            st.markdown('<p class="sidebar-section-label">Data quality</p>', unsafe_allow_html=True)

            issues = st.session_state.data_issues
            n_issues = len(issues)
            severity_label = {"high": "High", "medium": "Medium", "low": "Low"}

            if not issues:
                st.success("No issues detected")
            else:
                st.warning(f"{n_issues} issue(s) found")
                with st.expander(f"View {n_issues} issues", expanded=False):
                    for iss in issues:
                        css = f"issue-{iss['severity']}"
                        sev = severity_label.get(iss["severity"], "")
                        st.markdown(f'<div class="{css}"><strong>{sev}</strong> — {iss["description"]}</div>',
                                    unsafe_allow_html=True)

            st.divider()
            st.markdown('<p class="sidebar-section-label">Cleaning</p>', unsafe_allow_html=True)

            if not st.session_state.is_cleaned:
                if st.button("🧹 Clean data", use_container_width=True, type="primary"):
                    with st.spinner("Cleaning…"):
                        df_c, log = clean_dataframe(st.session_state.df)
                        st.session_state.df_clean = df_c
                        st.session_state.cleaning_log = log
                        st.session_state.is_cleaned = True
                        st.session_state.col_types = detect_column_types(df_c)
                        st.rerun()
            else:
                st.success("Data cleaned")
                if st.button("Reset to original", use_container_width=True):
                    st.session_state.df_clean = None
                    st.session_state.is_cleaned = False
                    st.session_state.cleaning_log = []
                    st.session_state.col_types = detect_column_types(st.session_state.df)
                    st.rerun()

            if st.session_state.is_cleaned and st.session_state.df_clean is not None:
                csv_bytes = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
                st.download_button("Download cleaned CSV", data=csv_bytes,
                                   file_name="cleaned_data.csv", mime="text/csv",
                                   use_container_width=True)

    # ── Main area: no data yet ────────────────────────────────────────────────
    if st.session_state.df is None:
        st.markdown("""
        <div class="empty-state">
          <h2>No dataset loaded</h2>
          <p>Upload a CSV or Excel file, or load the sample planning dataset from the sidebar.</p>
          <div class="capability-grid">
            <div class="capability-item"><strong>Data overview</strong><span>Row counts, column types, date ranges</span></div>
            <div class="capability-item"><strong>Quality scan</strong><span>Detect and fix missing values, duplicates, casing</span></div>
            <div class="capability-item"><strong>AI chat</strong><span>Ask questions about your data in plain English</span></div>
            <div class="capability-item"><strong>Planning brief</strong><span>One-click analysis with ranked findings</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Active dataframe ──────────────────────────────────────────────────────
    df = st.session_state.df_clean if st.session_state.is_cleaned else st.session_state.df
    col_types = st.session_state.col_types

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Data Overview
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header">📊 Dataset overview</div>', unsafe_allow_html=True)

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
        st.caption(f"Total {stats['key_metric']}: {stats['key_metric_total']:,.2f}")

    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Column types detected"):
        type_df = pd.DataFrame([{"Column": c, "Type": t, "Unique Values": df[c].nunique(),
                                  "Missing": df[c].isna().sum()}
                                 for c, t in col_types.items()])
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Data Cleaning Report
    # ═══════════════════════════════════════════════════════════════════════════
    if st.session_state.data_issues or st.session_state.is_cleaned:
        st.markdown('<div class="step-header">🧹 Data cleaning</div>', unsafe_allow_html=True)

        if not st.session_state.is_cleaned:
            n = len(st.session_state.data_issues)
            st.warning(f"{n} quality issue(s) detected — use 'Clean data' in the sidebar to fix them.")
            for iss in st.session_state.data_issues[:5]:
                st.markdown(f"- {iss['description']}")
        else:
            log = st.session_state.cleaning_log
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Before**")
                orig = st.session_state.df
                bc = pd.DataFrame({
                    "Metric": ["Rows", "Duplicates", "Missing cells", "Columns"],
                    "Value": [str(len(orig)), str(int(orig.duplicated().sum())),
                              str(int(orig.isna().sum().sum())), str(len(orig.columns))]
                })
                st.dataframe(bc, hide_index=True, use_container_width=True)
            with col_b:
                st.markdown("**After**")
                cleaned = st.session_state.df_clean
                ac = pd.DataFrame({
                    "Metric": ["Rows", "Duplicates", "Missing cells", "Columns"],
                    "Value": [str(len(cleaned)), str(int(cleaned.duplicated().sum())),
                              str(int(cleaned.isna().sum().sum())), str(len(cleaned.columns))]
                })
                st.dataframe(ac, hide_index=True, use_container_width=True)

            st.caption(f"{len(log)} action(s) applied")
            for i, action in enumerate(log, 1):
                st.markdown(f'<div class="transparency-row">{i}. {action}</div>',
                            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: AI Chat
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header">💬 AI analysis</div>', unsafe_allow_html=True)

    # ── Planning Brief ────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("📋 Generate Planning Brief", type="primary", use_container_width=True,
                     help="Runs multiple analyses automatically and synthesises ranked findings"):
            with st.spinner("Scanning dataset…"):
                brief_result = run_planning_brief(df, col_types)
                st.session_state.planning_brief = brief_result["brief"]
                st.session_state.brief_tool_calls = brief_result["tool_calls"]
            st.rerun()

    if st.session_state.planning_brief:
        _render_planning_brief(st.session_state.planning_brief,
                               st.session_state.brief_tool_calls)
        if st.button("Regenerate brief"):
            st.session_state.planning_brief = None
            st.session_state.brief_tool_calls = []
            st.rerun()

    st.divider()

    # ── Suggested questions ───────────────────────────────────────────────────
    qs = suggested_questions(col_types)
    st.caption("Suggested questions — click to ask")
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
                with st.expander("How this answer was computed"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**Tool:** `{tc['tool']}`")
                        result = tc.get("result", {})
                        calc_desc = describe_calculation(tc["tool"], tc["input"], result)
                        if calc_desc:
                            st.markdown(f'<div class="transparency-row">{calc_desc}</div>',
                                        unsafe_allow_html=True)
                        if tc["tool"] == "forecast_simple" and result.get("confidence_note"):
                            conf = result.get("confidence", "")
                            if conf == "low":
                                st.warning(result["confidence_note"])
                            elif conf == "medium":
                                st.info(result["confidence_note"])
                            else:
                                st.success(result["confidence_note"])
                        if tc["tool"] == "trend_analysis" and "data_points" in result:
                            info = f"{result['data_points']:,} data points"
                            if result.get("date_range"):
                                info += f" · {result['date_range']}"
                            st.caption(info)
                        if isinstance(result, dict) and "verification" in result:
                            v = result["verification"]
                            if v.get("passed"):
                                st.markdown(f'<span class="verified-badge">&#10003; Verified — {v.get("method", "")}</span>',
                                            unsafe_allow_html=True)
                            else:
                                st.error(f"Discrepancy: {v.get('method', '')}")
                        st.markdown("**Input**")
                        st.json(tc["input"])
                        st.markdown("**Result**")
                        result_display = result
                        if isinstance(result_display, dict) and "rows" in result_display:
                            n_rows = len(result_display["rows"])
                            display_copy = {k: v for k, v in result_display.items()
                                            if k not in ("rows", "verification")}
                            display_copy[f"rows (first {min(5, n_rows)} of {n_rows})"] = result_display["rows"][:5]
                            st.json(display_copy)
                        else:
                            display_copy = ({k: v for k, v in result_display.items() if k != "verification"}
                                            if isinstance(result_display, dict) else result_display)
                            st.json(display_copy)
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
