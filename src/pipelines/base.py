"""
base.py — Extracción Universal de Features (72 features fijas)
Convierte CUALQUIER tabla BigQuery en un Day-Vector de dimensión fija.

Changelog v8:
  - G3b: Estabilidad temporal categórica (2 features)
  - G3c: Distribución numérica (5 features)
  - G3d: Texto libre (2 features)
  - G6:  coh_null_x_volume_inverse (1 feature)
  - Unknown values parametrizables via cfg["bq"]["unknown_values"]
  - Warmup lag-7: primeros 6 días explícitamente sin referencia
  - fillna(0.0) con logging de features afectadas
"""

import pickle, logging, warnings
from typing import List, Tuple, Dict, Set
import numpy as np, pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

UNIVERSAL_FEATURE_NAMES: List[str] = [
    # G1: Calidad (15)
    "qual_mean_pct_null", "qual_std_pct_null", "qual_p25_pct_null",
    "qual_p75_pct_null", "qual_p95_pct_null", "qual_max_pct_null",
    "qual_gini_pct_null", "qual_pct_cols_over10_null", "qual_pct_cols_fully_null",
    "qual_mean_pct_empty", "qual_max_pct_empty",
    "qual_mean_pct_unknown", "qual_max_pct_unknown",
    "qual_n_cols_any_null", "qual_pct_cols_zero_null",
    # G2: Volumen (3)
    "vol_log_row_count", "vol_row_count_raw", "vol_row_count_7d_slope",
    # G3: Distribución categórica (13)
    "dist_mean_entropy_cat", "dist_std_entropy_cat", "dist_p25_entropy_cat",
    "dist_p75_entropy_cat", "dist_min_entropy_cat",
    "dist_mean_hhi", "dist_max_hhi", "dist_p25_hhi", "dist_p75_hhi",
    "dist_mean_top1_share", "dist_max_top1_share",
    "dist_mean_n_cats", "dist_std_n_cats",
    # G3b: Estabilidad temporal categórica (2) — dinámicas, calculadas en build_daily_features
    "dist_mean_jaccard_top10_7d", "dist_pct_cols_top1_changed",
    # G3c: Distribución numérica (5)
    "num_mean_cv", "num_mean_skewness", "num_mean_kurtosis",
    "num_mean_iqr_norm", "num_pct_cols_with_outliers",
    # G3d: Texto libre (2)
    "text_mean_strlen", "text_std_strlen",
    # G4: Dinámica (15)
    "dyn_row_count_delta_pct",
    "dyn_n_new_cat_values", "dyn_n_disappeared_cat_values",
    "dyn_pct_new_cat_values", "dyn_pct_disappeared_cat_values",
    "dyn_delta_mean_pct_null", "dyn_delta_max_pct_null", "dyn_delta_gini_pct_null",
    "dyn_delta_mean_entropy_cat", "dyn_delta_mean_hhi", "dyn_delta_mean_top1_share",
    "dyn_delta_mean_pct_empty", "dyn_delta_mean_pct_unknown", "dyn_delta_n_cats",
    "dyn_delta_pct_cols_degraded",
    # G5: Esquema (3)
    "schema_n_total_cols", "schema_n_cat_cols", "schema_n_new_cols",
    # G6: Coherencia (10)
    "coh_null_x_volume", "coh_null_x_volume_inverse",
    "coh_null_entropy", "coh_pct_cols_degraded",
    "coh_quality_score", "coh_distribution_drift", "coh_volume_quality_ratio",
    "coh_null_spread_ratio", "coh_cat_stability_score", "coh_null_volume_trend",
    # G7: Temporal (4)
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
]
assert len(UNIVERSAL_FEATURE_NAMES) == 72, f"Expected 72, got {len(UNIVERSAL_FEATURE_NAMES)}"

_CYCLIC_COLS  = {"day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"}
_RAW_VOL_COLS = {"vol_row_count_raw"}
_DEFAULT_UNKNOWN_VALS = {"UNKNOWN", "N/A", "NULL", "NONE", ""}
_LAG_S = 7  # Seasonal lag (days)
_COL_TYPE_CACHE = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entropy_norm(s: pd.Series) -> float:
    vc = s.value_counts(normalize=True).values
    if len(vc) <= 1: return 0.0
    raw = float(-np.sum(vc * np.log(vc + 1e-12)))
    return raw / np.log(len(vc))

def _hhi(s: pd.Series) -> float:
    if s.empty: return 1.0
    return float((s.value_counts(normalize=True).values ** 2).sum())

def _gini(arr: np.ndarray) -> float:
    a = np.sort(np.abs(arr.astype(float)))
    n = len(a)
    if n == 0 or a.sum() == 0: return 0.0
    return float((2 * (np.arange(1, n+1) * a).sum()) / (n * a.sum()) - (n+1) / n)

def _detect_cat_cols(df: pd.DataFrame, date_col: str) -> List[str]:
    return [c for c in df.select_dtypes(include=["object","category","bool"]).columns
            if c != date_col and df[c].nunique() <= 500]

def _detect_num_cols(df: pd.DataFrame, date_col: str) -> List[str]:
    """Columnas numéricas (excluyendo booleanas y date)."""
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c != date_col and not pd.api.types.is_bool_dtype(df[c])]

def _detect_freetext_cols(df: pd.DataFrame, date_col: str) -> List[str]:
    """Columnas string con alta cardinalidad (>500 únicos) — texto libre."""
    return [c for c in df.select_dtypes(include=["object"]).columns
            if c != date_col and df[c].nunique() > 500]

def _get_unknown_vals(cfg: dict = None) -> set:
    """Unknown values from cfg with fallback to defaults."""
    if cfg is None:
        return _DEFAULT_UNKNOWN_VALS
    custom = cfg.get("bq", {}).get("unknown_values")
    if custom is not None:
        return set(v.upper() for v in custom)
    return _DEFAULT_UNKNOWN_VALS

def _pcts(arr, percentiles=(25,75,95)):
    return {p: float(np.percentile(arr, p)) for p in percentiles}


# ── BQ Download ───────────────────────────────────────────────────────────────

def build_extraction_query(cfg_bq: dict) -> str:
    date_col = cfg_bq["date_col"]
    fqn = f"`{cfg_bq['project_id']}.{cfg_bq['dataset']}.{cfg_bq['table']}`"
    exc = cfg_bq.get("exclude_cols", [])
    select = f"* EXCEPT({', '.join(exc)})" if exc else "*"
    where = f"WHERE {date_col} BETWEEN '{cfg_bq.get('start_date','2021-01-01')}' AND '{cfg_bq.get('end_date','2025-12-31')}'"
    h, pct = cfg_bq.get("sample_hash_col"), cfg_bq.get("sample_pct", 100)
    if h and pct < 100:
        where += f" AND MOD(ABS(FARM_FINGERPRINT(CAST({h} AS STRING))), 100) < {pct}"
    return f"SELECT {select} FROM {fqn} {where} ORDER BY {date_col}"

def download_from_bigquery(cfg: dict) -> pd.DataFrame:
    from google.cloud import bigquery
    raw_path = cfg["paths"]["raw_data"]
    if raw_path.exists() and not cfg.get("force_download", False):
        log.info(f"[CACHÉ] {raw_path}")
        return pd.read_parquet(raw_path)
    client = bigquery.Client(project=cfg["bq"]["project_id"])
    df = client.query(build_extraction_query(cfg["bq"])).to_dataframe(progress_bar_type="tqdm")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)
    log.info(f"[BQ] {len(df):,} filas → {raw_path}")
    return df


# ── Day-Vector ────────────────────────────────────────────────────────────────

def compute_day_vector(day_df: pd.DataFrame, date_col: str, cfg: dict = None) -> Tuple[Dict, Dict[str, Set], Dict[str, Set], Dict[str, str]]:
    """
    Calcula el vector estático de un día.

    Returns:
        vec:            dict de features estáticas
        cat_value_sets: {col: set(all values)}        — para G4 new/disappeared
        cat_top10_sets: {col: set(top 10 values)}     — para G3b Jaccard
        cat_top1_vals:  {col: top_1_value_as_str}     — para G3b top1_changed
    """
    
    # Excluir columnas configuradas (por si raw_df viene de caché pre-exclusión)
    _exc = set(cfg.get("bq", {}).get("exclude_cols", [])) if cfg else set()
    if _exc:
        day_df = day_df.drop(columns=[c for c in _exc if c in day_df.columns])

    n_rows = len(day_df)
    vec: Dict = {}
    unknown_vals = _get_unknown_vals(cfg)

    # Filtrar columnas REPEATED/STRUCT (contienen arrays no escalares)
    _cache_key = tuple(c for c in day_df.columns if c != date_col)

    if _cache_key in _COL_TYPE_CACHE:
        all_cols, cat_cols, num_cols, text_cols = _COL_TYPE_CACHE[_cache_key]
    else:
        all_cols = []
        for c in day_df.columns:
            if c == date_col:
                continue
            try:
                day_df[c].nunique()
                all_cols.append(c)
            except TypeError:
                pass

        cat_cols  = [c for c in all_cols
                     if day_df[c].dtype in ("object", "category", "bool")
                     and day_df[c].nunique() <= 500]
        num_cols  = [c for c in all_cols
                     if pd.api.types.is_numeric_dtype(day_df[c])]
        text_cols = [c for c in all_cols
                     if day_df[c].dtype == "object"
                     and day_df[c].nunique() > 500]

        _COL_TYPE_CACHE[_cache_key] = (all_cols, cat_cols, num_cols, text_cols)
                                      
    n_cols, n_cat = len(all_cols), len(cat_cols)

    # G1: Calidad — robusto ante columnas object no-string
    null_arr  = np.array([day_df[c].isna().mean() for c in all_cols])
    empty_arr = np.array([(day_df[c]=="").mean() if day_df[c].dtype == object else 0. for c in all_cols])

    _unk = []
    for c in all_cols:
        if day_df[c].dtype != object:
            _unk.append(0.)
        else:
            try:
                _unk.append(float(day_df[c].str.upper().isin(unknown_vals).mean()))
            except (AttributeError, TypeError):
                _unk.append(0.)
    unk_arr = np.array(_unk)

    if n_cols > 0:
        vec.update({
            "qual_mean_pct_null": float(null_arr.mean()),
            "qual_std_pct_null":  float(null_arr.std()),
            "qual_p25_pct_null":  float(np.percentile(null_arr, 25)),
            "qual_p75_pct_null":  float(np.percentile(null_arr, 75)),
            "qual_p95_pct_null":  float(np.percentile(null_arr, 95)),
            "qual_max_pct_null":  float(null_arr.max()),
            "qual_gini_pct_null": _gini(null_arr),
            "qual_pct_cols_over10_null": float((null_arr > 0.10).mean()),
            "qual_pct_cols_fully_null":  float((null_arr >= 1.0).mean()),
            "qual_mean_pct_empty":   float(empty_arr.mean()),
            "qual_max_pct_empty":    float(empty_arr.max()),
            "qual_mean_pct_unknown": float(unk_arr.mean()),
            "qual_max_pct_unknown":  float(unk_arr.max()),
            "qual_n_cols_any_null":  float((null_arr > 0).sum()),
            "qual_pct_cols_zero_null": float((null_arr == 0).mean()),
        })
    else:
        vec.update({f: 0.0 for f in UNIVERSAL_FEATURE_NAMES if f.startswith("qual_")})

    # G2: Volumen
    vec["vol_log_row_count"] = float(np.log1p(n_rows))
    vec["vol_row_count_raw"] = float(n_rows)

    # G3: Distribución categórica + metadata para G3b
    cat_value_sets: Dict[str, Set] = {}
    cat_top10_sets: Dict[str, Set] = {}
    cat_top1_vals:  Dict[str, str] = {}

    if n_cat > 0:
        e, h, t1, nc = [], [], [], []
        for col in cat_cols:
            clean = day_df[col].dropna()
            if clean.empty:
                e.append(0.); h.append(1.); t1.append(1.); nc.append(0)
                cat_value_sets[col] = set()
                cat_top10_sets[col] = set()
            else:
                vc = clean.value_counts(normalize=True)
                e.append(_entropy_norm(clean)); h.append(_hhi(clean))
                t1.append(float(vc.iloc[0])); nc.append(int(clean.nunique()))
                # Metadata para G3b y G4
                cat_value_sets[col] = set(clean.unique())
                vc_raw = clean.value_counts()
                cat_top10_sets[col] = set(vc_raw.head(10).index)
                cat_top1_vals[col]  = str(vc_raw.index[0])
        e, h, nc = np.array(e), np.array(h), np.array(nc)
        vec.update({
            "dist_mean_entropy_cat": float(e.mean()), "dist_std_entropy_cat": float(e.std()),
            "dist_p25_entropy_cat": float(np.percentile(e,25)), "dist_p75_entropy_cat": float(np.percentile(e,75)),
            "dist_min_entropy_cat": float(e.min()),
            "dist_mean_hhi": float(h.mean()), "dist_max_hhi": float(h.max()),
            "dist_p25_hhi": float(np.percentile(h,25)), "dist_p75_hhi": float(np.percentile(h,75)),
            "dist_mean_top1_share": float(np.mean(t1)), "dist_max_top1_share": float(np.max(t1)),
            "dist_mean_n_cats": float(nc.mean()), "dist_std_n_cats": float(nc.std()),
        })
    else:
        vec.update({f: 0.0 for f in UNIVERSAL_FEATURE_NAMES if f.startswith("dist_")})

    # G3c: Distribución numérica
    if num_cols:
        cvs, skews, kurts, iqrs, outlier_flags = [], [], [], [], []
        for col in num_cols:
            vals = day_df[col].dropna().values.astype(float)
            if len(vals) < 2:
                continue
            m, s = float(vals.mean()), float(vals.std())
            cvs.append(s / abs(m) if abs(m) > 1e-10 else 0.0)
            skews.append(float(pd.Series(vals).skew()))
            kurts.append(float(pd.Series(vals).kurtosis()))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            med = float(np.median(vals))
            iqrs.append(iqr / abs(med) if abs(med) > 1e-10 else 0.0)
            outlier_flags.append(1.0 if (s > 1e-10 and float((np.abs(vals - m) > 3*s).mean()) > 0) else 0.0)

        if cvs:
            vec["num_mean_cv"]                = float(np.mean(cvs))
            vec["num_mean_skewness"]          = float(np.nanmean(skews))
            vec["num_mean_kurtosis"]          = float(np.nanmean(kurts))
            vec["num_mean_iqr_norm"]          = float(np.mean(iqrs))
            vec["num_pct_cols_with_outliers"] = float(np.mean(outlier_flags))
        else:
            for f in UNIVERSAL_FEATURE_NAMES:
                if f.startswith("num_"): vec[f] = 0.0
    else:
        for f in UNIVERSAL_FEATURE_NAMES:
            if f.startswith("num_"): vec[f] = 0.0

    # G3d: Texto libre
    if text_cols:
        mean_lens, std_lens = [], []
        for col in text_cols:
            lens = day_df[col].dropna().astype(str).str.len()
            if not lens.empty:
                mean_lens.append(float(lens.mean()))
                std_lens.append(float(lens.std()))
        vec["text_mean_strlen"] = float(np.mean(mean_lens)) if mean_lens else 0.0
        vec["text_std_strlen"]  = float(np.mean(std_lens))  if std_lens  else 0.0
    else:
        vec["text_mean_strlen"] = 0.0
        vec["text_std_strlen"]  = 0.0

    # G5: Esquema
    vec["schema_n_total_cols"] = float(n_cols)
    vec["schema_n_cat_cols"]   = float(n_cat)

    # G6: Coherencia (parcial — el resto en build_daily_features)
    mn, mx = vec.get("qual_mean_pct_null",0), vec.get("qual_max_pct_null",0)
    vec["coh_null_entropy"] = float(scipy_entropy(null_arr/null_arr.sum()+1e-10)) if n_cols>0 and null_arr.sum()>0 else 0.0
    vec["coh_quality_score"] = 0.4*mn + 0.3*vec.get("qual_mean_pct_empty",0) + 0.2*vec.get("qual_mean_pct_unknown",0) + 0.1*vec.get("qual_pct_cols_fully_null",0)
    vec["coh_null_spread_ratio"] = float(mx/mn) if mn > 1e-6 else 1.0
    vec["coh_cat_stability_score"] = float(vec.get("dist_mean_entropy_cat",0) * (1-vec.get("dist_mean_hhi",0))) if n_cat>0 else 0.0

    return vec, cat_value_sets, cat_top10_sets, cat_top1_vals


# ── Dataset completo con dinámicas vectorizadas ──────────────────────────────

def build_daily_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    date_col = cfg["bq"]["date_col"]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_date"] = df[date_col].dt.date
    dates_sorted = sorted(df["_date"].unique())
    n_days = len(dates_sorted)

    # Fase 1: vectores estáticos por día
    static_vecs, cat_sets_list, top10_sets_list, top1_vals_list, cols_per_day = [], [], [], [], []
    for date in dates_sorted:
        day_df = df[df["_date"] == date].drop(columns=["_date"])
        vec, cat_sets, top10_sets, top1_vals = compute_day_vector(day_df, date_col, cfg)
        vec["date"] = pd.Timestamp(date)
        static_vecs.append(vec)
        cat_sets_list.append(cat_sets)
        top10_sets_list.append(top10_sets)
        top1_vals_list.append(top1_vals)
        cols_per_day.append(set(c for c in day_df.columns if c != date_col))

    fdf = pd.DataFrame(static_vecs).sort_values("date").reset_index(drop=True)

    # G2: tendencia 7d (vectorizada)
    rc = fdf["vol_row_count_raw"].values
    slope = np.zeros(n_days)
    for i in range(1, n_days):
        w = rc[max(0,i-6):i+1]
        if len(w) >= 2: slope[i] = float(np.polyfit(np.arange(len(w),dtype=float), np.log1p(w), 1)[0])
    fdf["vol_row_count_7d_slope"] = slope

    # G3b: Estabilidad temporal categórica (lag-7)
    jaccard_arr = np.zeros(n_days)
    top1_changed_arr = np.zeros(n_days)
    for i in range(_LAG_S, n_days):  # warmup: primeros 6 días = 0.0
        ref = i - _LAG_S
        t10_today, t10_ref = top10_sets_list[i], top10_sets_list[ref]
        t1_today,  t1_ref  = top1_vals_list[i],  top1_vals_list[ref]
        common_cols = set(t10_today) & set(t10_ref)
        if common_cols:
            jaccards = []
            n_changed = 0
            for col in common_cols:
                s_today, s_ref = t10_today[col], t10_ref[col]
                union = s_today | s_ref
                jaccards.append(len(s_today & s_ref) / len(union) if union else 1.0)
                if t1_today.get(col) != t1_ref.get(col):
                    n_changed += 1
            jaccard_arr[i] = float(np.mean(jaccards))
            top1_changed_arr[i] = n_changed / len(common_cols)
    fdf["dist_mean_jaccard_top10_7d"] = jaccard_arr
    fdf["dist_pct_cols_top1_changed"] = top1_changed_arr

    # G4: deltas lag-7 (vectorizadas con shift)
    _DELTA_MAP = {
        "dyn_delta_mean_pct_null":    "qual_mean_pct_null",
        "dyn_delta_max_pct_null":     "qual_max_pct_null",
        "dyn_delta_gini_pct_null":    "qual_gini_pct_null",
        "dyn_delta_mean_entropy_cat": "dist_mean_entropy_cat",
        "dyn_delta_mean_hhi":         "dist_mean_hhi",
        "dyn_delta_mean_top1_share":  "dist_mean_top1_share",
        "dyn_delta_mean_pct_empty":   "qual_mean_pct_empty",
        "dyn_delta_mean_pct_unknown": "qual_mean_pct_unknown",
        "dyn_delta_n_cats":           "dist_mean_n_cats",
    }
    for dyn_col, src_col in _DELTA_MAP.items():
        fdf[dyn_col] = (fdf[src_col] - fdf[src_col].shift(_LAG_S)).fillna(0.0)

    # row_count_delta_pct: lag-1 (intraday pipeline failures)
    prev_rc = fdf["vol_row_count_raw"].shift(1).bfill()
    fdf["dyn_row_count_delta_pct"] = ((fdf["vol_row_count_raw"] - prev_rc) / prev_rc.clip(lower=1)).fillna(0.0)

    # Categóricos: new/disappeared — con warmup explícito
    new_v, dis_v, pct_new, pct_dis = [np.zeros(n_days) for _ in range(4)]
    for i in range(1, n_days):
        if i < _LAG_S:
            continue  # warmup: sin referencia lag-7 válida, deja en 0.0
        ref = i - _LAG_S
        ts, ps = cat_sets_list[i], cat_sets_list[ref]
        new_tot = dis_tot = prev_tot = 0
        for col in set(ts) & set(ps):
            new_tot += len(ts[col] - ps[col])
            dis_tot += len(ps[col] - ts[col])
            prev_tot += len(ps[col])
        new_v[i], dis_v[i] = new_tot, dis_tot
        pct_new[i] = new_tot / prev_tot if prev_tot else 0.
        pct_dis[i] = dis_tot / prev_tot if prev_tot else 0.
    fdf["dyn_n_new_cat_values"] = new_v
    fdf["dyn_n_disappeared_cat_values"] = dis_v
    fdf["dyn_pct_new_cat_values"] = pct_new
    fdf["dyn_pct_disappeared_cat_values"] = pct_dis

    # G5: columnas nuevas
    fdf["schema_n_new_cols"] = [0.] + [float(len(cols_per_day[i]-cols_per_day[i-1])) for i in range(1, n_days)]

    # G6: coherencia (vectorizada)
    null_delta = fdf["dyn_delta_mean_pct_null"]
    vol_delta  = fdf["dyn_row_count_delta_pct"]
    fdf["coh_null_x_volume"]         = np.clip(null_delta * (-vol_delta), 0, None)
    fdf["coh_null_x_volume_inverse"] = np.clip((-null_delta) * vol_delta, 0, None)

    fdf["coh_pct_cols_degraded"] = (
        (fdf["dyn_delta_mean_pct_null"]>0.01).astype(float) +
        (fdf["dyn_delta_mean_pct_empty"]>0.01).astype(float) +
        (fdf["dyn_delta_mean_pct_unknown"]>0.01).astype(float)
    ) / 3.0
    fdf["coh_distribution_drift"] = fdf["dyn_delta_mean_entropy_cat"].abs() + fdf["dyn_delta_mean_hhi"].abs() + fdf["dyn_delta_mean_top1_share"].abs()

    log_vol = np.log1p(fdf["vol_row_count_raw"].values)
    fdf["coh_volume_quality_ratio"] = (1.0 - fdf["coh_quality_score"]) * (log_vol / max(log_vol.max(), 1.0))
    fdf["dyn_delta_pct_cols_degraded"] = (fdf["coh_pct_cols_degraded"] - fdf["coh_pct_cols_degraded"].shift(_LAG_S)).fillna(0.0)

    null_vals = fdf["qual_mean_pct_null"].values
    null_trend = np.zeros(n_days)
    for i in range(1, n_days):
        wn = null_vals[max(0,i-6):i+1]
        wv = log_vol[max(0,i-6):i+1]
        if len(wn) >= 3 and wv.std() > 1e-6:
            null_trend[i] = float(np.polyfit(np.arange(len(wn)), wn, 1)[0])
    fdf["coh_null_volume_trend"] = null_trend

    # G7: temporal
    dow, mon = fdf["date"].dt.dayofweek, fdf["date"].dt.month
    fdf["day_of_week_sin"] = np.sin(2*np.pi*dow/7)
    fdf["day_of_week_cos"] = np.cos(2*np.pi*dow/7)
    fdf["month_sin"] = np.sin(2*np.pi*(mon-1)/12)
    fdf["month_cos"] = np.cos(2*np.pi*(mon-1)/12)

    # Rellenar features faltantes + logging de NaN
    for f in UNIVERSAL_FEATURE_NAMES:
        if f not in fdf.columns: fdf[f] = 0.0

    nan_counts = fdf[UNIVERSAL_FEATURE_NAMES].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if not nan_features.empty:
        log.warning(f"[FEATURES] Features con NaN rellenados a 0: {dict(nan_features)}")
        max_nan_pct = nan_features.max() / len(fdf)
        if max_nan_pct > 0.10:
            log.error(f"[FEATURES] >10% NaN en alguna feature — posible bug en el cálculo")

    fdf = fdf[["date"] + UNIVERSAL_FEATURE_NAMES].fillna(0.0)
    log.info(f"[FEATURES] {len(fdf)} días × {len(UNIVERSAL_FEATURE_NAMES)} features")
    return fdf


# ── Escalado ──────────────────────────────────────────────────────────────────

def identify_feature_groups(feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    g1 = [c for c in feature_cols if c.startswith("qual_") and c not in _CYCLIC_COLS]
    g2 = [c for c in feature_cols if not c.startswith("qual_") and c not in _CYCLIC_COLS]
    return g1, g2

def fit_and_scale(features_df: pd.DataFrame, cfg: dict, train_mask: pd.Series, force=False) -> Tuple[pd.DataFrame, List[str]]:
    sq_path, ss_path = cfg["paths"]["scaler_quality"], cfg["paths"]["scaler_structural"]
    feat_cols = [c for c in features_df.columns if c != "date"]
    g1, g2 = identify_feature_groups(feat_cols)
    MIN_IQR = 0.05

    if sq_path.exists() and ss_path.exists() and not force:
        with open(sq_path,"rb") as f: sq = pickle.load(f)
        with open(ss_path,"rb") as f: ss = pickle.load(f)
    else:
        train_df = features_df[train_mask.values].copy()
        vol_g2 = [c for c in g2 if c in _RAW_VOL_COLS]
        if vol_g2: train_df[vol_g2] = np.log1p(train_df[vol_g2].clip(lower=0))
        sq = RobustScaler(); ss = RobustScaler()
        if g1: sq.fit(train_df[g1]); sq.scale_ = np.maximum(sq.scale_, MIN_IQR)
        if g2: ss.fit(train_df[g2]); ss.scale_ = np.maximum(ss.scale_, MIN_IQR)
        sq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sq_path,"wb") as f: pickle.dump(sq, f)
        with open(ss_path,"wb") as f: pickle.dump(ss, f)

    scaled = features_df[["date"]].copy()
    if g1: scaled[g1] = sq.transform(features_df[g1])
    if g2:
        g2_in = features_df[g2].copy()
        vol = [c for c in g2 if c in _RAW_VOL_COLS]
        if vol: g2_in[vol] = np.log1p(g2_in[vol].clip(lower=0))
        scaled[g2] = ss.transform(g2_in)
    for c in _CYCLIC_COLS:
        if c in features_df.columns: scaled[c] = features_df[c].values
    clip_cols = [c for c in scaled.columns if c != "date" and c not in _CYCLIC_COLS]
    scaled[clip_cols] = scaled[clip_cols].clip(-5.0, 5.0)
    return scaled, g1 + g2 + [c for c in _CYCLIC_COLS if c in scaled.columns]


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_feature_pipeline(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    scaled_path = cfg["paths"]["scaled_data"]
    raw_path    = cfg["paths"].get("raw_data")
    force = cfg.get("force_download", False) or cfg.get("force_features", False)

    # 1. raw_df siempre disponible (necesario para Tier 1/2 y T4)
    if raw_path and raw_path.exists() and not force:
        raw_df = pd.read_parquet(raw_path)
    else:
        raw_df = download_from_bigquery(cfg)
        if raw_path:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_parquet(raw_path, index=False)

    # 2. scaled_df desde caché si existe y no force
    if scaled_path.exists() and not force:
        scaled_df = pd.read_parquet(scaled_path)
        feat_cols = [c for c in scaled_df.columns if c != "date"]
        return raw_df, scaled_df, feat_cols

    # 3. Recomputar features + escalado
    features_df = build_daily_features(raw_df, cfg)
    n_train = int(len(features_df) * cfg.get("tranad", {}).get("train_split", 0.95))
    train_mask = pd.Series([True]*n_train + [False]*(len(features_df)-n_train))
    scaled_df, feat_cols = fit_and_scale(features_df, cfg, train_mask, force=force)
    scaled_df.to_parquet(scaled_path, index=False)

    return raw_df, scaled_df, feat_cols


def describe_day_vector(features_df: pd.DataFrame, date: str) -> None:
    row = features_df[features_df["date"] == pd.Timestamp(date)]
    if row.empty: print(f"Fecha {date} no encontrada."); return
    prefixes = {"qual_":"G1 Calidad","vol_":"G2 Volumen","dist_":"G3 Distribución",
                "num_":"G3c Numérica","text_":"G3d Texto","dyn_":"G4 Dinámica",
                "schema_":"G5 Esquema","coh_":"G6 Coherencia"}
    print(f"\n{'='*62}\n  Day-Vector — {date}\n{'='*62}")
    for pfx, label in prefixes.items():
        cols = [f for f in UNIVERSAL_FEATURE_NAMES if f.startswith(pfx)]
        if cols:
            print(f"\n  {label}")
            for c in cols: print(f"    {c:<44} {row[c].values[0]:.6f}")
