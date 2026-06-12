"""
pipelines/discounts.py — Pipeline para semantic_discounts
==========================================================
La tabla de descuentos no tiene date_col: cada fila es un descuento
con init_date y max_end_date. Para generar day-vectors, reconstruimos
"snapshots diarios" filtrando los descuentos activos en cada fecha D:
  WHERE init_date <= D AND max_end_date >= D

Flujo:
  1. download_discounts_raw()     → BQ → parquet (~3.8M filas al 10%)
  2. build_all_from_snapshots()   → para cada día D, filtra activos y calcula:
       a) Day-vector (62 features universales) → input TranAD (Tier 3)
       b) Per-column stats (long format)       → input Tier 1 + Tier 2
  3. fit_and_scale()              → RobustScaler (reutiliza base.py)
  4. run_pipeline()               → orquesta todo

Un solo pase sobre los ~2000 días: ~20-30 min la primera vez, cacheado después.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

from pipelines.base import (
    compute_day_vector,
    fit_and_scale,
    UNIVERSAL_FEATURE_NAMES,
)
from core.column_attribution import (
    _stats_for_one_column,
    _STAT_COLS,
    _per_column_stats_path,
)
from config import CFG_DISCOUNTS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGA
# ─────────────────────────────────────────────────────────────────────────────

def _build_download_query(cfg_bq: dict) -> str:
    exc = cfg_bq.get("exclude_cols", [])
    except_clause = f"* EXCEPT({', '.join(exc)})" if exc else "*"

    hash_col = cfg_bq.get("sample_hash_col")
    pct      = cfg_bq.get("sample_pct", 100)
    sample_clause = (
        f"WHERE MOD(ABS(FARM_FINGERPRINT(CAST({hash_col} AS STRING))), 100) < {pct}"
        if hash_col and pct < 100 else ""
    )

    return f"""
    SELECT {except_clause}
    FROM `{cfg_bq['project_id']}.{cfg_bq['dataset']}.{cfg_bq['table']}`
    {sample_clause}
    """


def download_discounts_raw(cfg: dict) -> pd.DataFrame:
    from google.cloud import bigquery

    raw_path = cfg["paths"]["raw_data"]
    if raw_path.exists() and not cfg.get("force_download", False):
        log.info(f"[CACHÉ] {raw_path}")
        return pd.read_parquet(raw_path)

    log.info("[BQ] Descargando semantic_discounts...")
    client = bigquery.Client(project=cfg["bq"]["project_id"])
    df = client.query(_build_download_query(cfg["bq"])).to_dataframe(
        progress_bar_type="tqdm",
    )

    for col in ("init_date", "end_date_planned", "min_end_date", "max_end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)
    log.info(f"[BQ] {len(df):,} filas → {raw_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECONSTRUCCIÓN DE SNAPSHOTS → features + per_col_stats
# ─────────────────────────────────────────────────────────────────────────────

def _generate_date_range(raw_df: pd.DataFrame, cfg: dict) -> pd.DatetimeIndex:
    bq       = cfg["bq"]
    init_col = bq.get("init_date_col", "init_date")
    end_col  = bq.get("end_date_col", "max_end_date")

    data_min = raw_df[init_col].min()
    data_max_obs = raw_df[init_col].max()

    cfg_start = pd.Timestamp(bq.get("start_date", "2020-01-01"))
    cfg_end   = pd.Timestamp(bq.get("end_date", "2025-12-31"))

    start = max(cfg_start, data_min)
    end   = min(cfg_end, data_max_obs, pd.Timestamp("today"))

    dates = pd.date_range(start, end, freq="D")
    log.info(f"[SNAPSHOTS] Rango: {start.date()} → {end.date()} = {len(dates)} días")
    return dates


def _meta_columns(cfg: dict) -> set:
    """Columnas a excluir del cómputo (fechas, PKs, snapshot)."""
    bq   = cfg.get("bq", {})
    meta = {"snapshot_date"}
    for k in ("init_date_col", "end_date_col", "snapshot_col",
              "pk_col", "sample_hash_col"):
        v = bq.get(k)
        if v:
            meta.add(v)
    meta.update({"end_date_planned", "min_end_date"})
    return meta


def build_all_from_snapshots(
    raw_df: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruye snapshots diarios. Calcula en un solo pase:
      a) Day-vector (62 features) → input TranAD (Tier 3)
      b) Per-column stats (long)  → input Tier 1 + Tier 2

    Returns: (features_df, per_col_df)
    """
    bq       = cfg["bq"]
    init_col = bq.get("init_date_col", "init_date")
    end_col  = bq.get("end_date_col", "max_end_date")
    dates    = _generate_date_range(raw_df, cfg)
    n_days   = len(dates)
    meta     = _meta_columns(cfg)

    init_vals = raw_df[init_col].values.astype("datetime64[D]")
    # NaT en end_col = "descuento indefinido / sin fecha de fin" → tratar como
    # siempre activo rellenando con fecha futura lejana antes de castear.
    # Sin esto, `NaT >= d` es False y los activos indefinidos se pierden del
    # snapshot, dejando n_active=0 para toda ventana reciente.
    end_filled = raw_df[end_col].fillna(pd.Timestamp("2099-12-31"))
    end_vals   = end_filled.values.astype("datetime64[D]")

    analysis_cols = [c for c in raw_df.columns if c not in meta]

    # Aggregator para columnas ARRAY<STRUCT> (si cfg lo declara).
    # Carga el ya entrenado por row_level_pipeline; si no existe, fit on-the-fly.
    aggregator = None
    array_cols = []
    if cfg.get("array_struct_features"):
        from pathlib import Path
        from core.array_struct_aggregator import ArrayStructAggregator
        paths = cfg.get("paths", {})
        agg_path = paths.get("array_struct_aggregator")
        if agg_path is None:
            enc = paths.get("row_level_encoder")
            if enc:
                agg_path = Path(enc).parent / "array_struct_aggregator.pkl"
        if agg_path and Path(agg_path).exists():
            aggregator = ArrayStructAggregator.load(Path(agg_path))
        else:
            log.info("[SNAPSHOTS] Aggregator no persistido — fit on-the-fly")
            aggregator = ArrayStructAggregator().fit(raw_df, cfg)
        array_cols = aggregator.list_array_cols()
        # Las columnas array salen del loop normal (las trata mal _stats_for_one_column)
        analysis_cols = [c for c in analysis_cols if c not in array_cols]

    static_vecs   = []
    cat_sets_list = []
    cols_per_day  = []
    per_col_rows  = []

    for i, date in enumerate(dates):
        d_np     = np.datetime64(date, "D")
        mask     = (init_vals <= d_np) & (end_vals >= d_np)
        n_active = int(mask.sum())

        if n_active == 0:
            vec = {f: 0.0 for f in UNIVERSAL_FEATURE_NAMES}
            vec["date"] = date
            vec["vol_row_count_raw"] = 0.0
            vec["vol_log_row_count"] = 0.0
            static_vecs.append(vec)
            cat_sets_list.append({})
            cols_per_day.append(set())
            continue

        # day_df_full incluye TODAS las columnas (también las array para el aggregator);
        # day_df solo las analysis_cols (sin arrays) para los stats normales.
        day_df_full = raw_df.loc[mask]

        # Submuestreo: >50K filas no mejoran los estadísticos
        if len(day_df_full) > 50_000:
            day_df_full = day_df_full.sample(50_000, random_state=42)
            n_active = 50_000

        day_df = day_df_full[analysis_cols]

        # A) Day-vector (62 features)
        day_with_date = day_df.copy()
        day_with_date["snapshot_date"] = date
        vec, cat_sets, *_ = compute_day_vector(day_with_date, "snapshot_date")
        vec["date"] = date
        static_vecs.append(vec)
        cat_sets_list.append(cat_sets)
        cols_per_day.append(set(analysis_cols))

        # B) Per-column stats (Tier 1 + 2)
        for col in analysis_cols:
            stats = _stats_for_one_column(day_df[col], n_active)
            stats["date"]   = date
            stats["column"] = col
            per_col_rows.append(stats)

        # B-bis) Pseudo-columnas de array-struct (funcional → entropy/hhi/top1...)
        if aggregator is not None:
            for s in aggregator.compute_day_stats(day_df_full):
                col = s.pop("column")
                s["date"], s["column"] = date, col
                per_col_rows.append(s)

        if (i + 1) % 100 == 0:
            log.info(f"[SNAPSHOTS] {i+1}/{n_days} días ({n_active:,} activos)")

    log.info(f"[SNAPSHOTS] {n_days} snapshots completados")

    # Ensamblar per_col_df
    per_col_df = pd.DataFrame(per_col_rows)
    ordered = ["date", "column"] + _STAT_COLS
    per_col_df = per_col_df[ordered].sort_values(["date", "column"]).reset_index(drop=True)

    # Ensamblar features_df con G2/G4/G5/G6/G7
    features_df = _assemble_dynamic_features(
        pd.DataFrame(static_vecs), cat_sets_list, cols_per_day, n_days,
    )

    return features_df, per_col_df


def _assemble_dynamic_features(
    fdf: pd.DataFrame,
    cat_sets_list: list,
    cols_per_day: list,
    n_days: int,
) -> pd.DataFrame:
    """Añade G2 tendencia, G4 dinámicas lag-7, G5 esquema, G6 coherencia, G7 temporal."""
    fdf = fdf.sort_values("date").reset_index(drop=True)

    # G2: tendencia 7d
    raw_counts = fdf["vol_row_count_raw"].values
    slope_7d = np.zeros(n_days)
    for i in range(1, n_days):
        w = raw_counts[max(0, i - 6): i + 1]
        if len(w) >= 2:
            slope_7d[i] = float(np.polyfit(np.arange(len(w), dtype=float),
                                           np.log1p(w), 1)[0])
    fdf["vol_row_count_7d_slope"] = slope_7d

    # G4: features dinámicas (lag-7 seasonal, excepto row_count_delta que es lag-1)
    LAG_S = 7
    dyn_keys = [
        "dyn_row_count_delta_pct", "dyn_n_new_cat_values",
        "dyn_n_disappeared_cat_values", "dyn_pct_new_cat_values",
        "dyn_pct_disappeared_cat_values", "dyn_delta_mean_pct_null",
        "dyn_delta_max_pct_null", "dyn_delta_gini_pct_null",
        "dyn_delta_mean_entropy_cat", "dyn_delta_mean_hhi",
        "dyn_delta_mean_top1_share", "dyn_delta_mean_pct_empty",
        "dyn_delta_mean_pct_unknown", "dyn_delta_n_cats",
    ]
    dyn = {k: np.zeros(n_days) for k in dyn_keys}

    for i in range(1, n_days):
        prev1 = fdf.iloc[i - 1]
        prev7 = fdf.iloc[i - LAG_S] if i >= LAG_S else fdf.iloc[0]
        curr  = fdf.iloc[i]

        prev_rc = prev1["vol_row_count_raw"]
        curr_rc = curr["vol_row_count_raw"]
        dyn["dyn_row_count_delta_pct"][i] = (
            (curr_rc - prev_rc) / prev_rc if prev_rc > 0 else 0.0
        )

        ts = cat_sets_list[i]
        ps = cat_sets_list[i - LAG_S] if i >= LAG_S else cat_sets_list[0]
        common = set(ts.keys()) & set(ps.keys())
        new_tot = disap_tot = prev_tot = 0
        for col in common:
            new_tot   += len(ts[col] - ps[col])
            disap_tot += len(ps[col] - ts[col])
            prev_tot  += len(ps[col])
        dyn["dyn_n_new_cat_values"][i]           = float(new_tot)
        dyn["dyn_n_disappeared_cat_values"][i]   = float(disap_tot)
        dyn["dyn_pct_new_cat_values"][i]         = new_tot   / prev_tot if prev_tot else 0.0
        dyn["dyn_pct_disappeared_cat_values"][i] = disap_tot / prev_tot if prev_tot else 0.0

        dyn["dyn_delta_mean_pct_null"][i]    = curr["qual_mean_pct_null"]    - prev7["qual_mean_pct_null"]
        dyn["dyn_delta_max_pct_null"][i]     = curr["qual_max_pct_null"]     - prev7["qual_max_pct_null"]
        dyn["dyn_delta_gini_pct_null"][i]    = curr["qual_gini_pct_null"]    - prev7["qual_gini_pct_null"]
        dyn["dyn_delta_mean_entropy_cat"][i] = curr["dist_mean_entropy_cat"] - prev7["dist_mean_entropy_cat"]
        dyn["dyn_delta_mean_hhi"][i]         = curr["dist_mean_hhi"]         - prev7["dist_mean_hhi"]
        dyn["dyn_delta_mean_top1_share"][i]  = curr["dist_mean_top1_share"]  - prev7["dist_mean_top1_share"]
        dyn["dyn_delta_mean_pct_empty"][i]   = curr["qual_mean_pct_empty"]   - prev7["qual_mean_pct_empty"]
        dyn["dyn_delta_mean_pct_unknown"][i] = curr["qual_mean_pct_unknown"] - prev7["qual_mean_pct_unknown"]
        dyn["dyn_delta_n_cats"][i]           = curr["dist_mean_n_cats"]      - prev7["dist_mean_n_cats"]

    for col, arr in dyn.items():
        fdf[col] = arr

    # G5: columnas nuevas
    schema_new = np.zeros(n_days)
    for i in range(1, n_days):
        schema_new[i] = float(len(cols_per_day[i] - cols_per_day[i - 1]))
    fdf["schema_n_new_cols"] = schema_new

    # G6: coherencia
    null_delta = fdf["dyn_delta_mean_pct_null"].values
    vol_delta  = fdf["dyn_row_count_delta_pct"].values
    fdf["coh_null_x_volume"] = np.clip(null_delta * (-vol_delta), 0, None)
    fdf["coh_pct_cols_degraded"] = (
        (fdf["dyn_delta_mean_pct_null"]    > 0.01).astype(float) +
        (fdf["dyn_delta_mean_pct_empty"]   > 0.01).astype(float) +
        (fdf["dyn_delta_mean_pct_unknown"] > 0.01).astype(float)
    ) / 3.0
    fdf["coh_distribution_drift"] = (
        fdf["dyn_delta_mean_entropy_cat"].abs() +
        fdf["dyn_delta_mean_hhi"].abs() +
        fdf["dyn_delta_mean_top1_share"].abs()
    )
    log_vol = np.log1p(fdf["vol_row_count_raw"].values)
    max_log = log_vol.max() if log_vol.max() > 0 else 1.0
    fdf["coh_volume_quality_ratio"] = (
        (1.0 - fdf["coh_quality_score"]) * (log_vol / max_log)
    )

    pct_deg = fdf["coh_pct_cols_degraded"].values
    dyn_delta_deg = np.zeros(n_days)
    for i in range(n_days):
        ref = i - 7 if i >= 7 else 0
        dyn_delta_deg[i] = pct_deg[i] - pct_deg[ref]
    fdf["dyn_delta_pct_cols_degraded"] = dyn_delta_deg

    null_vals  = fdf["qual_mean_pct_null"].values
    null_trend = np.zeros(n_days)
    for i in range(1, n_days):
        w_null = null_vals[max(0, i-6): i+1]
        w_vol  = log_vol[max(0, i-6): i+1]
        if len(w_null) >= 3 and w_vol.std() > 1e-6:
            null_trend[i] = float(np.polyfit(np.arange(len(w_null)), w_null, 1)[0])
    fdf["coh_null_volume_trend"] = null_trend

    # G7: temporal
    dow = fdf["date"].dt.dayofweek
    mon = fdf["date"].dt.month
    fdf["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    fdf["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    fdf["month_sin"]       = np.sin(2 * np.pi * (mon - 1) / 12)
    fdf["month_cos"]       = np.cos(2 * np.pi * (mon - 1) / 12)

    missing = [f for f in UNIVERSAL_FEATURE_NAMES if f not in fdf.columns]
    if missing:
        log.warning(f"[FEATURES] Faltantes (→ 0): {missing}")
        for f in missing:
            fdf[f] = 0.0

    return fdf[["date"] + UNIVERSAL_FEATURE_NAMES].fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    cfg: dict = None,
    **overrides,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """
    Pipeline completo para discounts. T1 + T2 + T3.

    Returns: (raw_df, scaled_df, feature_cols, per_col_df)
    """
    c = dict(cfg or CFG_DISCOUNTS)
    c.update(overrides)

    scaled_path = c["paths"]["scaled_data"]
    percol_path = _per_column_stats_path(c)
    force = c.get("force_download", False) or c.get("force_features", False)

    if (scaled_path.exists() and percol_path and percol_path.exists()
            and not force):
        log.info("[PIPELINE] Cargando desde caché")
        scaled_df  = pd.read_parquet(scaled_path)
        per_col_df = pd.read_parquet(percol_path)
        feat_cols  = [col for col in scaled_df.columns if col != "date"]
        raw_path   = c["paths"]["raw_data"]
        raw_df     = pd.read_parquet(raw_path) if raw_path.exists() else scaled_df
        return raw_df, scaled_df, feat_cols, per_col_df

    # 1. Descarga
    raw_df = download_discounts_raw(c)

    # 2. Snapshots → features + per_col_stats (un solo pase)
    features_df, per_col_df = build_all_from_snapshots(raw_df, c)

    # 3. Guardar per_col_stats
    if percol_path:
        percol_path.parent.mkdir(parents=True, exist_ok=True)
        per_col_df.to_parquet(percol_path, index=False)
        log.info(f"[PIPELINE] per_col_stats → {percol_path}")

    # 4. Escalado
    train_split = c.get("tranad", {}).get("train_split", 0.95)
    n_days      = len(features_df)
    n_train     = int(n_days * train_split)
    train_mask  = pd.Series([True] * n_train + [False] * (n_days - n_train))

    scaled_df, feat_cols = fit_and_scale(features_df, c, train_mask, force=force)
    scaled_df.to_parquet(scaled_path, index=False)

    return raw_df, scaled_df, feat_cols, per_col_df
