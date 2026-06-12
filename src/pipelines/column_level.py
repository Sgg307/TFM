"""
column_level.py — Pipeline column-level para Tier 1 y Tier 2
=============================================================
Construye el tensor [n_dias × (n_cols × n_stats)] que PCA (Tier 2) consume.

A diferencia de base.py (62 features universales agnósticas al esquema),
este pipeline produce features anchored-to-schema: cada feature es una
pareja (columna_raw, stat).

Flujo (FIT, primera vez o force=True):
  1. Stats por columna (cached desde column_attribution)
  2. Filtrar columnas: IDs únicos, fechas, exclude_cols
  3. Pivot long → wide: [n_dias × (col__stat)]
  4. Drop con >max_nan_fraction NaN sobre train
  5. Forward-fill + median fallback
  6. RobustScaler con IQR floor
  7. Persist: scaled parquet + scaler + frozen feature list

Flujo (INFERENCIA, scaler + contract ya existen):
  1. Stats por columna
  2. Filtrar columnas
  3. Pivot long → wide
  4. REINDEXAR al contract frozen (no refiltrar por NaN)
  5. Imputar NaN con ffill + mediana del propio frame + 0
  6. Aplicar scaler frozen
  7. NO persistir (no contaminar el contract)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.column_attribution import build_per_column_stats_history

log = logging.getLogger(__name__)

COLUMN_LEVEL_STATS: List[str] = [
    "pct_null", "pct_empty", "pct_unknown",
    "entropy_norm", "hhi", "top1_share", "n_cats",
]

QUALITY_STATS    = {"pct_null", "pct_empty", "pct_unknown"}
STRUCTURAL_STATS = {"entropy_norm", "hhi", "top1_share", "n_cats"}

_DEFAULT_MAX_NAN_FRACTION = 0.30
_MIN_IQR = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILTRADO DE COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────

def _excluded_columns(cfg: dict) -> set:
    bq = cfg.get("bq", {})
    excluded = set()
    for k in ("date_col", "init_date_col", "end_date_col", "snapshot_col"):
        v = bq.get(k)
        if v:
            excluded.add(v)
    for k in ("sample_hash_col", "phone_col", "pk_col"):
        v = bq.get(k)
        if v:
            excluded.add(v)
    excluded.update(bq.get("exclude_cols", []) or [])
    return excluded


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIVOTADO LONG → WIDE
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_long_to_wide(per_col_df: pd.DataFrame, excluded: set) -> pd.DataFrame:
    df = per_col_df[~per_col_df["column"].isin(excluded)].copy()
    if df.empty:
        raise ValueError(f"[COL_LEVEL] per_col_df vacío tras exclusiones: {sorted(excluded)}")

    melted = df.melt(
        id_vars=["date", "column"], value_vars=COLUMN_LEVEL_STATS,
        var_name="stat", value_name="value",
    )
    melted["feature"] = melted["column"].astype(str) + "__" + melted["stat"]
    wide = melted.pivot(index="date", columns="feature", values="value")
    return wide.sort_index().reindex(sorted(wide.columns), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FILTRADO Y RELLENO DE NaNs (modo FIT)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_and_impute(
    wide_df:      pd.DataFrame,
    train_mask:   np.ndarray,
    max_nan_frac: float = _DEFAULT_MAX_NAN_FRACTION,
) -> Tuple[pd.DataFrame, List[str]]:
    n_train = int(train_mask.sum())
    if n_train == 0:
        raise ValueError("[COL_LEVEL] train_mask vacío.")

    train_df = wide_df.iloc[:n_train]
    nan_frac = train_df.isna().mean(axis=0)

    kept_features = nan_frac[nan_frac <= max_nan_frac].index.tolist()
    dropped = nan_frac[nan_frac > max_nan_frac]
    if len(dropped):
        log.info(f"[COL_LEVEL] Drop {len(dropped)} pares con NaN>{max_nan_frac:.0%}")

    if not kept_features:
        raise ValueError("[COL_LEVEL] Ninguna feature sobrevivió al filtro de NaN.")

    out = wide_df[kept_features].copy()
    out = out.ffill()
    train_medians = out.iloc[:n_train].median(numeric_only=True)
    out = out.fillna(train_medians)

    if out.isna().any().any():
        residual = out.isna().sum()
        raise RuntimeError(f"[COL_LEVEL] NaN residual: {residual[residual > 0].to_dict()}")

    return out, kept_features


# ─────────────────────────────────────────────────────────────────────────────
# 3-bis. REINDEXADO Y RELLENO DE NaNs (modo INFERENCIA)
# ─────────────────────────────────────────────────────────────────────────────

def _reindex_and_impute_inference(
    wide_df:  pd.DataFrame,
    contract: List[str],
) -> pd.DataFrame:
    """Modo inferencia: reindexa el wide al contract frozen (sin refiltrar) y
    rellena NaN con ffill + mediana propia + 0.

    No usa train_mask: el escalado posterior usa el scaler frozen, que ya
    tiene las constantes (mediana, IQR) del fit original. Solo necesitamos
    valores numéricos no-NaN en las features del contract, en ese orden."""
    actual   = set(wide_df.columns)
    expected = set(contract)
    missing  = expected - actual
    extra    = actual - expected

    if missing:
        log.warning(
            f"[COL_LEVEL] Inferencia: {len(missing)} features del contract no están "
            f"en el pivot (ejemplos: {sorted(missing)[:3]}). Se rellenarán con NaN→mediana/0."
        )
        for c in missing:
            wide_df[c] = np.nan
    if extra:
        log.info(f"[COL_LEVEL] Inferencia: {len(extra)} features extras del pivot ignoradas")

    out = wide_df[contract].copy()                 # reindex al contract en orden
    out = out.ffill().bfill()                       # rellenar huecos contiguos
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)
    out = out.fillna(0.0)                           # ultima salvaguarda

    if out.isna().any().any():
        residual = out.isna().sum()
        raise RuntimeError(f"[COL_LEVEL] NaN residual en inferencia: "
                           f"{residual[residual > 0].to_dict()}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. ESCALADO
# ─────────────────────────────────────────────────────────────────────────────

def _fit_or_load_scaler(
    df: pd.DataFrame, train_mask: np.ndarray, scaler_path: Path, force: bool,
) -> RobustScaler:
    if scaler_path.exists() and not force:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)

    n_train = int(train_mask.sum())
    scaler = RobustScaler()
    scaler.fit(df.iloc[:n_train])
    scaler.scale_ = np.maximum(scaler.scale_, _MIN_IQR)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"[COL_LEVEL] Scaler guardado en {scaler_path}")
    return scaler


def _load_scaler(scaler_path: Path) -> RobustScaler:
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTRATO DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _save_feature_contract(features: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"features": list(features), "version": 1}, f, indent=2)
    log.info(f"[COL_LEVEL] Contrato: {len(features)} features → {path}")


def _load_feature_contract(path: Path) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)["features"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def run_column_level_pipeline(
    raw_df: pd.DataFrame,
    cfg: dict,
    per_col_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Returns: (scaled_df, scaled_data, feature_cols)

    Tres modos, mutuamente excluyentes:
      A) CACHE HIT  — scaled_path y contract_path existen y not force.
                       Devuelve el parquet pre-escalado tal cual.
      B) INFERENCIA — scaler_path y contract_path existen y not force,
                       pero scaled_path no existe (o no aplica para esta
                       ventana). Usa scaler + contract frozen, NO refittea.
      C) FIT        — force=True o scaler_path no existe. Recalcula filtrado
                       de NaN, ajusta el scaler, persiste todos los artefactos.
    """
    paths         = cfg["paths"]
    scaled_path   = paths["column_level_data"]
    scaler_path   = paths["column_level_scaler"]
    contract_path = paths["column_level_features"]

    force = (
        cfg.get("force_features", False)
        or cfg.get("force_download", False)
        or cfg.get("force_retrain", False)
    )

    # ── MODO A: cache hit ────────────────────────────────────────────────────
    if scaled_path.exists() and contract_path.exists() and not force:
        log.info(f"[COL_LEVEL] Cargando desde caché {scaled_path}")
        scaled_df    = pd.read_parquet(scaled_path)
        feature_cols = _load_feature_contract(contract_path)
        return scaled_df, scaled_df[feature_cols].values, feature_cols

    # ── Stats por columna ────────────────────────────────────────────────────
    if per_col_df is None:
        per_col_df = build_per_column_stats_history(raw_df, cfg, force=force)

    excluded = _excluded_columns(cfg)
    log.info(f"[COL_LEVEL] Excluyendo {len(excluded)} columnas")

    wide = _pivot_long_to_wide(per_col_df, excluded)
    log.info(f"[COL_LEVEL] Wide: {wide.shape[0]} días × {wide.shape[1]} pares")

    # ── MODO B: inferencia (scaler + contract frozen) ────────────────────────
    if scaler_path.exists() and contract_path.exists() and not force:
        log.info(f"[COL_LEVEL] Modo INFERENCIA: usando scaler + contract frozen "
                 f"(no se refitea, no se persiste)")
        contract = _load_feature_contract(contract_path)
        log.info(f"[COL_LEVEL] Contract frozen: {len(contract)} features")

        imputed    = _reindex_and_impute_inference(wide, contract)
        scaler     = _load_scaler(scaler_path)
        scaled_arr = np.clip(scaler.transform(imputed), -5.0, 5.0)

        scaled_df = pd.DataFrame(scaled_arr, columns=contract, index=imputed.index)
        scaled_df = scaled_df.reset_index()
        scaled_df["date"] = pd.to_datetime(scaled_df["date"])
        return scaled_df, scaled_arr, contract

    # ── MODO C: fit completo ─────────────────────────────────────────────────
    log.info(f"[COL_LEVEL] Modo FIT: recalculando filtro NaN + scaler + contract")
    train_split = cfg.get("tranad", {}).get("train_split", 0.95)
    n_days  = len(wide)
    n_train = int(n_days * train_split)
    train_mask = np.array([True] * n_train + [False] * (n_days - n_train))

    max_nan = cfg.get("column_level", {}).get("max_nan_fraction", _DEFAULT_MAX_NAN_FRACTION)
    imputed, kept = _filter_and_impute(wide, train_mask, max_nan_frac=max_nan)

    scaler     = _fit_or_load_scaler(imputed, train_mask, scaler_path, force=force)
    scaled_arr = np.clip(scaler.transform(imputed), -5.0, 5.0)

    scaled_df = pd.DataFrame(scaled_arr, columns=kept, index=imputed.index)
    scaled_df = scaled_df.reset_index().rename(columns={"date": "date"})
    scaled_df["date"] = pd.to_datetime(scaled_df["date"])

    _save_feature_contract(kept, contract_path)
    scaled_df.to_parquet(scaled_path, index=False)

    return scaled_df, scaled_arr, kept