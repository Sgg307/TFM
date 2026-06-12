"""statistical_detector.py — Tier 1: Atribución DOW-aware por columna (z-scores MAD)

REFACTOR (Fase 0 del DriftMonitor):
  La construcción del baseline DOW-aware se ha extraído a `build_dow_baseline`,
  función de módulo que pasa a ser el ÚNICO punto de verdad para "median+MAD
  per-(dow, col, stat)". La usan:
    - ColumnStatisticalDetector.fit()  → baseline CONGELADO de detección.
    - drift_core.standardize_residuals  → mismo baseline para medir drift.

GUARDA DE VOLUMEN (min_rows):
  Los días con n_rows < self.min_rows devuelven `low_volume=True` y no
  generan flagged columns ni atribución — con tan pocas filas los estadísticos
  de distribución son ruido (la fracción de nulos de una columna solo puede
  tomar valores {0, 1/n, 2/n…}).
"""

from __future__ import annotations
import logging, pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Set
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_STATS = ["pct_null", "pct_empty", "pct_unknown",
          "entropy_norm", "hhi", "top1_share", "n_cats"]
_QUALITY_STATS = {"pct_null", "pct_empty", "pct_unknown"}
_MAD_K = 1.4826
_Z_CAP = 15.0
_MAD_FLOOR = {"pct_null": .01, "pct_empty": .01, "pct_unknown": .01,
              "entropy_norm": .03, "hhi": .03, "top1_share": .03, "n_cats": 1.}


def build_dow_baseline(
    per_col_df: pd.DataFrame,
    dates_subset: Iterable,
    excluded_cols: Optional[Set[str]] = None,
    min_history: int = 8,
) -> Dict[int, Dict[tuple, tuple]]:
    """Construye el baseline DOW-aware: {dow: {(col, stat): (median, mad, n)}}.

    Único punto de verdad. `dates_subset` define qué días entran en el ajuste
    (training para el frozen; ventana pre-rampa para el drift monitor).
    """
    excluded = excluded_cols or set()
    dates_set = {pd.Timestamp(d) for d in dates_subset}
    df = per_col_df[~per_col_df["column"].isin(excluded)].copy()
    df["_ts"]  = pd.to_datetime(df["date"])
    df = df[df["_ts"].isin(dates_set)]
    df["_dow"] = df["_ts"].dt.dayofweek

    baselines: Dict[int, Dict[tuple, tuple]] = {}
    for dow in range(7):
        dd = df[df["_dow"] == dow]
        if dd.empty:
            continue
        bl: Dict[tuple, tuple] = {}
        for col in dd["column"].unique():
            cd = dd[dd["column"] == col]
            for stat in _STATS:
                vals = cd[stat].dropna().values
                if len(vals) >= min_history:
                    med = float(np.median(vals))
                    bl[(col, stat)] = (med, float(np.median(np.abs(vals - med))), len(vals))
        baselines[dow] = bl
    return baselines


def effective_mad(med: float, mad: float, stat: str) -> float:
    """MAD efectivo con suelos (idéntico al usado en score_day). Expuesto para
    que drift_core estandarice con EXACTAMENTE la misma definición."""
    return max(mad, _MAD_FLOOR.get(stat, .01), abs(med) * 0.05)


class ColumnStatisticalDetector:
    def __init__(self, z_threshold: float = 4.0, min_flagged_pct: float = 0.10,
                 min_history: int = 8, min_rows: int = 100):
        self.z_threshold     = z_threshold
        self.min_flagged_pct = min_flagged_pct
        self.min_history     = min_history
        self.min_rows        = min_rows
        self.baselines: Dict = {}

    def fit(self, per_col_df: pd.DataFrame, train_mask: np.ndarray,
            excluded_cols: Optional[Set[str]] = None):
        dates_sorted = sorted(per_col_df["date"].unique())
        train_dates  = {d for d, m in zip(dates_sorted, train_mask) if m}
        self.baselines = build_dow_baseline(per_col_df, train_dates,
                                            excluded_cols, self.min_history)
        log.info(f"[TIER1] {sum(len(v) for v in self.baselines.values())} baselines")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "baselines":       self.baselines,
                "z_threshold":     self.z_threshold,
                "min_flagged_pct": self.min_flagged_pct,
                "min_history":     self.min_history,
                "min_rows":        self.min_rows,
            }, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.baselines       = d["baselines"]
        self.z_threshold     = d["z_threshold"]
        self.min_flagged_pct = d["min_flagged_pct"]
        self.min_history     = d["min_history"]
        # Compat con artefactos viejos sin min_rows (default conservador).
        self.min_rows        = d.get("min_rows", 100)
        log.info(f"[TIER1] Cargado desde {path}")

    def score_day(self, per_col_df: pd.DataFrame, target_date,
                  excluded_cols: Optional[Set[str]] = None) -> Dict:
        excluded = excluded_cols or set()
        ts = pd.Timestamp(target_date)
        bl = self.baselines.get(ts.dayofweek)
        if not bl:
            return _empty(target_date)

        today = per_col_df[(per_col_df["date"] == target_date)
                           & ~per_col_df["column"].isin(excluded)]
        if today.empty:
            return _empty(target_date)

        # Guarda de volumen: con pocas filas los stats son ruido (valores
        # discretos en {0, 1/n, 2/n...}); no se puntúa.
        n_rows_today = today["n_rows"].iloc[0] if "n_rows" in today.columns else None
        if n_rows_today is not None and float(n_rows_today) < self.min_rows:
            out = _empty(target_date)
            out["low_volume"] = True
            out["n_rows"] = float(n_rows_today)
            return out

        rows = []
        for _, row in today.iterrows():
            col = row["column"]
            for stat in _STATS:
                key = (col, stat)
                if key not in bl:
                    continue
                val, (med, mad, _) = row[stat], bl[key]
                ch = "Quality" if stat in _QUALITY_STATS else "Structural"
                if pd.isna(val):
                    rows.append({"column": col, "stat": stat, "value": np.nan,
                                 "expected": med, "z_score": np.nan,
                                 "flagged": True, "channel": ch, "reason": "NaN"})
                    continue
                eff_mad = effective_mad(med, mad, stat)
                z = float(np.clip((val - med) / (_MAD_K * eff_mad), -_Z_CAP, _Z_CAP))
                rows.append({
                    "column": col, "stat": stat, "value": float(val),
                    "expected": med, "z_score": z,
                    "flagged": abs(z) > self.z_threshold, "channel": ch,
                    "reason": f"z={z:.1f}" if abs(z) > self.z_threshold else "",
                })

        if not rows:
            return _empty(target_date)
        df      = pd.DataFrame(rows)
        flagged = df[df["flagged"]]
        col_rms = df.groupby("column").agg(
            rms_z=("z_score",  lambda x: float(np.sqrt((x.dropna() ** 2).mean()))),
            n_flagged=("flagged", "sum"),
            max_z=("z_score",  lambda x: float(x.abs().max()) if x.notna().any() else 0.),
            channels=("channel", lambda x: "+".join(sorted(set(x)))),
        ).sort_values("rms_z", ascending=False).reset_index()

        return {
            "date":        target_date,
            "anomaly":     False,
            "type":        "attribution",
            "n_flagged":   len(flagged),
            "n_total":     len(df),
            "pct_flagged": len(flagged) / max(len(df), 1),
            "max_z":       float(df["z_score"].abs().max()) if df["z_score"].notna().any() else 0.,
            "low_volume":  False,
            "n_rows":      float(n_rows_today) if n_rows_today is not None else None,
            "details_df":  df,
            "top_columns": col_rms.head(10),
        }


def _empty(date):
    return {
        "date":        date,
        "anomaly":     False,
        "type":        "attribution",
        "n_flagged":   0,
        "n_total":     0,
        "pct_flagged": 0.,
        "max_z":       0.,
        "low_volume":  False,
        "n_rows":      None,
        "details_df":  pd.DataFrame(),
        "top_columns": pd.DataFrame(),
    }
