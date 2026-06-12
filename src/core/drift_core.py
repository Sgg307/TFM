"""drift_core.py — Núcleo de cómputo de drift (Fase 1).

Mide cuánto se ha alejado una referencia CONGELADA (pre-deriva) de la realidad
reciente, SIN tocar la detección de anomalías. Tres señales, UN mecanismo
(PSI sobre z-residuales DOW-aware):

  (a) DATA drift    — PSI de las marginales por (columna, stat).
  (b) corroboración — desviación de la mediana de los z-residuales recientes.
  (c) CONCEPT drift — PSI del error de reconstrucción de un PCA CONGELADO
                      pre-deriva (deriva de la estructura de correlación).

Decisiones de diseño (de la discusión previa):
  · bins=5 + IC bootstrap → el verdict se lee sobre el borde INFERIOR del IC,
    nunca sobre el PSI puntual (con ventanas cortas el puntual parpadea).
  · z-residual DOW-aware idéntico al de Tier 1 (effective_mad compartido)
    → elimina el confound de DOW y unifica (a) y (b) en un mismo flujo.
  · métrica principal = %(col,stat) con PSI_lo > banda. Interpretable, sin
    pesos mágicos. (b) y (c) corroboran/explican; NO se fusionan en un número.
  · verdict es ADVISORY: PSI alto sostenido no distingue "el mundo cambió"
    (retrain) de "el pipeline lleva roto N días" (arreglar pipeline). Decide
    un humano; la Fase 2 (persistencia) lo afina parcialmente.
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Set
import numpy as np
import pandas as pd

from core.statistical_detector import _STATS, _MAD_K, _Z_CAP, effective_mad

# Bandas PSI estándar de model-monitoring
PSI_STABLE = 0.10
PSI_MODERATE = 0.25  # > banda  →  drift significativo


# ─────────────────────────────────────────────────────────────────────────────
# z-residual DOW-aware  (definición IDÉNTICA a Tier1.score_day)
# ─────────────────────────────────────────────────────────────────────────────
def standardize_residuals(
    per_col_df: pd.DataFrame,
    frozen_baseline: Dict[int, Dict[tuple, tuple]],
    excluded: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Convierte cada (día, col, stat) en su z-residual contra la mediana DOW
    CONGELADA. Mismo cálculo que Tier 1 → su divergencia ES la métrica de drift.

    Devuelve long: [date, dow, column, stat, z].
    """
    excluded = excluded or set()
    df = per_col_df[~per_col_df["column"].isin(excluded)].copy()
    df["_ts"] = pd.to_datetime(df["date"])
    df["_dow"] = df["_ts"].dt.dayofweek

    out = []
    for dow, grp in df.groupby("_dow"):
        bl = frozen_baseline.get(int(dow))
        if not bl:
            continue
        for _, row in grp.iterrows():
            col = row["column"]
            for stat in _STATS:
                key = (col, stat)
                if key not in bl:
                    continue
                val = row[stat]
                if pd.isna(val):
                    continue
                med, mad, _ = bl[key]
                z = (val - med) / (_MAD_K * effective_mad(med, mad, stat))
                out.append({"date": row["_ts"], "dow": int(dow), "column": col,
                            "stat": stat, "z": float(np.clip(z, -_Z_CAP, _Z_CAP))})
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# PSI  (+ IC bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 5) -> float:
    """Population Stability Index. Bins por cuantiles de `expected` (training)."""
    expected = np.asarray(expected, float); expected = expected[np.isfinite(expected)]
    actual = np.asarray(actual, float); actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) == 0:
        return np.nan
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)              # stats casi constantes → bordes colapsan
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None); a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_ci(expected, actual, bins: int = 5, n_boot: int = 200,
           alpha: float = 0.10, seed: int = 0):
    """PSI puntual + IC bootstrap (resampleo de `actual`). El verdict se lee
    sobre el borde inferior `lo` → no dispara por ruido de ventana corta."""
    point = psi(expected, actual, bins)
    actual = np.asarray(actual, float); actual = actual[np.isfinite(actual)]
    if not np.isfinite(point) or len(actual) == 0:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.array([psi(expected, rng.choice(actual, len(actual), replace=True), bins)
                      for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return point, np.nan, np.nan
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# ─────────────────────────────────────────────────────────────────────────────
# (a) + (b)  DATA drift por (columna, stat)
# ─────────────────────────────────────────────────────────────────────────────
def drift_by_colstat(z_ref: pd.DataFrame, z_recent: pd.DataFrame,
                     bins: int = 5, n_boot: int = 200,
                     band: float = PSI_MODERATE) -> pd.DataFrame:
    """Para cada (col, stat): PSI(ref vs recent) + IC  [señal (a)]  y
    median(z_recent)  [señal (b), desviación de 0]. Ordenado por PSI_lo."""
    if z_ref.empty or z_recent.empty:
        return pd.DataFrame()
    recent_g = {k: v["z"].values for k, v in z_recent.groupby(["column", "stat"])}
    rows = []
    for (col, stat), g in z_ref.groupby(["column", "stat"]):
        act = recent_g.get((col, stat))
        if act is None or len(act) == 0:
            continue
        p, lo, hi = psi_ci(g["z"].values, act, bins, n_boot)
        rows.append({"column": col, "stat": stat, "psi": p, "psi_lo": lo, "psi_hi": hi,
                     "median_shift": float(np.median(act)),
                     "drift": bool(np.isfinite(lo) and lo > band)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("psi_lo", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# (c)  CONCEPT drift  vía recon-error de un PCA CONGELADO
# ─────────────────────────────────────────────────────────────────────────────
def recon_z_series(tier2_frozen, scaled_wide: np.ndarray, dates_wide,
                   date_subset: Iterable) -> np.ndarray:
    """Error de reconstrucción del PCA congelado por día, estandarizado por DOW
    con las error_stats del propio PCA (z ~ N(0,1) si no hay concept drift)."""
    dates_wide = pd.DatetimeIndex(dates_wide)
    want = {pd.Timestamp(d) for d in date_subset}
    zs = []
    for i, d in enumerate(dates_wide):
        if pd.Timestamp(d) not in want:
            continue
        st = tier2_frozen.error_stats.get(pd.Timestamp(d).dayofweek)
        if st is None:
            continue
        te = tier2_frozen.score_day(scaled_wide[i], d)["total_error"]
        zs.append((te - st["mu"]) / st["sigma"])
    return np.asarray(zs, dtype=float)


def concept_psi(ref_z: np.ndarray, recent_z: np.ndarray,
                bins: int = 5, n_boot: int = 200) -> Dict:
    """PSI del recon-error estandarizado: ref (pre-deriva) vs reciente."""
    p, lo, hi = psi_ci(ref_z, recent_z, bins, n_boot)
    return {"psi": p, "psi_lo": lo, "psi_hi": hi,
            "median_z_recon": float(np.median(recent_z)) if len(recent_z) else np.nan,
            "n_ref": int(len(ref_z)), "n_recent": int(len(recent_z))}


# ─────────────────────────────────────────────────────────────────────────────
# Veredicto de una ventana  (ADVISORY)
# ─────────────────────────────────────────────────────────────────────────────
def assess_window(z_ref: pd.DataFrame, z_recent: pd.DataFrame, concept: Dict,
                  band: float = PSI_MODERATE, n_cols_alert: float = 0.15) -> Dict:
    """Combina las tres señales SIN fusionarlas en un número mágico.

    Métrica principal:   frac_data_drift = %(col,stat) con PSI_lo > banda.
    Corroboración:       concept_drift (PSI_lo del recon-error > banda).
    Verdict (advisory):  OK | MONITOR | SUGIERE_RETRAIN.
    """
    data_df = drift_by_colstat(z_ref, z_recent, band=band)
    n_total = len(data_df)
    n_drift = int(data_df["drift"].sum()) if n_total else 0
    frac = n_drift / n_total if n_total else 0.0

    c_lo = concept.get("psi_lo", np.nan)
    concept_drift = bool(np.isfinite(c_lo) and c_lo > band)

    if frac >= n_cols_alert or concept_drift:
        verdict = "SUGIERE_RETRAIN"
    elif frac > 0 or (np.isfinite(c_lo) and c_lo > PSI_STABLE):
        verdict = "MONITOR"
    else:
        verdict = "OK"

    return {"frac_data_drift": frac, "n_drift": n_drift, "n_colstat": n_total,
            "concept_psi": concept.get("psi"), "concept_psi_lo": c_lo,
            "concept_drift": concept_drift, "verdict": verdict,
            "top_data": data_df.head(10) if n_total else pd.DataFrame()}
