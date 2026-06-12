"""drift_validation.py — Validación cruzada del DriftMonitor con ground-truth.

TESIS: el staleness sube monótonamente al acercarse a la rampa de discounts
(~agosto 2024) y la frontera AUTOMÁTICA coincide con la diagnosticada A MANO
(t2_error 0.038→0.170). En una ventana SANA de portabilidades se queda bajo
banda. Dos métodos independientes convergiendo = validación cruzada limpia,
y la figura del capítulo de drift del TFM.

CLAVE METODOLÓGICA (señal c): el harness ajusta su PROPIO PCA pre-rampa
internamente (≤ frozen_end). NO usa el tier2_pca operativo de discounts, que
se fiteó hasta 2024-06-01 → ya contiene el arranque de la rampa (in-sample,
sucio). Así el "frozen" es honesto y todo depende de UN artefacto: per_col_df.

USO (en el notebook, donde hay BQ):

    # ── 1. Corrida DEDICADA: materializar la zona sucia de discounts ─────────
    # El per_col_df cacheado para a 2024-06-01 por el cap operativo; aquí
    # levantamos end_date para reconstruir snapshots A TRAVÉS de la rampa.
    # Las raw rows llegan a 2025 → no se re-descarga, solo se reconstruye.
    res = runner.run_multi_tier("discounts", force_refeatures=True)   # end_date alto en CFG_DISCOUNTS["bq"]
    pcd_dirty = res["per_col_df"]            # debe cubrir feb-2024 → 2025

    from drift_validation import run_experiment, plot_validation
    excl = set(get_cfg("discounts")["bq"]["exclude_cols"])
    dirty = run_experiment(pcd_dirty, frozen_end="2024-01-31", excluded=excl)

    # ── 2. Control SANO: ventana estable de portabilidades ───────────────────
    res_p = runner.run_multi_tier("portabilidades")
    pcd_p = res_p["per_col_df"]
    excl_p = set(get_cfg("portabilidades")["bq"]["exclude_cols"])
    healthy = run_experiment(pcd_p, frozen_end="2024-06-30", excluded=excl_p)

    # ── 3. Figura ────────────────────────────────────────────────────────────
    plot_validation(dirty, healthy, ramp_date="2024-08-01")
"""

from __future__ import annotations
import logging
from typing import List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.pca_detector import CrossColumnPCADetector
from core.statistical_detector import _STATS, build_dow_baseline
from core.drift_core import (standardize_residuals, recon_z_series, concept_psi,
                             assess_window, PSI_MODERATE)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def pivot_wide(per_col_df: pd.DataFrame, excluded: Optional[Set[str]] = None) -> pd.DataFrame:
    """per_col_df (long) → matriz wide [date × col__stat], ffill+0 como producción.
    Representación cross-column auto-contenida para el PCA de la señal (c)."""
    excluded = excluded or set()
    df = per_col_df[~per_col_df["column"].isin(excluded)].copy()
    df["date"] = pd.to_datetime(df["date"])
    long = df.melt(id_vars=["date", "column"], value_vars=_STATS,
                   var_name="stat", value_name="val")
    long["feature"] = long["column"] + "__" + long["stat"]
    wide = long.pivot_table(index="date", columns="feature", values="val", aggfunc="first")
    return wide.sort_index().ffill().fillna(0.0)


def fit_frozen_references(per_col_df: pd.DataFrame, frozen_end, excluded=None,
                          min_history: int = 8, explained_variance: float = 0.95,
                          ref_holdout_frac: float = 0.30) -> dict:
    """Ajusta TODAS las referencias congeladas sobre la zona pre-rampa (≤ frozen_end):
    Tier 1 baseline DOW-aware + scaler RobustScaler + PCA per-DOW.

    SESGO IN-SAMPLE (señal c): un PCA reconstruye sus días de fit mejor que días
    nuevos, así que comparar la distribución de recon-error de fit (in-sample)
    contra ventanas out-of-sample da concept-drift FALSO. Por eso reservamos una
    cola hold-out del pre-rampa (`ref_holdout_frac`) que el PCA NO ve: la
    distribución de referencia de recon-error se mide ahí → ref y ventanas son
    ambas out-of-sample (comparación justa). El baseline de Tier 1 sí usa todo el
    pre-rampa (mediana/MAD robusta no infla fuera de muestra).
    """
    excluded = excluded or set()
    frozen_end = pd.Timestamp(frozen_end)
    pre = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in per_col_df["date"].unique()
                                   if pd.Timestamp(d) <= frozen_end}))
    log.info(f"[FROZEN] referencia pre-rampa: {len(pre)} días ≤ {frozen_end.date()}")
    if len(pre) < 40:
        log.warning(f"[FROZEN] solo {len(pre)} días pre-rampa — PCA per-DOW / hold-out justos")

    # Split temporal: PCA se ajusta en la cabeza; la cola queda hold-out (out-of-sample sano)
    n_hold = max(int(len(pre) * ref_holdout_frac), 1)
    pca_fit_dates = pre[:-n_hold] if 0 < n_hold < len(pre) else pre
    ref_holdout_dates = pre[-n_hold:] if 0 < n_hold < len(pre) else pre
    log.info(f"[FROZEN] PCA fit: {len(pca_fit_dates)} días | ref hold-out: {len(ref_holdout_dates)} días")

    # Tier 1 — baseline congelado sobre TODO el pre-rampa (mismo build que producción)
    t1_baseline = build_dow_baseline(per_col_df, pre, excluded, min_history)

    # Tier 2 — scaler + PCA per-DOW congelados, ajustados SOLO en pca_fit_dates
    wide = pivot_wide(per_col_df, excluded)
    dates_wide = pd.DatetimeIndex(wide.index)
    fit_set = set(pca_fit_dates)
    mask_fit = np.array([d in fit_set for d in dates_wide])
    scaler = RobustScaler().fit(wide.values[mask_fit])
    scaled = np.clip(scaler.transform(wide.values), -5.0, 5.0).astype(np.float32)
    t2 = CrossColumnPCADetector(explained_variance=explained_variance)
    t2.fit(scaled, dates_wide, mask_fit, list(wide.columns))

    return {"t1_baseline": t1_baseline, "t2_frozen": t2, "scaler": scaler,
            "scaled_wide": scaled, "dates_wide": dates_wide,
            "pre_dates": pre, "pca_fit_dates": pca_fit_dates,
            "ref_holdout_dates": ref_holdout_dates}


def sliding_endpoints(dates, frozen_end, window_days: int = 56, step_days: int = 7) -> List[pd.Timestamp]:
    """Endpoints de ventana deslizante a partir de frozen_end (no exige que el
    endpoint sea una fecha existente; la máscara de ventana selecciona lo que caiga)."""
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates)))
    after = dates[dates > pd.Timestamp(frozen_end)]
    if len(after) == 0:
        return []
    cur, last, out = after.min(), after.max(), []
    while cur <= last:
        out.append(cur)
        cur = cur + pd.Timedelta(days=step_days)
    return out


def run_experiment(per_col_df: pd.DataFrame, frozen_end, excluded: Optional[Set[str]] = None,
                   window_days: int = 56, step_days: int = 7, band: float = PSI_MODERATE,
                   n_cols_alert: float = 0.15, n_boot: int = 150) -> pd.DataFrame:
    """Barre ventanas deslizantes a través de la zona post-frozen y devuelve el
    staleness por endpoint: [date, frac_data_drift, concept_psi_lo, verdict, ...]."""
    excluded = excluded or set()
    refs = fit_frozen_references(per_col_df, frozen_end, excluded)

    pcd = per_col_df.copy()
    pcd["date"] = pd.to_datetime(pcd["date"])

    # Distribución de referencia — se calcula UNA vez.
    #  (a): z-residuales sobre todo el pre-rampa (mediana/MAD robusta, sin sesgo OOS).
    #  (c): recon-error sobre el HOLD-OUT pre-rampa (out-of-sample → comparación justa).
    z_ref = standardize_residuals(pcd[pcd["date"].isin(set(refs["pre_dates"]))],
                                  refs["t1_baseline"], excluded)
    ref_recon_z = recon_z_series(refs["t2_frozen"], refs["scaled_wide"],
                                 refs["dates_wide"], refs["ref_holdout_dates"])

    rows = []
    for end in sliding_endpoints(pcd["date"], frozen_end, window_days, step_days):
        w0 = end - pd.Timedelta(days=window_days)
        win_df = pcd[(pcd["date"] > w0) & (pcd["date"] <= end)]
        n_days = win_df["date"].nunique()
        if n_days < window_days // 2:
            continue
        win_dates = pd.DatetimeIndex(sorted(win_df["date"].unique()))

        z_recent = standardize_residuals(win_df, refs["t1_baseline"], excluded)
        recent_recon_z = recon_z_series(refs["t2_frozen"], refs["scaled_wide"],
                                         refs["dates_wide"], win_dates)
        concept = concept_psi(ref_recon_z, recent_recon_z, n_boot=n_boot)

        a = assess_window(z_ref, z_recent, concept, band, n_cols_alert)
        rows.append({"date": end, "n_days": n_days,
                     **{k: a[k] for k in ("frac_data_drift", "n_drift", "n_colstat",
                                          "concept_psi", "concept_psi_lo",
                                          "concept_drift", "verdict")}})
        log.info(f"[EXP] {end.date()}  data={a['frac_data_drift']:.0%}  "
                 f"concept_lo={a['concept_psi_lo']:.3f}  → {a['verdict']}")
    return pd.DataFrame(rows)


def plot_validation(dirty: pd.DataFrame, healthy: Optional[pd.DataFrame] = None,
                    ramp_date: str = "2024-08-01", band_frac: float = 0.15,
                    out: str = "plots/drift_validation.png"):
    """Figura de validación cruzada: dos paneles (data-drift y concept-drift),
    discounts vs portabilidades-sano, con frontera manual y bandas de alerta."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    ramp = pd.Timestamp(ramp_date)

    def _panel(ax, col_y, band_y, band_label, ylabel, title, show_ramp_label):
        ax.plot(dirty["date"], dirty[col_y], "o-", color="#c0392b", label="discounts (rampa)")
        if healthy is not None and not healthy.empty:
            ax.plot(healthy["date"], healthy[col_y], "s-", color="#27ae60", label="portabilidades (sano)")
        ax.axhline(band_y, ls="--", color="gray", lw=1, label=band_label)
        ax.axvline(ramp, ls=":", color="black", lw=1.3,
                   label="frontera manual (~ago 2024)" if show_ramp_label else None)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlabel("endpoint de ventana")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5))
    _panel(ax[0], "frac_data_drift", band_frac, f"banda RETRAIN ({band_frac:.0%})",
           "% (col,stat) en data-drift",
           "Señal (a) — DATA drift (marginales por columna)", show_ramp_label=True)
    _panel(ax[1], "concept_psi_lo", PSI_MODERATE, f"banda PSI ({PSI_MODERATE})",
           "PSI recon-error T2 (IC inferior)",
           "Señal (c) — CONCEPT drift (estructura de correlación)", show_ramp_label=False)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log.info(f"[EXP] figura guardada → {out}")
    return fig
