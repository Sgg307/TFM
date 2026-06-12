"""row_level_drift.py — Drift a nivel de fila (extensión del DriftMonitor).

Mide si la POBLACIÓN de filas se ha alejado de la referencia, vía PSI sobre la
distribución del error de reconstrucción por fila del AE row-level congelado.
Complementa el drift de vectores diarios: capta cambios en la distribución
CONJUNTA de filas (subgrupos nuevos, combinaciones de valores) que los
estadísticos agregados por columna no ven.

Disciplina in-sample (igual que el PCA congelado del concept-drift diario):
el AE debe estar entrenado SOLO con datos pre-evento. Si no, mides la rampa
con un modelo entrenado sobre la rampa. La ventana de validación del pipeline
(OOS, sana) hace de referencia.

USO:
    # 1. AE dedicado pre-evento (override train_end en cfg)
    cfg = get_cfg("portabilidades")
    cfg["row_level"]["train_end"] = "2024-04-30"     # pre-evento
    cfg["row_level"]["val_start"] = "2024-05-01"
    cfg["row_level"]["val_end"]   = "2024-06-30"      # referencia OOS sana
    pipe = RowLevelPipeline(cfg); pipe.train()

    # 2. Drift row-level (reutiliza el scorer y las filas ya descargadas)
    from core.row_level_drift import run_row_level_drift, plot_row_level_drift
    drift = run_row_level_drift(pipe.scorer, pipe._raw_df, cfg["bq"]["date_col"],
                                ref_start="2024-05-01", ref_end="2024-06-30")
    print(drift[["date","psi_lo","median_score","verdict"]].to_string(index=False))
    plot_row_level_drift(drift, ramp_date="2024-08-01")
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

from core.drift_core import psi_ci, PSI_STABLE, PSI_MODERATE

log = logging.getLogger(__name__)


def score_by_day(scorer, raw_df: pd.DataFrame, date_col: str) -> Dict[pd.Timestamp, np.ndarray]:
    """Error de reconstrucción por fila agrupado por día — una pasada del AE por día.
    Puntuar cada fila una sola vez y luego concatenar evita re-puntuar solapamientos."""
    dates = pd.to_datetime(raw_df[date_col]).dt.normalize()
    by_day = {}
    for d in dates.unique():
        scores, _ = scorer._score_rows_raw(raw_df[dates.values == d])
        by_day[pd.Timestamp(d)] = np.asarray(scores, dtype=float)
    return by_day


def _pool(by_day: Dict[pd.Timestamp, np.ndarray], lo, hi) -> np.ndarray:
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    chunks = [v for d, v in by_day.items() if lo <= d <= hi]
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def run_row_level_drift(scorer, raw_df: pd.DataFrame, date_col: str,
                        ref_start, ref_end, eval_start=None,
                        window_days: int = 56, step_days: int = 14,
                        bins: int = 5, n_boot: int = 150,
                        band: float = PSI_MODERATE, min_rows: int = 200) -> pd.DataFrame:
    """PSI de la distribución de error por fila: ventana deslizante vs referencia
    OOS sana fija. Reutiliza psi_ci del DriftMonitor diario (verdict sobre IC inferior)."""
    by_day = score_by_day(scorer, raw_df, date_col)
    ref = _pool(by_day, ref_start, ref_end)
    log.info(f"[RL-DRIFT] referencia OOS: {len(ref):,} filas ({ref_start} → {ref_end})")

    days = pd.DatetimeIndex(sorted(by_day))
    start = pd.Timestamp(eval_start or ref_end)
    ends = days[days > start]
    if len(ends) == 0:
        return pd.DataFrame()

    rows, cur, last = [], ends.min(), ends.max()
    while cur <= last:
        win = _pool(by_day, cur - pd.Timedelta(days=window_days), cur)
        if len(win) >= min_rows:
            p, lo, hi = psi_ci(ref, win, bins, n_boot)
            drift = bool(np.isfinite(lo) and lo > band)
            verdict = ("SUGIERE_RETRAIN" if drift
                       else "MONITOR" if np.isfinite(lo) and lo > PSI_STABLE
                       else "OK")
            rows.append({"date": cur, "n_rows": len(win), "psi": p,
                         "psi_lo": lo, "psi_hi": hi,
                         "median_score": float(np.median(win)),
                         "drift": drift, "verdict": verdict})
            log.info(f"[RL-DRIFT] {cur.date()}  psi_lo={lo:.3f}  n={len(win):,}  → {verdict}")
        cur += pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)


def plot_row_level_drift(drift: pd.DataFrame, ramp_date: Optional[str] = None,
                         band: float = PSI_MODERATE, out: str = "plots/row_level_drift.png"):
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(drift["date"], drift["psi_lo"], "o-", color="#8e44ad",
            label="PSI error/fila (IC inferior)")
    ax.axhline(band, ls="--", color="gray", lw=1, label=f"banda ({band})")
    if ramp_date:
        ax.axvline(pd.Timestamp(ramp_date), ls=":", color="red", lw=1.5, label="evento")
    ax.set_title("Row-level drift — distribución de error de reconstrucción por fila")
    ax.set_ylabel("PSI (IC inferior)"); ax.set_xlabel("endpoint de ventana")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig
