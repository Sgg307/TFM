"""
row_level_scoring.py — Scoring y detección de anomalías row-level.

  1. Score por fila:  error de reconstrucción total → ¿fila anómala?
  2. Métricas por día: agregación de scores → ¿día anómalo?
  3. Explicabilidad:  error por columna → ¿qué columnas lo causan?

CALIBRACIÓN — dinámica (análoga a Tier 3):
  El umbral por fila se calcula como mean + K×MAD sobre los scores de los
  días de CONTEXTO (anteriores a eval_days). Esto evita el problema de que
  el percentil sobre validación se infle/desinfle según incidentes presentes
  y hace que el threshold sea independiente del nº de días evaluados.

  La calibración "estática" (percentil sobre validación) se conserva como
  fallback para bootstrap y se guarda en `row_level_thresholds.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from core.schema_encoder import SchemaEncoder
from core.row_level_model import TabularAE

log = logging.getLogger(__name__)


@dataclass
class DayReport:
    """Resultado del scoring de un día."""
    date: pd.Timestamp
    dow: int
    n_total: int
    n_anomalous: int
    pct_anomalous: float
    mean_score: float
    p95_score: float
    max_score: float
    score_std: float
    threshold_used: float
    top_columns: List[Dict]
    example_rows: Optional[pd.DataFrame] = None

    @property
    def summary(self) -> str:
        lines = [
            f"📅 {self.date.strftime('%Y-%m-%d')} (DOW={self.dow})",
            f"   Filas: {self.n_total:,} total, {self.n_anomalous:,} anómalas "
            f"({self.pct_anomalous:.2%})  (umbral={self.threshold_used:.4f})",
            f"   Score: mean={self.mean_score:.4f}, p95={self.p95_score:.4f}, max={self.max_score:.4f}",
        ]
        if self.top_columns:
            lines.append("   Top columnas anómalas:")
            for tc in self.top_columns[:5]:
                lines.append(f"     • {tc['col']:<30} error={tc['mean_error']:.3f}  "
                             f"({tc['pct_rows_affected']:.1%} de filas anómalas)")
        return "\n".join(lines)


class RowLevelScorer:
    """Scoring row-level + calibración dinámica sobre ventana de inferencia."""

    def __init__(self, model: TabularAE, encoder: SchemaEncoder, cfg: dict,
                 device: str = "auto"):
        self.model = model
        self.encoder = encoder
        self.cfg = cfg
        self.rl  = cfg.get("row_level", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        # Umbral estático (percentil sobre val) — fallback para bootstrap.
        self.row_threshold: Optional[float] = None
        # Baselines por columna (error medio/desv en val) — para `top_columns`.
        self.col_baselines: Optional[Dict[str, Dict[str, float]]] = None

    # ── Calibración estática (entrenamiento) ───────────────────────────────────

    def calibrate(self, val_df: pd.DataFrame, percentile: float = None) -> Dict:
        """Calibra umbral fallback (percentil) y baselines por columna sobre val.

        En inferencia se prefiere el umbral DINÁMICO (median + K×MAD sobre
        ventana de inferencia). Este percentil solo se usa si no se dispone
        de contexto suficiente.
        """
        pctl = percentile or self.rl.get("score_percentile", 99.0)
        log.info(f"[SCORER] Calibrando sobre {len(val_df):,} filas (percentil={pctl})")
        scores, col_errors = self._score_rows_raw(val_df)

        self.row_threshold = float(np.percentile(scores, pctl))
        self.col_baselines = {
            col: {"mean": float(np.mean(errs)),
                  "std":  float(np.std(errs)),
                  "p95":  float(np.percentile(errs, 95))}
            for col, errs in col_errors.items()
        }
        log.info(f"[SCORER] Umbral fallback: {self.row_threshold:.6f} "
                 f"(p{pctl} sobre {len(scores):,} filas)")
        return {"row_threshold": self.row_threshold, "col_baselines": self.col_baselines}

    # ── Score de un día ─────────────────────────────────────────────────────────

    def score_day(self, day_df: pd.DataFrame, date: pd.Timestamp,
                  top_k_cols: int = 10, top_k_rows: int = 5,
                  threshold: Optional[float] = None) -> DayReport:
        """Evalúa un día. Si `threshold` se pasa, se usa; si no, fallback estático."""
        thr = threshold if threshold is not None else self.row_threshold
        if thr is None:
            raise RuntimeError("Sin umbral: llamar calibrate() o pasar threshold=…")

        scores, col_errors = self._score_rows_raw(day_df)
        is_anomalous = scores > thr
        n_total      = len(scores)
        n_anomalous  = int(is_anomalous.sum())

        top_columns = []
        if n_anomalous > 0 and self.col_baselines is not None:
            for col, errs in col_errors.items():
                anomalous_errs = errs[is_anomalous]
                bl = self.col_baselines.get(col, {})
                baseline     = bl.get("mean", 1e-6)
                baseline_p95 = bl.get("p95",  baseline)
                mean_err     = float(np.mean(anomalous_errs)) if len(anomalous_errs) else 0.0
                pct_affected = (float(np.mean(anomalous_errs > baseline_p95))
                                if len(anomalous_errs) else 0.0)
                top_columns.append({
                    "col": col,
                    "mean_error":       mean_err,
                    "ratio_vs_normal":  mean_err / max(baseline, 1e-6),
                    "pct_rows_affected": pct_affected,
                    "col_type": "categorical" if col in self.encoder.cat_cols else "numeric",
                })
            top_columns.sort(key=lambda x: x["ratio_vs_normal"], reverse=True)
            top_columns = top_columns[:top_k_cols]

        example_rows = None
        if top_k_rows > 0 and n_anomalous > 0:
            top_indices = np.argsort(scores)[-top_k_rows:][::-1]
            example_rows = day_df.iloc[top_indices].copy()
            example_rows["_anomaly_score"] = scores[top_indices]

        return DayReport(
            date=pd.Timestamp(date),
            dow=pd.Timestamp(date).dayofweek,
            n_total=n_total,
            n_anomalous=n_anomalous,
            pct_anomalous=n_anomalous / max(n_total, 1),
            mean_score=float(np.mean(scores)),
            p95_score=float(np.percentile(scores, 95)),
            max_score=float(np.max(scores)),
            score_std=float(np.std(scores)),
            threshold_used=float(thr),
            top_columns=top_columns,
            example_rows=example_rows,
        )

    # ── Calibración DINÁMICA sobre ventana de inferencia ────────────────────────
    #
    # Análogo a compute_dynamic_thresholds() de Tier 3: el umbral por fila se
    # calcula sobre los días de CONTEXTO (no los de evaluación) para que el
    # criterio no dependa de cuántos días estés mirando.

    def compute_dynamic_threshold(self, scores_by_day: Dict[pd.Timestamp, np.ndarray],
                                  n_eval: int, k_sigma: Optional[float] = None
                                  ) -> float:
        """Calcula el umbral por fila como median + K×MAD sobre TODAS las filas
        de los días de CONTEXTO (todos los días excepto los últimos n_eval).

        Si hay menos de `min_context_days` días de contexto, usa la ventana
        completa (toda la entrada).
        """
        k = k_sigma if k_sigma is not None else self.rl.get("k_sigma", 5.0)
        days_sorted = sorted(scores_by_day.keys())
        if n_eval is not None and 0 < n_eval < len(days_sorted):
            ctx_days = days_sorted[:-n_eval]
        else:
            ctx_days = days_sorted

        if len(ctx_days) < 3:
            ctx_days = days_sorted
            log.warning(f"[SCORER-DYN] solo {len(days_sorted)} días disponibles; "
                        f"usando ventana completa como contexto")

        ctx_scores = np.concatenate([scores_by_day[d] for d in ctx_days])
        if len(ctx_scores) == 0:
            # Sin contexto utilizable, recurrir al umbral estático.
            return self.row_threshold if self.row_threshold is not None else 0.0

        med = float(np.median(ctx_scores))
        mad = float(np.median(np.abs(ctx_scores - med)))
        sigma_est = mad * 1.4826
        if sigma_est < 1e-9:
            sigma_est = float(np.percentile(ctx_scores, 75)
                              - np.percentile(ctx_scores, 25)) / 1.35
        if sigma_est < 1e-9:
            sigma_est = float(np.std(ctx_scores))

        thr = med + k * sigma_est
        log.info(f"[SCORER-DYN] umbral fila dinámico={thr:.6f} "
                 f"(median={med:.6f}, σ={sigma_est:.6f}, k={k}, "
                 f"context={len(ctx_days)}d/{len(ctx_scores):,} filas)")
        return thr

    def score_rows_by_day(self, df: pd.DataFrame, date_col: str
                          ) -> Dict[pd.Timestamp, np.ndarray]:
        """Puntúa cada fila y agrupa los scores por día. Una pasada del AE por día."""
        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col]).dt.normalize()
        scores_by_day: Dict[pd.Timestamp, np.ndarray] = {}
        for d in sorted(df["_date"].unique()):
            day_mask = df["_date"] == d
            scores, _ = self._score_rows_raw(df[day_mask].drop(columns=["_date"]))
            scores_by_day[pd.Timestamp(d)] = scores
        return scores_by_day

    # ── Persistencia de thresholds ─────────────────────────────────────────────

    def save_thresholds(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"row_threshold": self.row_threshold,
                       "col_baselines": self.col_baselines}, f, indent=2)
        log.info(f"[SCORER] Thresholds → {path}")

    def load_thresholds(self, path: Path):
        with open(path) as f:
            state = json.load(f)
        self.row_threshold = state["row_threshold"]
        self.col_baselines = state["col_baselines"]
        log.info(f"[SCORER] Thresholds ← {path}")

    # ── Internos ───────────────────────────────────────────────────────────────

    def _score_rows_raw(self, df: pd.DataFrame, batch_size: int = None
                        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Score total + error por columna por fila. → (scores[n], col_errors{col→[n]})."""
        bs = batch_size or self.rl.get("batch_size", 4096)
        encoded = self.encoder.transform(df)

        all_total = []
        all_col_errors = {col: [] for col in self.encoder.cat_cols + self.encoder.num_cols}

        n = encoded.n_rows
        for start in range(0, n, bs):
            end = min(start + bs, n)
            cat_batch = {col: t[start:end].to(self.device)
                         for col, t in encoded.cat_tensors.items()}
            num_batch = encoded.num_tensor[start:end].to(self.device)

            with torch.no_grad():
                errors = self.model.reconstruction_error(cat_batch, num_batch)

            all_total.append(errors["total"].cpu().numpy())
            for col, ce in errors["per_cat_col"].items():
                all_col_errors[col].append(ce.cpu().numpy())
            per_num = errors["per_num_col"].cpu().numpy()
            for i, col in enumerate(self.encoder.num_cols):
                if per_num.shape[1] > i:
                    all_col_errors[col].append(per_num[:, i])

        scores = np.concatenate(all_total)
        col_errors = {col: np.concatenate(errs) for col, errs in all_col_errors.items()
                      if len(errs) > 0}
        return scores, col_errors
