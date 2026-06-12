"""scoring.py — 3-channel scoring: Quality / Volume / Structural

Diseño:
  - _channel_max: top-K mean sobre scores CRUDOS, sin z-normalización por
    ventana → determinista por día.
  - compute_dynamic_thresholds: median + K×MAD sobre los días de CONTEXTO
    (excluyendo los de evaluación). Cambiar eval_days no cambia los umbrales.
  - generate_alerts: aplica guarda de fiabilidad por volumen (Tier 3) usando
    cfg["scoring"]["min_reliable_rows"] y "volume_collapse_k".
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
_CYCLIC = {"day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"}
_EXCLUDE_FROM_SCORING = {
    "text_mean_strlen", "schema_n_cat_cols", "text_std_strlen",
    "dyn_pct_disappeared_cat_values", "dyn_pct_new_cat_values",
}
_SKIP = _CYCLIC | _EXCLUDE_FROM_SCORING


def split_features_by_channel(feature_cols: List[str], cfg: dict
                              ) -> Tuple[List[int], List[int], List[int]]:
    s = cfg.get("scoring", {})
    q_pfx = s.get("quality_prefixes", ["qual_"])
    v_pfx = s.get("volume_prefixes",  ["vol_"])
    idx_q, idx_v, idx_s = [], [], []
    for i, col in enumerate(feature_cols):
        if col in _SKIP:
            continue
        elif any(col.startswith(p) for p in q_pfx):
            idx_q.append(i)
        elif any(col.startswith(p) for p in v_pfx):
            idx_v.append(i)
        else:
            idx_s.append(i)
    return idx_q, idx_v, idx_s


def _channel_max(scores: np.ndarray, indices: List[int]) -> np.ndarray:
    """Top-K mean sobre scores crudos por día. Determinista (no depende de la ventana).

    Canales pequeños (≤3 features, ej Volume): max directo.
    Canales grandes (Quality, Structural): top-3 mean.
    """
    if not indices:
        return np.zeros(len(scores))
    ch = scores[:, indices]
    n_feat = ch.shape[1]
    if n_feat <= 3:
        return ch.max(axis=1)
    k = min(3, n_feat)
    top_k = np.sort(ch, axis=1)[:, -k:]
    return top_k.mean(axis=1)


def compute_thresholds(errors_w1, errors_w2, feature_cols, train_mask, seq_len, cfg):
    """Umbrales absolutos: percentil sobre train mask. Usado solo en entrenamiento."""
    s = cfg.get("scoring", {})
    pct = s.get("threshold_percentile", 99.5)
    feat_scores = errors_w1 + errors_w2
    train_scores = feat_scores[train_mask[seq_len:][:len(feat_scores)]]
    idx_q, idx_v, idx_s = split_features_by_channel(feature_cols, cfg)
    factors = {"quality":    s.get("quality_threshold_factor",    1.0),
               "volume":     s.get("volume_threshold_factor",     1.0),
               "structural": s.get("structural_threshold_factor", 1.2)}
    thr = {ch: float(np.percentile(_channel_max(train_scores, idx), pct)) * f
           for ch, idx, f in [("quality",    idx_q, factors["quality"]),
                              ("volume",     idx_v, factors["volume"]),
                              ("structural", idx_s, factors["structural"])]}
    log.info(f"[SCORING] Umbrales — Q={thr['quality']:.4f} | "
             f"V={thr['volume']:.4f} | S={thr['structural']:.4f}")
    return thr


def compute_dynamic_thresholds(errors_w1, errors_w2, feature_cols, cfg,
                               k_sigma: float = 5.0, n_eval=None):
    """Umbrales dinámicos basados en días de CONTEXTO (median + K×MAD).

    Si n_eval se pasa, los umbrales se calculan SOLO sobre los días anteriores
    a la ventana de evaluación. Cambiar eval_days no cambia los umbrales
    porque el contexto (seq_len + 10) es el mismo.
    """
    feat_scores = errors_w1 + errors_w2
    idx_q, idx_v, idx_s = split_features_by_channel(feature_cols, cfg)

    thresholds = {}
    for ch, idx in [("quality", idx_q), ("volume", idx_v), ("structural", idx_s)]:
        all_scores = _channel_max(feat_scores, idx)

        if n_eval is not None and 0 < n_eval < len(all_scores):
            context_scores = all_scores[:-n_eval]
        else:
            context_scores = all_scores

        if len(context_scores) < 5:
            log.warning(f"[SCORING-DYN] {ch}: solo {len(context_scores)} días de contexto, "
                        f"usando ventana completa ({len(all_scores)} días)")
            context_scores = all_scores

        med = float(np.median(context_scores))
        mad = float(np.median(np.abs(context_scores - med)))
        sigma_est = mad * 1.4826

        if sigma_est < 1e-6:
            sigma_est = float(np.percentile(context_scores, 75)
                              - np.percentile(context_scores, 25)) / 1.35
        if sigma_est < 1e-6:
            sigma_est = float(np.std(context_scores))

        thr = med + k_sigma * sigma_est
        thresholds[ch] = thr
        log.info(f"[SCORING-DYN] {ch}: median={med:.4f} MAD_σ={sigma_est:.4f} "
                 f"thr={thr:.4f}  (context={len(context_scores)}, eval={n_eval or '?'})")

    log.info(f"[SCORING-DYN] Umbrales — Q={thresholds['quality']:.4f} | "
             f"V={thresholds['volume']:.4f} | S={thresholds['structural']:.4f}")
    return thresholds


def compute_dow_volume_baseline(per_col_df: pd.DataFrame, n_eval=None):
    """Mediana y sigma(MAD) del nº de filas por DOW, sobre días de CONTEXTO.
    Devuelve {dow: (median, sigma_mad)}.
    """
    rc = per_col_df.groupby("date")["n_rows"].first().sort_index()
    if n_eval is not None and 0 < n_eval < len(rc):
        rc = rc.iloc[:-n_eval]
    dows = pd.to_datetime(rc.index).dayofweek
    out = {}
    for d in range(7):
        vals = rc.values[dows == d]
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        out[d] = (med, mad * 1.4826)
    return out


def compute_channel_scores(errors_w1, errors_w2, feature_cols, cfg):
    idx_q, idx_v, idx_s = split_features_by_channel(feature_cols, cfg)
    fs = errors_w1 + errors_w2
    return {
        "quality":    _channel_max(fs, idx_q),
        "volume":     _channel_max(fs, idx_v),
        "structural": _channel_max(fs, idx_s),
        "combined":   fs.mean(axis=1),
        "idx_quality":    idx_q,
        "idx_volume":     idx_v,
        "idx_structural": idx_s,
        "feature_scores": fs,
    }


def generate_alerts(errors_w1, errors_w2, dates, feature_cols, thresholds, cfg,
                    row_counts=None, dow_volume_baseline=None):
    cs = compute_channel_scores(errors_w1, errors_w2, feature_cols, cfg)
    thr_q, thr_v, thr_s = thresholds["quality"], thresholds["volume"], thresholds["structural"]
    sq, sv, ss = cs["quality"], cs["volume"], cs["structural"]
    aq, av, as_ = sq > thr_q, sv > thr_v, ss > thr_s

    # ── Guarda de fiabilidad por volumen ─────────────────────────────────────
    low_volume = np.zeros(len(dates), dtype=bool)
    if row_counts is not None and dow_volume_baseline is not None:
        s = cfg.get("scoring", {})
        min_reliable = s.get("min_reliable_rows", 200)
        k_collapse   = s.get("volume_collapse_k", 4.0)
        dows = pd.to_datetime(dates).dayofweek
        for i in range(len(dates)):
            n = row_counts[i]
            if n is None or (isinstance(n, float) and np.isnan(n)):
                continue
            med, sig = dow_volume_baseline.get(int(dows[i]), (np.nan, np.nan))
            is_collapse = (not np.isnan(med)) and (n < med - k_collapse * sig)
            reliable = n >= min_reliable
            av[i] = bool(av[i] or is_collapse)           # un colapso DOW siempre es alerta de volumen
            if not reliable:
                low_volume[i] = True
                aq[i]  = False                            # stats de calidad no fiables a n bajo
                as_[i] = False                            # idem distribución/estructura
                av[i]  = bool(is_collapse)                # ignora delta% ruidoso; solo colapso real

    dominant = np.array([
        "+".join(filter(None, ["Quality"*q, "Volume"*v, "Structural"*s])) or "None"
        for q, v, s in zip(aq, av, as_)
    ])

    alerts_df = pd.DataFrame({
        "date": dates,
        "score_quality": sq, "score_volume": sv, "score_structural": ss,
        "score_combined": cs["combined"],
        "threshold_quality": thr_q, "threshold_volume": thr_v, "threshold_structural": thr_s,
        "alert_quality": aq, "alert_volume": av, "alert_structural": as_,
        "anomaly": aq | av | as_, "dominant_channel": dominant,
        "low_volume": low_volume,
        "confidence_quality":    sq / (thr_q + 1e-10),
        "confidence_volume":     sv / (thr_v + 1e-10),
        "confidence_structural": ss / (thr_s + 1e-10),
    })
    n   = alerts_df["anomaly"].sum()
    nlv = int(low_volume.sum())
    log.info(f"[ALERTAS] {n} anomalías / {len(alerts_df)} días "
             f"({100*n/len(alerts_df):.1f}%)  | {nlv} días bajo volumen (Q/S suprimidos)")
    return alerts_df, cs


def top_features_for_day(day_idx, feature_scores, feature_cols, top_k=10):
    scores  = feature_scores[day_idx]
    top_idx = np.argsort(scores)[::-1][:top_k]
    total   = scores.sum() + 1e-10
    _pfx = {"qual_": "Calidad", "vol_": "Volumen", "dist_": "Distribución",
            "dyn_": "Dinámica", "schema_": "Esquema", "coh_": "Coherencia",
            "day_": "Temporal", "month_": "Temporal"}

    def grp(c):
        return next((g for p, g in _pfx.items() if c.startswith(p)), "Otro")

    return pd.DataFrame({
        "feature":      [feature_cols[i] for i in top_idx],
        "score":        scores[top_idx],
        "pct_of_total": scores[top_idx] / total * 100,
        "group":        [grp(feature_cols[i]) for i in top_idx],
    })
