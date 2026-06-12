"""alert_fusion.py — Fusión Multi-Tier (v2)
================================================

Filosofía de la v2
------------------
Cada tier es un detector independiente. La fusión NO combina sus salidas en
un score heurístico (la "confianza 80%" de la v1). En su lugar, por cada día
y cada tier evaluado se reporta:

    fires    : booleano — el tier ha superado su umbral propio
    severity : ratio "veces sobre umbral" (1.0 = en umbral, 5.0 = 5× sobre)

A nivel de día:

    n_firing     : cuántos tiers disparan (0..n_evaluados)
    max_severity : máximo de las severities de los tiers que disparan
    anomaly      : (n_firing >= 1)

Diseño explícito: si un tier no se evalúa (e.g. `tiers=[1,2]`) NO entra en
`n_firing` ni en `n_total` ni en `tiers_present` — el módulo es agnóstico a
qué tiers están activos en cada llamada.

Lógica por tier
---------------
T1 (Stats column-level):
    fires si  max_z >= cfg.tier1.z_threshold  AND  n_flagged >= cfg.tier1.min_concentrated_cols
    severity = max_z / cfg.tier1.z_threshold
T2 (PCA cross-column):
    fires si  tier2_result["anomaly"]
    severity = z_score / cfg.tier2.z_threshold
T3 (TranAD agregado):
    fires si  cualquier alert_{quality,volume,structural}
    severity = max(score_q/thr_q, score_v/thr_v, score_s/thr_s)
T4 (Row-level AE):
    fires si  pct_anomalous > threshold_pct (umbral dinámico de inferencia)
    severity = pct_anomalous / threshold_pct
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults — usados solo si el cfg no los provee
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "tier1_z_threshold":          4.0,
    "tier1_min_concentrated_cols": 2,    # opción (C) del rediseño
    "tier2_z_threshold":          3.0,
    "tier4_threshold_pct":        0.05,   # 5% — fallback si no calibramos dinámicamente
    "tier4_k_sigma":              4.0,    # K×MAD para calibración dinámica day-level
    "tier4_min_threshold_pct":    0.05,   # piso absoluto del umbral day-level
    "tier4_low_volume_pct":       0.01,   # menos del 1% del volumen típico = ruido estadístico
    "tier4_low_volume_abs":       200,    # o menos de 200 filas en absoluto
}


# ─────────────────────────────────────────────────────────────────────────────
# T4: umbral DAY-level dinámico
# ─────────────────────────────────────────────────────────────────────────────
def compute_t4_day_threshold(
    t4_summary: pd.DataFrame,
    k_sigma: float = 4.0,
    min_threshold: float = 0.05,
) -> float:
    """Calibra el umbral day-level (sobre pct_anomalous) sobre la ventana,
    con el mismo principio que T3: median + K × 1.4826 × MAD.

    Excluye días de bajo volumen (n_total < 10% del mediano) para evitar que
    domingos con 30 filas y 80% "anómalas" contaminen el baseline.

    Floor en `min_threshold` para no caer a valores absurdamente bajos cuando
    todos los días son sanos y muy similares.
    """
    if t4_summary is None or len(t4_summary) == 0:
        return min_threshold

    pct = t4_summary["pct_anomalous"].astype(float).values
    if "n_total" in t4_summary.columns:
        n = t4_summary["n_total"].astype(float).values
        median_n = float(np.median(n))
        if median_n > 0:
            valid = n >= 0.10 * median_n
            if valid.sum() >= 3:
                pct = pct[valid]

    if len(pct) == 0:
        return min_threshold

    median_pct = float(np.median(pct))
    mad        = float(np.median(np.abs(pct - median_pct)))
    thr        = median_pct + k_sigma * 1.4826 * mad
    return max(thr, min_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# combine_day — núcleo de la fusión
# ─────────────────────────────────────────────────────────────────────────────
def combine_day(
    date,
    tier1_result: Optional[Dict] = None,
    tier2_result: Optional[Dict] = None,
    tier3_result: Optional[Dict] = None,
    tier4_result: Optional[Dict] = None,
    cfg: Optional[Dict] = None,
) -> Dict:
    """Combina veredictos día-a-día. Cualquier tierN puede ser None.

    El resultado contiene:
      - Los flags globales (anomaly, n_firing, max_severity, ...)
      - Un dict `tiers` con la entrada por cada tier evaluado:
            {"T1": {"fires", "severity", "label", ...payload}, ...}
      - Las claves `_raw_t1`...`_raw_t4` con los dicts originales (sin tocar),
        que el dashboard y los reportes usan para extraer explicabilidad.
    """
    cfg = cfg or {}
    tier1_cfg = cfg.get("tier1", {})
    tier2_cfg = cfg.get("tier2", {})

    z_thr_t1  = float(tier1_cfg.get("z_threshold",          _DEFAULTS["tier1_z_threshold"]))
    min_c_t1  = int(  tier1_cfg.get("min_concentrated_cols", _DEFAULTS["tier1_min_concentrated_cols"]))
    z_thr_t2  = float(tier2_cfg.get("z_threshold",          _DEFAULTS["tier2_z_threshold"]))

    tiers: Dict[str, Dict] = {}

    # ── T1 ───────────────────────────────────────────────────────────────────
    if tier1_result is not None:
        tiers["T1"] = _build_t1_entry(tier1_result, z_thr_t1, min_c_t1)

    # ── T2 ───────────────────────────────────────────────────────────────────
    if tier2_result is not None:
        tiers["T2"] = _build_t2_entry(tier2_result, z_thr_t2)

    # ── T3 ───────────────────────────────────────────────────────────────────
    if tier3_result is not None:
        tiers["T3"] = _build_t3_entry(tier3_result)

    # ── T4 ───────────────────────────────────────────────────────────────────
    if tier4_result is not None:
        tiers["T4"] = _build_t4_entry(tier4_result)

    # ── Veredicto global ─────────────────────────────────────────────────────
    firing       = [(name, t) for name, t in tiers.items() if t["fires"]]
    n_firing     = len(firing)
    n_total      = len(tiers)
    max_severity = max((t["severity"] for _, t in firing), default=0.0)
    tiers_firing = [name for name, _ in firing]

    return {
        "date":         pd.Timestamp(date),
        "anomaly":      n_firing >= 1,
        "n_firing":     n_firing,
        "n_total":      n_total,
        "max_severity": float(max_severity),
        "tiers_firing": tiers_firing,
        "tiers":        tiers,
        # Compat con código antiguo (report.py, format_executive, etc.).
        # No usar para lógica nueva — la canónica es `tiers` + `tiers_firing`.
        "confidence":   _legacy_confidence_alias(max_severity, n_firing >= 1),
        "n_tiers":      n_firing,
        "type":         "",
        # Acceso a payloads completos (para extraer top_columns, details_df, ...)
        "_raw_t1": tier1_result, "_raw_t2": tier2_result,
        "_raw_t3": tier3_result, "_raw_t4": tier4_result,
        # Alias usados por el dashboard antiguo y reports
        "tier1": tier1_result, "tier2": tier2_result,
        "tier3": tier3_result, "tier4": tier4_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constructores por tier
# ─────────────────────────────────────────────────────────────────────────────
def _build_t1_entry(t1: Dict, z_thr: float, min_cols: int) -> Dict:
    """T1 entry. Dispara si hay 'concentración suficiente' de columnas con
    z-score fuerte (criterio configurable — opción C del rediseño)."""
    top      = t1.get("top_columns")
    details  = t1.get("details_df")
    max_z    = float(t1.get("max_z", 0.0) or 0.0)
    n_flagged = 0
    top_col   = None
    top_stat  = ""

    if isinstance(top, pd.DataFrame) and not top.empty and "n_flagged" in top.columns:
        flagged_cols = top[top["n_flagged"] > 0]
        n_flagged    = len(flagged_cols)
        if n_flagged > 0:
            top_col = str(flagged_cols.iloc[0]["column"])

    # Si max_z no venía en el dict de resultado, lo deducimos del details_df.
    if max_z <= 0.0 and isinstance(details, pd.DataFrame) and not details.empty:
        flagged = details[details["flagged"]] if "flagged" in details.columns else details
        if not flagged.empty and "z_score" in flagged.columns:
            max_z = float(flagged["z_score"].abs().max())
            # Stat más fuerte (para descripción rápida en el dashboard)
            try:
                best_idx = flagged["z_score"].abs().idxmax()
                top_stat = str(flagged.loc[best_idx].get("stat", ""))
            except Exception:
                pass

    fires    = (max_z >= z_thr) and (n_flagged >= min_cols)
    severity = (max_z / z_thr) if z_thr > 0 else 0.0

    return {
        "fires":    fires,
        "severity": float(severity if fires else 0.0),
        "label":    "Stats column-level",
        "max_z":    float(max_z),
        "n_cols":   int(n_flagged),
        "top_col":  top_col,
        "top_stat": top_stat,
        "low_volume": bool(t1.get("low_volume", False)),
    }


def _build_t2_entry(t2: Dict, z_thr: float) -> Dict:
    """T2 entry. tier2.score_day ya devuelve 'anomaly' booleano y 'z_score'."""
    fires = bool(t2.get("anomaly", False))
    z     = float(t2.get("z_score", 0.0) or 0.0)
    severity = (z / z_thr) if z_thr > 0 else 0.0

    # Top columnas afectadas (cuando T2 produce explicabilidad por columna).
    top_cols: List[str] = []
    top = t2.get("top_columns")
    if isinstance(top, pd.DataFrame) and not top.empty:
        col_field = "column" if "column" in top.columns else top.columns[0]
        top_cols  = top[col_field].head(3).astype(str).tolist()

    return {
        "fires":    fires,
        "severity": float(severity if fires else 0.0),
        "label":    "PCA cross-column",
        "z_score":  float(z),
        "top_cols": top_cols,
    }


def _build_t3_entry(t3: Dict) -> Dict:
    """T3 entry. Usamos los threshold_* y alert_* que ya emite generate_alerts.

    severity = max(score_canal / threshold_canal) sobre los canales que dispararon.
    Si por alguna razón no nos llegan los thresholds, fallback: 1.0 si fires.
    """
    fires = bool(t3.get("anomaly", False))

    q_s, q_t = float(t3.get("score_quality",    0)), float(t3.get("threshold_quality",    0))
    v_s, v_t = float(t3.get("score_volume",     0)), float(t3.get("threshold_volume",     0))
    s_s, s_t = float(t3.get("score_structural", 0)), float(t3.get("threshold_structural", 0))

    aq = bool(t3.get("alert_quality",    False))
    av = bool(t3.get("alert_volume",     False))
    as_ = bool(t3.get("alert_structural", False))

    def _r(score, thr, active):
        if not active or thr <= 0:
            return 0.0
        return score / thr

    sev_q = _r(q_s, q_t, aq)
    sev_v = _r(v_s, v_t, av)
    sev_s = _r(s_s, s_t, as_)
    severity = max(sev_q, sev_v, sev_s)

    if fires and severity == 0.0:
        # Fallback: no nos llegaron thresholds → severity neutral 1.0
        severity = 1.0

    return {
        "fires":    fires,
        "severity": float(severity if fires else 0.0),
        "label":    "TranAD agregado",
        "channel":  str(t3.get("dominant_channel", "")),
        "sev_quality":    float(sev_q),
        "sev_volume":     float(sev_v),
        "sev_structural": float(sev_s),
        "alert_quality":    aq,
        "alert_volume":     av,
        "alert_structural": as_,
        "score_quality":    q_s, "score_volume": v_s, "score_structural": s_s,
        "low_volume":      bool(t3.get("low_volume", False)),
    }


def _build_t4_entry(t4: Dict) -> Dict:
    """T4 entry. Espera al menos {pct_anomalous, threshold_pct, top_col}."""
    pct = float(t4.get("pct_anomalous", 0.0) or 0.0)
    thr = float(t4.get("threshold_pct", _DEFAULTS["tier4_threshold_pct"]) or _DEFAULTS["tier4_threshold_pct"])
    fires = pct > thr
    severity = (pct / thr) if thr > 0 else 0.0

    return {
        "fires":    fires,
        "severity": float(severity if fires else 0.0),
        "label":    "Row-level AE",
        "pct_anomalous": float(pct),
        "threshold_pct": float(thr),
        "top_col":       str(t4.get("top_col", "") or ""),
        "n_total":       int(t4.get("n_total", 0) or 0),
        "n_anomalous":   int(t4.get("n_anomalous", 0) or 0),
        "top_drivers":   t4.get("top_drivers", []),  # lista opcional [{col, ratio_vs_normal}, ...]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compat alias
# ─────────────────────────────────────────────────────────────────────────────
def _legacy_confidence_alias(max_severity: float, anomaly: bool) -> float:
    """Normaliza max_severity al [0, 1] capeado en severity=5 → conf=1.0.

    Existe solo para no romper formatters de texto antiguos en report.py
    (`_conf_bar`, etc.). El dashboard usa `max_severity` directamente.
    """
    if not anomaly:
        return 0.0
    return float(min(max_severity / 5.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# score_period — itera combine_day sobre una ventana
# ─────────────────────────────────────────────────────────────────────────────
def score_period(
    per_col_df,
    scaled_data,
    dates,
    tier1,
    tier2,
    eval_dates,
    excluded_cols=None,
    tier3_alerts: Optional[pd.DataFrame] = None,
    tier4_summary: Optional[pd.DataFrame] = None,
    tier4_threshold: Optional[float] = None,
    cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """Evalúa una lista de fechas combinando los tiers disponibles.

    Args:
        tier1, tier2:    detectores instanciados (o None si ese tier no se evalúa)
        tier3_alerts:    DataFrame de generate_alerts (o None)
        tier4_summary:   DataFrame del row-level con cols
                         [date, pct_anomalous, top_col, n_total?, n_anomalous?]
        tier4_threshold: umbral dinámico usado en la inferencia row-level (float)
        cfg:             config dict — para leer tier{1,2}.z_threshold y demás.

    Returns: period_df con una fila por fecha y columnas planas para downstream
             consumers (dashboard, report.py).
    """
    # ── T3 lookup ────────────────────────────────────────────────────────────
    t3_lookup: Dict[pd.Timestamp, Dict] = {}
    if tier3_alerts is not None and len(tier3_alerts):
        for _, r in tier3_alerts.iterrows():
            t3_lookup[pd.Timestamp(r["date"])] = {
                "anomaly":          bool(r.get("anomaly", False)),
                "dominant_channel": r.get("dominant_channel", ""),
                "score_quality":    float(r.get("score_quality",    0)),
                "score_volume":     float(r.get("score_volume",     0)),
                "score_structural": float(r.get("score_structural", 0)),
                "threshold_quality":    float(r.get("threshold_quality",    0)),
                "threshold_volume":     float(r.get("threshold_volume",     0)),
                "threshold_structural": float(r.get("threshold_structural", 0)),
                "alert_quality":    bool(r.get("alert_quality",    False)),
                "alert_volume":     bool(r.get("alert_volume",     False)),
                "alert_structural": bool(r.get("alert_structural", False)),
                "low_volume":       bool(r.get("low_volume",       False)),
            }

    # ── T4 lookup ────────────────────────────────────────────────────────────
    t4_lookup: Dict[pd.Timestamp, Dict] = {}
    if tier4_summary is not None and len(tier4_summary):
        # Si no nos pasan threshold, calibramos dinámicamente.
        if tier4_threshold is None:
            t4_cfg = (cfg or {}).get("tier4", {})
            k = float(t4_cfg.get("k_sigma",          _DEFAULTS["tier4_k_sigma"]))
            m = float(t4_cfg.get("min_threshold_pct", _DEFAULTS["tier4_min_threshold_pct"]))
            thr = compute_t4_day_threshold(tier4_summary, k_sigma=k, min_threshold=m)
        else:
            thr = float(tier4_threshold)

        # Low-volume guard: días con MUY pocas filas no disparan (ruido)
        t4_cfg = (cfg or {}).get("tier4", {})
        lvp = float(t4_cfg.get("low_volume_pct", _DEFAULTS["tier4_low_volume_pct"]))
        lva = int(  t4_cfg.get("low_volume_abs", _DEFAULTS["tier4_low_volume_abs"]))
        median_n = float(tier4_summary["n_total"].median()) if "n_total" in tier4_summary.columns else 0.0

        for _, r in tier4_summary.iterrows():
            d = pd.Timestamp(r["date"])
            n = int(r.get("n_total", 0)) if "n_total" in r else 0
            low_vol = (n < lva) or (median_n > 0 and n < lvp * median_n)
            t4_lookup[d] = {
                "pct_anomalous": float(r.get("pct_anomalous", 0)),
                "threshold_pct": thr,
                "top_col":       str(r.get("top_col", "")),
                "n_total":       n,
                "n_anomalous":   int(r.get("n_anomalous", 0)) if "n_anomalous" in r else 0,
                "top_drivers":   r.get("top_drivers", []) if "top_drivers" in r else [],
                "low_volume":    bool(low_vol),
            }

    # Precompute date→idx para T2.
    date_index = {pd.Timestamp(d): i for i, d in enumerate(dates)} if tier2 is not None else {}

    rows = []
    for target_date in eval_dates:
        ts = pd.Timestamp(target_date)

        r1 = tier1.score_day(per_col_df, target_date, excluded_cols) if tier1 is not None else None
        r2 = None
        if tier2 is not None:
            di = date_index.get(ts)
            if di is not None and di < len(scaled_data):
                r2 = tier2.score_day(scaled_data[di], target_date)
            else:
                r2 = {"anomaly": False, "z_score": 0.0, "total_error": 0.0}
        r3 = t3_lookup.get(ts)
        r4 = t4_lookup.get(ts)

        combined = combine_day(target_date, r1, r2, r3, r4, cfg=cfg)
        rows.append(_flatten_combined(combined))

    return pd.DataFrame(rows)


def _flatten_combined(c: Dict) -> Dict:
    """Convierte la salida de combine_day a una fila plana para period_df."""
    out = {
        "date":         c["date"],
        "anomaly":      c["anomaly"],
        "n_firing":     c["n_firing"],
        "n_total":      c["n_total"],
        "max_severity": c["max_severity"],
        "tiers_firing": ",".join(c["tiers_firing"]),
        # Compat con report.py / format_executive
        "confidence":   c["confidence"],
        "n_tiers":      c["n_tiers"],
        "type":         c["type"],
    }
    tiers = c["tiers"]
    if "T1" in tiers:
        t = tiers["T1"]
        out["t1_fires"]     = t["fires"]
        out["t1_severity"]  = t["severity"]
        out["t1_n_flagged"] = t["n_cols"]
        out["t1_max_z"]     = t["max_z"]
        out["t1_low_volume"] = t["low_volume"]
    if "T2" in tiers:
        t = tiers["T2"]
        out["t2_fires"]    = t["fires"]
        out["t2_severity"] = t["severity"]
        out["t2_anomaly"]  = t["fires"]
        out["t2_z_score"]  = t["z_score"]
    if "T3" in tiers:
        t = tiers["T3"]
        out["t3_fires"]    = t["fires"]
        out["t3_severity"] = t["severity"]
        out["t3_anomaly"]  = t["fires"]
        out["t3_channel"]  = t["channel"]
    if "T4" in tiers:
        t = tiers["T4"]
        out["t4_fires"]    = t["fires"]
        out["t4_severity"] = t["severity"]
        out["t4_anomaly"]  = t["fires"]
        out["t4_pct"]      = t["pct_anomalous"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Merge tardío de T4 (cuando run_multi_tier_mode no lo computa)
# ─────────────────────────────────────────────────────────────────────────────
def merge_tier4_into_period(
    period_df: pd.DataFrame,
    t4_summary: Optional[pd.DataFrame],
    t4_threshold: Optional[float] = None,
    cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """Añade T4 a un period_df ya producido sin T4 y recomputa el veredicto.

    Útil cuando T4 se calcula en un paso posterior al `run_multi_tier_mode`
    (caso actual de `dev_tools.runner.monitor`). NO toca filas para las que
    no hay datos T4.

    Si `t4_threshold` viene None, se calibra dinámicamente sobre la propia
    ventana (median + K×MAD de pct_anomalous, K=4 por defecto, configurable
    vía cfg["tier4"]["k_sigma"]).
    """
    if t4_summary is None or len(t4_summary) == 0:
        return period_df

    # Threshold dinámico day-level si no se especifica.
    if t4_threshold is None:
        t4_cfg = (cfg or {}).get("tier4", {})
        k = float(t4_cfg.get("k_sigma",          _DEFAULTS["tier4_k_sigma"]))
        m = float(t4_cfg.get("min_threshold_pct", _DEFAULTS["tier4_min_threshold_pct"]))
        thr = compute_t4_day_threshold(t4_summary, k_sigma=k, min_threshold=m)
    else:
        thr = float(t4_threshold)

    # Low-volume guard: relativo (<1% mediano) O absoluto (<200 filas)
    t4_cfg_root = (cfg or {}).get("tier4", {})
    lvp = float(t4_cfg_root.get("low_volume_pct", _DEFAULTS["tier4_low_volume_pct"]))
    lva = int(  t4_cfg_root.get("low_volume_abs", _DEFAULTS["tier4_low_volume_abs"]))
    median_n = float(t4_summary["n_total"].median()) if "n_total" in t4_summary.columns else 0.0

    t4_lookup: Dict[pd.Timestamp, tuple] = {}
    for _, r in t4_summary.iterrows():
        n = int(r.get("n_total", 0)) if "n_total" in t4_summary.columns else 0
        low_vol = (n < lva) or (median_n > 0 and n < lvp * median_n)
        t4_lookup[pd.Timestamp(r["date"])] = (
            float(r.get("pct_anomalous", 0)),
            str(r.get("top_col", "")),
            bool(low_vol),
            n,
        )

    df = period_df.copy()
    if "t4_fires" not in df.columns:
        df["t4_fires"]      = False
        df["t4_severity"]   = 0.0
        df["t4_pct"]        = np.nan
        df["t4_anomaly"]    = False
        df["t4_low_volume"] = False
        df["t4_n_total"]    = 0

    for i, row in df.iterrows():
        d = pd.Timestamp(row["date"])
        if d not in t4_lookup:
            continue
        pct, _top, low_vol, n = t4_lookup[d]
        fires = (pct > thr) and (not low_vol)
        sev   = (pct / thr) if thr > 0 else 0.0
        df.at[i, "t4_pct"]        = pct
        df.at[i, "t4_fires"]      = fires
        df.at[i, "t4_anomaly"]    = fires
        df.at[i, "t4_severity"]   = float(sev if fires else 0.0)
        df.at[i, "t4_low_volume"] = low_vol
        df.at[i, "t4_n_total"]    = n

    # Recompute global verdict ───────────────────────────────────────────────
    tier_fires_cols = [c for c in ("t1_fires", "t2_fires", "t3_fires", "t4_fires")
                       if c in df.columns]
    tier_sev_cols   = [c for c in ("t1_severity", "t2_severity", "t3_severity", "t4_severity")
                       if c in df.columns]

    df["n_firing"]     = df[tier_fires_cols].sum(axis=1).astype(int)
    df["n_total"]      = len(tier_fires_cols)
    df["max_severity"] = df[tier_sev_cols].max(axis=1) if tier_sev_cols else 0.0
    df["anomaly"]      = df["n_firing"] >= 1

    name_map = {"t1_fires": "T1", "t2_fires": "T2", "t3_fires": "T3", "t4_fires": "T4"}
    df["tiers_firing"] = df.apply(
        lambda r: ",".join(name_map[c] for c in tier_fires_cols if r.get(c, False)),
        axis=1,
    )
    df["confidence"] = (df["max_severity"] / 5.0).clip(upper=1.0).where(df["anomaly"], 0.0)
    df["n_tiers"]    = df["n_firing"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Formatters de texto (verbose mode, terminal)
# ─────────────────────────────────────────────────────────────────────────────
_CHANNEL_ES = {
    "Quality": "Calidad", "Volume": "Volumen", "Structural": "Estructura",
}


def format_full_report(combined: Dict) -> str:
    """Reporte texto para verbose=True. API simplificada para fusion v2."""
    date = pd.Timestamp(combined["date"]).strftime("%Y-%m-%d")
    if not combined.get("anomaly"):
        return f"📅 {date}  |  ✅ Normal"

    n_firing = combined.get("n_firing", 0)
    n_total  = combined.get("n_total", 0)
    max_sev  = combined.get("max_severity", 0.0)
    tiers    = combined.get("tiers", {}) or {}

    lines = [
        "═" * 70,
        f"📅 {date}  |  🚨 ANOMALÍA  |  {n_firing}/{n_total} detectores  |  sev. máx {max_sev:.1f}×",
        "─" * 70,
    ]
    for name in ("T1", "T2", "T3", "T4"):
        info = tiers.get(name)
        if info is None:
            continue
        label = info.get("label", name)
        if info["fires"]:
            sev = info["severity"]
            extra = _tier_text_extra(name, info)
            lines.append(f"  {name} {label}: 🚨 {sev:.1f}× sobre umbral  {extra}")
        else:
            lines.append(f"  {name} {label}: ✅ normal")
    lines.append("═" * 70)
    return "\n".join(lines)


def _tier_text_extra(name: str, info: Dict) -> str:
    if name == "T1":
        return f"({info.get('n_cols', 0)} cols, max z={info.get('max_z', 0):.1f})"
    if name == "T2":
        return f"(z={info.get('z_score', 0):.1f})"
    if name == "T3":
        ch = _CHANNEL_ES.get(info.get("channel", ""), info.get("channel", ""))
        return f"(canal: {ch})"
    if name == "T4":
        return f"({info.get('pct_anomalous', 0)*100:.1f}% filas, top: {info.get('top_col', '')})"
    return ""


def summarize_period(period_df: pd.DataFrame) -> str:
    n, na = len(period_df), int(period_df["anomaly"].sum())
    lines = [
        "═" * 70,
        f"  RESUMEN — {n} días evaluados | {na} anomalías ({100*na/max(n,1):.1f}%)",
    ]
    if na > 0 and "n_firing" in period_df.columns:
        consensus = int((period_df["n_firing"] >= 2).sum())
        if consensus:
            lines.append(f"  Consenso ≥2 tiers: {consensus} día(s)")
        if "max_severity" in period_df.columns:
            anom_df = period_df[period_df["anomaly"]]
            lines.append(f"  Severidad máxima de la ventana: {anom_df['max_severity'].max():.1f}×")
    lines.append("═" * 70)
    return "\n".join(lines)


def format_tier1_report(result: Dict) -> str:
    """Stub mínimo para backward-compat con report.py antiguo. La explicabilidad
    detallada se renderiza ahora en el dashboard."""
    top = result.get("top_columns")
    if not isinstance(top, pd.DataFrame) or top.empty:
        return "    (sin columnas flagged)"
    flagged = top[top["n_flagged"] > 0] if "n_flagged" in top.columns else top
    return "\n".join(f"    → {r['column']}" for _, r in flagged.head(8).iterrows())
