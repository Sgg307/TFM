"""
explainability.py — Análisis de Causa y Visualización de Anomalías
==================================================================
Tres niveles de explicación:
  1. Contribution Analysis: ranking de features por score W1+W2
  2. SHAP Values: contribución de cada feature al error
  3. Clasificación automática del tipo de anomalía

Visualization:
  - Timeline interactivo (Plotly) con scores por canal
  - Gráfico estático de evolución score vs umbral
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Optional
from core.column_attribution import (
    attribute_features_to_columns,
    rank_columns_across_features,
    format_attribution_report,
)

log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, AttributeError) as e:
    SHAP_AVAILABLE = False
    log.warning(f"[explainability] SHAP no disponible: {type(e).__name__}: {e}")

_CYCLIC_COLS = {"day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"}


# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETACIONES DE NEGOCIO POR FEATURE
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_BUSINESS_MEANING = {
    "qual_mean_pct_null":         "Porcentaje medio de nulos entre columnas",
    "qual_p75_pct_null":          "Columnas con más nulos de lo habitual (percentil 75)",
    "qual_p95_pct_null":          "Columnas con nulos extremos (percentil 95)",
    "qual_max_pct_null":          "Columna con más nulos del día",
    "qual_pct_cols_fully_null":   "Fracción de columnas completamente vacías",
    "qual_pct_cols_over10_null":  "Fracción de columnas con >10% de nulos",
    "qual_pct_cols_zero_null":    "Fracción de columnas perfectamente sanas",
    "qual_gini_pct_null":         "Desigualdad en la distribución de nulos entre columnas",
    "qual_mean_pct_empty":        "Porcentaje medio de cadenas vacías entre columnas",
    "qual_max_pct_empty":         "Columna con más cadenas vacías del día",
    "qual_mean_pct_unknown":      "Porcentaje medio de valores 'unknown' entre columnas",
    "qual_max_pct_unknown":       "Columna con más valores 'unknown' del día",
    "qual_n_cols_any_null":       "Número de columnas con algún nulo",
    "qual_std_pct_null":          "Variabilidad en nulos entre columnas",
    "vol_log_row_count":          "Volumen total de registros del día (escala log)",
    "vol_row_count_raw":          "Volumen total de registros del día",
    "vol_row_count_7d_slope":     "Tendencia de volumen en los últimos 7 días",
    "dist_mean_entropy_cat":      "Diversidad media de categorías — baja si falta un operador",
    "dist_p75_entropy_cat":       "Diversidad categórica en columnas más variables",
    "dist_p25_entropy_cat":       "Diversidad categórica en columnas más estables",
    "dist_std_entropy_cat":       "Dispersión de la diversidad entre columnas categóricas",
    "dist_min_entropy_cat":       "Columna categórica con menos diversidad",
    "dist_mean_hhi":              "Concentración media de categorías — sube si uno domina",
    "dist_max_hhi":               "Columna con mayor concentración de un valor",
    "dist_p25_hhi":               "Concentración en columnas menos concentradas",
    "dist_p75_hhi":               "Concentración en columnas más concentradas",
    "dist_mean_top1_share":       "Cuota media del valor más frecuente por columna",
    "dist_max_top1_share":        "Columna donde un valor domina más",
    "dist_mean_n_cats":           "Número medio de categorías distintas por columna",
    "dist_std_n_cats":            "Variabilidad en cardinalidad entre columnas",
    "dyn_row_count_delta_pct":    "Cambio de volumen respecto al día anterior (%)",
    "dyn_delta_mean_pct_null":    "Cambio en nulos medios vs misma semana anterior",
    "dyn_delta_max_pct_null":     "Cambio en columna con más nulos vs misma semana anterior",
    "dyn_delta_gini_pct_null":    "Cambio en desigualdad de nulos vs misma semana anterior",
    "dyn_delta_mean_entropy_cat": "Cambio en diversidad categórica vs misma semana anterior",
    "dyn_delta_mean_hhi":         "Cambio en concentración vs misma semana anterior",
    "dyn_delta_mean_top1_share":  "Cambio en dominancia del valor más frecuente vs semana anterior",
    "dyn_delta_mean_pct_empty":   "Cambio en cadenas vacías vs misma semana anterior",
    "dyn_delta_mean_pct_unknown": "Cambio en valores 'unknown' vs misma semana anterior",
    "dyn_delta_n_cats":           "Cambio en número de categorías distintas vs semana anterior",
    "dyn_delta_pct_cols_degraded":"Cambio en fracción de columnas en mal estado vs semana anterior",
    "dyn_n_new_cat_values":       "Valores categóricos nuevos respecto a la semana anterior",
    "dyn_n_disappeared_cat_values":"Valores categóricos que desaparecieron vs la semana anterior",
    "dyn_pct_new_cat_values":     "Porcentaje de valores categóricos nuevos vs semana anterior",
    "dyn_pct_disappeared_cat_values": "Porcentaje de valores desaparecidos — alto si falta un operador",
    "schema_n_new_cols":          "Columnas nuevas que no existían ayer",
    "schema_n_total_cols":        "Total de columnas en la tabla hoy",
    "schema_n_cat_cols":          "Número de columnas categóricas",
    "coh_null_x_volume":          "Nulos y caída de volumen juntos — señal de filtrado upstream",
    "coh_pct_cols_degraded":      "Fracción de dimensiones de calidad que empeoran simultáneamente",
    "coh_distribution_drift":     "Drift distribucional total acumulado",
    "coh_volume_quality_ratio":   "Salud ponderada por volumen",
    "coh_null_volume_trend":      "Tendencia adversa: nulos subiendo mientras volumen baja",
    "coh_null_spread_ratio":      "Dispersión de nulos — alto si el problema es localizado",
    "coh_quality_score":          "Score de calidad global del día",
}


def interpret_feature(col: str) -> str:
    return _FEATURE_BUSINESS_MEANING.get(col, col)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASIFICACIÓN DEL TIPO DE ANOMALÍA
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_LABELS = {
    "qual_":   "Calidad (nulos/vacíos)",
    "vol_":    "Volumen",
    "dist_":   "Distribución categórica",
    "dyn_":    "Dinámica (cambio día a día)",
    "schema_": "Esquema",
    "coh_":    "Coherencia",
}

_ANOMALY_TYPE_MAP = {
    "vol_":    "Falta de datos (caída de volumen)",
    "qual_":   "Exceso de nulos o valores vacíos",
    "dist_":   "Cambio en distribución de categorías",
    "dyn_":    "Cambio brusco respecto al día anterior",
    "schema_": "Cambio en el esquema de la tabla",
    "coh_":    "Anomalía de coherencia multidimensional",
}


def classify_anomaly_type(top_features: List[str], cfg: dict = None) -> str:
    if not top_features:
        return "Desconocida"
    group_counts = {prefix: 0 for prefix in _ANOMALY_TYPE_MAP}
    for feat in top_features:
        for prefix in group_counts:
            if feat.startswith(prefix):
                group_counts[prefix] += 1
                break
    dominant = max(group_counts, key=group_counts.get)
    if group_counts[dominant] == 0:
        return "Anomalía estructural genérica"
    return _ANOMALY_TYPE_MAP.get(dominant, "Anomalía multidimensional")


def _feature_group_label(col: str) -> str:
    for prefix, label in _GROUP_LABELS.items():
        if col.startswith(prefix):
            return label
    return "Otro"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_COLORS = {
    "qual_":   "#d62728",
    "vol_":    "#9467bd",
    "dist_":   "#1f77b4",
    "dyn_":    "#ff7f0e",
    "schema_": "#2ca02c",
    "coh_":    "#8c564b",
    "day_":    "#7f7f7f",
    "month_":  "#7f7f7f",
}

def _feature_color(col: str) -> str:
    for prefix, color in _GROUP_COLORS.items():
        if col.startswith(prefix):
            return color
    return "#bcbd22"


def contribution_analysis(
    errors_day:   np.ndarray,
    feature_cols: List[str],
    top_k:        int = 10,
    cfg:          dict = None,
    save_path:    Optional[Path] = None,
    day_label:    str = "",
) -> Dict:
    """Ranking de las top_k features con mayor contribución al score W1+W2."""
    errors_day = errors_day.copy()
    for i, col in enumerate(feature_cols):
        if col in _CYCLIC_COLS:
            errors_day[i] = 0.0

    sorted_idx  = np.argsort(errors_day)[::-1]
    top_idx     = sorted_idx[:top_k]
    top_feat    = [feature_cols[i] for i in top_idx]
    top_err     = errors_day[top_idx]
    total_error = errors_day.sum() + 1e-9
    top_pct     = 100 * top_err / total_error
    anomaly_type = classify_anomaly_type(top_feat, cfg)

    colors = [_feature_color(f) for f in top_feat]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [f[:48] for f in top_feat[::-1]],
        top_err[::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=0.5,
    )
    for bar, pct in zip(bars, top_pct[::-1]):
        ax.text(bar.get_width() + total_error * 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9, color="#333333")
    ax.set_xlabel("Score de Anomalía (W1+W2)", fontsize=11)
    ax.set_title(f"Contribución al Error — {day_label}\nTipo: {anomaly_type}",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [
        plt.Rectangle((0,0), 1, 1, color=c, label=_GROUP_LABELS.get(p, p))
        for p, c in _GROUP_COLORS.items()
        if any(f.startswith(p) for f in top_feat)
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[XAI] Contribución guardada en {save_path}")
    plt.close()

    return {
        "top_features":    top_feat,
        "top_errors":      top_err.tolist(),
        "top_pct":         top_pct.tolist(),
        "anomaly_type":    anomaly_type,
        "total_error":     float(total_error),
        "groups":          [_feature_group_label(f) for f in top_feat],
        "interpretations": [interpret_feature(f) for f in top_feat],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def shap_analysis(
    model, window_tensor, target_tensor,
    feature_cols: List[str],
    background_windows, background_targets,
    save_path: Optional[Path] = None,
    day_label: str = "",
) -> Optional[np.ndarray]:
    """SHAP values sobre el score combinado W1+W2."""
    if not SHAP_AVAILABLE:
        log.warning("[SHAP] SHAP no disponible.")
        return None

    import torch
    import io, sys as _sys

    model.eval()
    model_cpu  = model.cpu()
    window_cpu = window_tensor.cpu()

    def score_fn(target_np: np.ndarray) -> np.ndarray:
        results = []
        for i in range(len(target_np)):
            t = torch.tensor(target_np[i:i+1], dtype=torch.float32)
            w = window_cpu.expand(1, -1, -1)
            with torch.no_grad():
                r1, r2 = model_cpu(w, t)
                err = float(((t - r1) ** 2).mean()) + float(((t - r2) ** 2).mean())
                results.append(err)
        return np.array(results)

    background_np = background_targets.cpu().numpy()[:50]
    target_np     = target_tensor.cpu().numpy()

    log.info(f"[SHAP] Calculando SHAP values para {day_label}...")

    # Silence SHAP verbose output
    import logging as _logging
    for _name in ("shap", "shap.explainers", "shap.explainers._kernel",
                  "shap.utils", "shap._explanation"):
        _logging.getLogger(_name).setLevel(_logging.ERROR)
    _old_stdout = _sys.stdout
    _sys.stdout = io.StringIO()
    try:
        explainer = shap.KernelExplainer(score_fn, background_np)
        shap_vals = explainer.shap_values(target_np, nsamples=100, silent=True)
    finally:
        _sys.stdout = _old_stdout

    shap_abs   = np.abs(shap_vals[0])
    sorted_idx = np.argsort(shap_abs)[::-1][:15]
    top_feat   = [feature_cols[i] for i in sorted_idx]
    top_shap   = shap_vals[0][sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [_feature_color(f) if v > 0 else "#aec7e8"
              for f, v in zip(top_feat[::-1], top_shap[::-1])]
    ax.barh([f[:48] for f in top_feat[::-1]], top_shap[::-1],
            color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (contribución al score W1+W2)", fontsize=11)
    ax.set_title(f"SHAP Analysis — {day_label}", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return shap_vals


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEMÁFORO DE ALERTA
# ─────────────────────────────────────────────────────────────────────────────

def alert_semaphore(confidence: float, label: str = "") -> str:
    suffix = f" {label}" if label else ""
    if confidence > 3.0:
        return f"🔴 CRITICAL  (conf={confidence:.2f}x umbral{suffix})"
    elif confidence > 1.0:
        return f"🟡 WARNING   (conf={confidence:.2f}x umbral{suffix})"
    else:
        return f"🟢 OK        (conf={confidence:.2f}x umbral{suffix})"


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIMELINE INTERACTIVO (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_timeline(
    alerts_df: pd.DataFrame,
    cfg: dict,
    save_path: Optional[Path] = None,
) -> go.Figure:
    thr_q = float(alerts_df["threshold_quality"].iloc[0]) if "threshold_quality" in alerts_df.columns else 1.0
    thr_v = float(alerts_df["threshold_volume"].iloc[0])  if "threshold_volume"  in alerts_df.columns else 1.0
    thr_s = float(alerts_df["threshold_structural"].iloc[0]) if "threshold_structural" in alerts_df.columns else 1.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=alerts_df["date"], y=alerts_df["score_structural"],
        name="Score Structural", line=dict(color="#1f77b4", width=1.5), mode="lines",
    ))
    if "score_volume" in alerts_df.columns:
        fig.add_trace(go.Scatter(
            x=alerts_df["date"], y=alerts_df["score_volume"],
            name="Score Volume", line=dict(color="#9467bd", width=1.5), mode="lines",
        ))
    fig.add_trace(go.Scatter(
        x=alerts_df["date"], y=alerts_df["score_quality"],
        name="Score Quality", line=dict(color="#d62728", width=1.5), mode="lines",
    ))

    anomaly_days = alerts_df[alerts_df["anomaly"]]
    if not anomaly_days.empty:
        score_cols = [c for c in ["score_quality", "score_volume", "score_structural"]
                      if c in anomaly_days.columns]
        anomaly_score = anomaly_days[score_cols].max(axis=1)
        conf_cols     = [c for c in ["confidence_quality", "confidence_volume", "confidence_structural"]
                         if c in anomaly_days.columns]
        conf_max      = anomaly_days[conf_cols].max(axis=1)
        fig.add_trace(go.Scatter(
            x=anomaly_days["date"], y=anomaly_score,
            name="🔴 Anomalía", mode="markers",
            marker=dict(color="#d62728", size=10, symbol="circle",
                        line=dict(width=1, color="white")),
            customdata=np.column_stack([anomaly_days["dominant_channel"].values, conf_max.values]),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<br>Canal: %{customdata[0]}<br>Conf: %{customdata[1]:.2f}x<extra></extra>",
        ))

    fig.add_hline(y=thr_s, line_dash="dash", line_color="#1f77b4", line_width=1.5,
                  annotation_text=f"Umbral Structural ({thr_s:.4f})", annotation_position="top left")
    if "threshold_volume" in alerts_df.columns:
        fig.add_hline(y=thr_v, line_dash="dash", line_color="#9467bd", line_width=1.5,
                      annotation_text=f"Umbral Volume ({thr_v:.4f})", annotation_position="bottom left")
    fig.add_hline(y=thr_q, line_dash="dash", line_color="#d62728", line_width=1.5,
                  annotation_text=f"Umbral Quality ({thr_q:.4f})", annotation_position="top right")

    n_anomalies = alerts_df["anomaly"].sum()
    fig.update_layout(
        title=dict(
            text=(f"<b>Data Health Monitor</b><br>"
                  f"<sub>{len(alerts_df)} días | {n_anomalies} anomalías ({100*n_anomalies/len(alerts_df):.1f}%)</sub>"),
        ),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Anomaly Score (W1+W2)", showgrid=True, gridcolor="#f0f0f0"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white", height=500,
    )

    if save_path:
        fig.write_html(str(save_path))
        log.info(f"[VIZ] Timeline guardado en {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVOLUCIÓN DE SCORES vs UMBRAL FIJO
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_evolution(
    alerts_df: pd.DataFrame,
    cfg: dict,
    save_path: Optional[Path] = None,
) -> None:
    if "score_combined" in alerts_df.columns:
        signal = alerts_df["score_combined"].values
    elif "score_quality" in alerts_df.columns:
        signal = alerts_df[["score_quality", "score_structural"]].max(axis=1).values
    else:
        log.warning("[VIZ] No se encontraron columnas de score")
        return

    if "threshold_quality" in alerts_df.columns:
        thr = float(alerts_df[["threshold_quality", "threshold_structural"]].min(axis=1).iloc[0])
    else:
        thr = 1.0

    dates        = alerts_df["date"].values
    anomaly_mask = alerts_df["anomaly"].values if "anomaly" in alerts_df.columns else (signal > thr)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, signal, color="#1f77b4", linewidth=1.2, label="Score combinado", zorder=3)
    ax.axhline(y=thr, color="#d62728", linewidth=1.5, linestyle="--",
               label=f"Umbral (pct={cfg['scoring'].get('threshold_percentile', 99.5)})", zorder=4)
    ax.fill_between(dates, signal, thr, where=(signal > thr),
                    alpha=0.25, color="#d62728", label="Zona anómala", zorder=2)
    if anomaly_mask.any():
        ax.scatter(dates[anomaly_mask], signal[anomaly_mask],
                   color="#d62728", s=30, zorder=5, label="Anomalía")

    ax.set_xlabel("Fecha"); ax.set_ylabel("Anomaly Score")
    ax.set_title("Score vs Umbral del Modelo", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = Path(cfg.get("paths", {}).get("plots_dir", "plots")) / "threshold_evolution.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. INFORME COMPLETO DE ANOMALÍA
# ─────────────────────────────────────────────────────────────────────────────

def explain_anomaly_day(
    day_idx:        int,
    feature_scores: np.ndarray,
    scaled_data:    np.ndarray,
    alerts_df:      pd.DataFrame,
    feature_cols:   List[str],
    cfg:            dict,
    model=None,
    save_dir:       Optional[Path] = None,
    per_col_df:     Optional[pd.DataFrame] = None,
) -> Dict:
    row      = alerts_df.iloc[day_idx]
    date_str = str(row["date"])[:10]
    conf_q   = float(row.get("confidence_quality", 0))
    conf_v   = float(row.get("confidence_volume", 0))
    conf_s   = float(row.get("confidence_structural", 0))
    day_label = f"{date_str} | conf_Q={conf_q:.2f}x | conf_S={conf_s:.2f}x"

    log.info(f"[XAI] Explicando anomalía del {date_str} | Canal: {row['dominant_channel']}")
    scores_day = feature_scores[day_idx]

    # 1. Contribution Analysis
    contrib_path = (save_dir / f"contribution_{date_str}.png") if save_dir else None
    contrib = contribution_analysis(
        scores_day, feature_cols, top_k=10, cfg=cfg,
        save_path=contrib_path, day_label=day_label,
    )

    # 2. SHAP
    shap_vals = None
    if model is not None and SHAP_AVAILABLE:
        import torch
        seq_len = cfg["tranad"]["seq_len"]
        if day_idx >= seq_len:
            window_np = scaled_data[day_idx - seq_len: day_idx]
            target_np = scaled_data[day_idx]
            window_t  = torch.tensor(window_np[np.newaxis], dtype=torch.float32)
            target_t  = torch.tensor(target_np[np.newaxis], dtype=torch.float32)

            bg_start = max(0, day_idx - seq_len - 60)
            bg_end   = max(0, day_idx - seq_len)
            bg_w = (torch.tensor(scaled_data[bg_start:bg_end][np.newaxis], dtype=torch.float32)
                    if bg_end > bg_start else window_t)
            bg_t = torch.tensor(scaled_data[bg_start:bg_end], dtype=torch.float32)

            shap_path = (save_dir / f"shap_{date_str}.png") if save_dir else None
            shap_vals = shap_analysis(
                model, window_t, target_t, feature_cols,
                bg_w, bg_t, shap_path, day_label,
            )

    # 3. Semáforo
    sem_q = alert_semaphore(conf_q, "Quality")
    sem_v = alert_semaphore(conf_v, "Volume")
    sem_s = alert_semaphore(conf_s, "Structural")

    print(f"\n{'━'*60}")
    print(f"  ANOMALÍA DETECTADA — {date_str}")
    print(f"{'━'*60}")
    print(f"  Canal: {row['dominant_channel']}  |  Tipo: {contrib['anomaly_type']}")
    print(f"  {sem_q}\n  {sem_v}\n  {sem_s}")
    print(f"\n  Top 5 features culpables:")
    for i, (feat, pct, interp) in enumerate(zip(
        contrib["top_features"][:5], contrib["top_pct"][:5], contrib["interpretations"][:5],
    )):
        print(f"    {i+1}. {feat:<44} ({pct:.1f}%)")
        print(f"       → {interp}")

    # 4. Atribución column-level
    attribution    = None
    column_ranking = None
    if per_col_df is not None:
        baseline_days = cfg.get("explainability", {}).get("attribution_baseline_days", 30)
        top_k_cols    = cfg.get("explainability", {}).get("attribution_top_k_columns", 5)
        try:
            attribution = attribute_features_to_columns(
                contrib["top_features"], contrib["top_pct"],
                pd.Timestamp(date_str), per_col_df, baseline_days, top_k_cols,
            )
            column_ranking = rank_columns_across_features(attribution, top_k=top_k_cols)
            print(f"\n  Columnas raw responsables:")
            print(format_attribution_report(column_ranking, baseline_days))
        except Exception as exc:
            log.warning(f"[XAI] Atribución column-level falló: {exc}")

    return {
        "date":           date_str,
        "contribution":   contrib,
        "shap":           shap_vals,
        "alert_row":      row.to_dict(),
        "attribution":    attribution,
        "column_ranking": column_ranking,
    }
