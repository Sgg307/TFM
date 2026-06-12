"""
dashboard.py — Data Health Dashboard (v2)
==========================================

Diseño:
  - Paleta light, profesional (slate/blue + severity = ámbar/naranja/rojo).
  - Adaptativo a `tiers_present`: si la inferencia se hizo con tiers=[1, 2],
    el dashboard solo renderiza pills T1 y T2.
  - Por cada día y cada tier evaluado: pill con "X.Y×" (severidad = ratio
    sobre umbral del tier). NO se reporta confianza heurística ni tipo.
  - Click en un día anómalo → bloque de explicabilidad limpio (T1 stats por
    columna, T2 top correlaciones, T3 canal + severidad por canal, T4 drivers).

Uso:
    from dashboard import display_dashboard
    display_dashboard(runner.results_multi)         # auto-pickea T4 si existe
    display_dashboard(runner.results_multi,
                      results_row_level=runner.results_row_level)
"""
from __future__ import annotations
import json
import uuid
from typing import Dict, List, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Configuración estática de tiers (label, short, color)
# ─────────────────────────────────────────────────────────────────────────────
_TIER_META = {
    "T1": {"label": "Stats column-level", "short": "T1 Stats",  "accent": "#2563eb"},
    "T2": {"label": "PCA cross-column",   "short": "T2 PCA",    "accent": "#7c3aed"},
    "T3": {"label": "TranAD agregado",    "short": "T3 TranAD", "accent": "#0891b2"},
    "T4": {"label": "Row-level AE",       "short": "T4 RowAE",  "accent": "#db2777"},
}

_STAT_ES = {
    "pct_null": "% nulos", "pct_empty": "% vacíos", "pct_unknown": "% desconocidos",
    "entropy_norm": "diversidad", "hhi": "concentración",
    "top1_share": "valor dominante", "n_cats": "nº categorías",
}

_CHANNEL_ES = {
    "Quality": "Calidad", "Volume": "Volumen", "Structural": "Estructura",
}


# ─────────────────────────────────────────────────────────────────────────────
# Extractores de explicabilidad por tier
# ─────────────────────────────────────────────────────────────────────────────
def _extract_t1_explanations(t1: Optional[Dict]) -> List[Dict]:
    """Top columnas de T1 con stat humanizado y z-score."""
    if not t1:
        return []
    details = t1.get("details_df")
    if not isinstance(details, pd.DataFrame) or details.empty:
        return []
    if "flagged" not in details.columns:
        return []
    flagged = details[details["flagged"]].copy()
    if flagged.empty:
        return []
    flagged["abs_z"] = flagged["z_score"].abs() if "z_score" in flagged.columns else 0.0
    out = []
    for _, r in flagged.nlargest(8, "abs_z").iterrows():
        col      = str(r.get("column", ""))
        stat     = str(r.get("stat", ""))
        val      = r.get("value", 0)
        expected = r.get("expected", 0)
        z        = float(r.get("z_score", 0))
        out.append({
            "column":      col,
            "stat":        stat,
            "z":           round(z, 1),
            "description": _humanize_t1_stat(col, stat, val, expected, z),
        })
    return out


def _humanize_t1_stat(col: str, stat: str, val, expected, z: float) -> str:
    import math
    try:
        v = float(val)
    except (TypeError, ValueError):
        v = float("nan")
    try:
        e = float(expected)
    except (TypeError, ValueError):
        e = float("nan")
    v_nan = math.isnan(v)
    e_nan = math.isnan(e)

    # Si ambos lados son NaN no podemos decir nada concreto.
    if v_nan and e_nan:
        return f"{col} — {_STAT_ES.get(stat, stat)} anómalo (sin valor comparable, z={z:+.1f})"

    direction = "↑" if z > 0 else "↓"

    if stat in ("pct_null", "pct_empty", "pct_unknown"):
        if v_nan or e_nan:
            return f"{col} — {_STAT_ES.get(stat, stat)} anómalo (z={z:+.1f})"
        return (f"{col} — {_STAT_ES.get(stat, stat)}: {v*100:.1f}% vs "
                f"{e*100:.1f}% esperado  {direction} (z={z:+.1f})")

    if stat == "n_cats":
        if v_nan or e_nan:
            return f"{col} — nº categorías anómalo (z={z:+.1f})"
        return f"{col} — {int(v)} categorías vs {int(e)} esperadas  {direction} (z={z:+.1f})"

    if stat == "top1_share":
        if v_nan or e_nan:
            return f"{col} — valor dominante anómalo (z={z:+.1f})"
        return (f"{col} — valor dominante {v*100:.0f}% vs {e*100:.0f}%  "
                f"{direction} (z={z:+.1f})")

    if stat in ("entropy_norm", "hhi"):
        if v_nan or e_nan:
            return f"{col} — distribución anómala (z={z:+.1f})"
        verb = "más concentrada" if (stat == "hhi" and v > e) or \
                                    (stat == "entropy_norm" and v < e) \
               else "menos concentrada"
        return f"{col} — distribución {verb} de lo normal (z={z:+.1f})"

    # Fallback genérico para stats no contemplados.
    v_str = "NaN" if v_nan else f"{v:.3f}"
    e_str = "NaN" if e_nan else f"{e:.3f}"
    return f"{col}.{stat}: {v_str} vs {e_str} (z={z:+.1f})"


def _extract_t2_explanations(t2: Optional[Dict]) -> List[Dict]:
    """Top columnas afectadas por la rotura de correlación T2."""
    if not t2:
        return []
    top = t2.get("top_columns")
    if not isinstance(top, pd.DataFrame) or top.empty:
        return []
    out = []
    col_field = "column" if "column" in top.columns else top.columns[0]
    pct_field = next((c for c in ("pct_of_total", "recon_pct", "weight") if c in top.columns), None)
    for _, r in top.head(5).iterrows():
        col = str(r[col_field])
        if pct_field:
            out.append({"column": col,
                        "description": f"{col} — {float(r[pct_field]):.0f}% del error de reconstrucción"})
        else:
            out.append({"column": col, "description": col})
    return out


def _extract_t4_explanations(t4_tier_dict: Optional[Dict]) -> List[Dict]:
    """Top drivers row-level. Si no hay drivers detallados, usa top_col."""
    if not t4_tier_dict:
        return []
    drivers = t4_tier_dict.get("top_drivers") or []
    out: List[Dict] = []
    if drivers:
        for d in drivers[:5]:
            col   = str(d.get("col") or d.get("column", ""))
            ratio = float(d.get("ratio_vs_normal", 0) or 0)
            pct   = float(d.get("pct_anomalous_rows", 0) or 0)
            desc  = f"{col} — {ratio:.1f}× vs normal"
            if pct > 0:
                desc += f" ({pct*100:.0f}% filas afectadas)"
            out.append({"column": col, "description": desc})
    elif t4_tier_dict.get("top_col"):
        col = t4_tier_dict["top_col"]
        out.append({"column": col, "description": f"{col} — columna con más error"})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del JSON que consume el JS embebido
# ─────────────────────────────────────────────────────────────────────────────
def _build_data_json(period_df: pd.DataFrame,
                     combined_reports: List[Dict],
                     row_level_summary: Optional[pd.DataFrame],
                     row_level_threshold: Optional[float],
                     table: str,
                     tiers_run: Optional[List[int]] = None,
                     row_level_reports: Optional[list] = None) -> str:
    # ── Detectar qué tiers están presentes en este period_df ─────────────────
    tiers_present: List[str] = []
    if tiers_run is not None:
        for t in sorted(set(tiers_run)):
            name = f"T{t}"
            if name in _TIER_META:
                tiers_present.append(name)
    else:
        for name in ("T1", "T2", "T3", "T4"):
            col = f"{name.lower()}_fires"
            if col in period_df.columns:
                tiers_present.append(name)

    # T4: si tenemos row_level_summary y aún no aparece en tiers_present, lo añadimos
    if "T4" not in tiers_present and row_level_summary is not None and len(row_level_summary):
        tiers_present.append("T4")

    # Lookup de reports por fecha
    report_lookup = {pd.Timestamp(r["date"]): r for r in (combined_reports or [])}

    # Lookup de top drivers row-level (lista [{col, ratio_vs_normal}, ...] por día)
    rl_drivers: Dict[pd.Timestamp, List[Dict]] = {}
    if row_level_reports:
        for r in row_level_reports:
            try:
                d = pd.Timestamp(r.date)
                rl_drivers[d] = list(getattr(r, "top_columns", []) or [])
            except Exception:
                continue

    # Lookup row-level. El threshold que aplica al day-level (pct_anomalous) se
    # CALIBRA DINÁMICAMENTE sobre la ventana — el `row_level_threshold` que llega
    # desde row_level() es el de FILA (sobre reconstruction error), no day-level.
    rl_lookup: Dict[pd.Timestamp, Dict] = {}
    if row_level_summary is not None and len(row_level_summary):
        from core.alert_fusion import compute_t4_day_threshold
        if row_level_threshold is not None and row_level_threshold > 0.01:
            # Aceptamos un threshold explícito solo si es razonable (>1% de filas);
            # cualquier valor menor probablemente sea el row-level confundido.
            thr_rl_day = float(row_level_threshold)
        else:
            thr_rl_day = compute_t4_day_threshold(row_level_summary)
        for _, r in row_level_summary.iterrows():
            rl_lookup[pd.Timestamp(r["date"])] = {
                "pct_anomalous": float(r.get("pct_anomalous", 0)),
                "threshold_pct": thr_rl_day,
                "top_col":       str(r.get("top_col", "")),
                "n_total":       int(r.get("n_total", 0)) if "n_total" in r else 0,
                "n_anomalous":   int(r.get("n_anomalous", 0)) if "n_anomalous" in r else 0,
            }

    days = []
    for _, row in period_df.sort_values("date").iterrows():
        d  = pd.Timestamp(row["date"])
        dr = report_lookup.get(d)

        # Per-tier dict para este día
        tiers_day: Dict[str, Dict] = {}
        explanations: Dict[str, List[Dict]] = {}

        if "T1" in tiers_present:
            t1 = (dr.get("tier1") if dr else None) or {}
            tiers_day["T1"] = {
                "fires":    bool(row.get("t1_fires", False)),
                "severity": float(row.get("t1_severity", 0) or 0),
                "n_cols":   int(row.get("t1_n_flagged", 0)),
                "max_z":    float(row.get("t1_max_z", 0) or 0),
            }
            explanations["T1"] = _extract_t1_explanations(t1)

        if "T2" in tiers_present:
            t2 = (dr.get("tier2") if dr else None) or {}
            tiers_day["T2"] = {
                "fires":    bool(row.get("t2_fires", row.get("t2_anomaly", False))),
                "severity": float(row.get("t2_severity", 0) or 0),
                "z_score":  float(row.get("t2_z_score", t2.get("z_score", 0)) or 0),
            }
            explanations["T2"] = _extract_t2_explanations(t2)

        if "T3" in tiers_present:
            t3 = (dr.get("tier3") if dr else None) or {}
            channel = str(row.get("t3_channel", "") or t3.get("dominant_channel", ""))
            tiers_day["T3"] = {
                "fires":    bool(row.get("t3_fires", row.get("t3_anomaly", False))),
                "severity": float(row.get("t3_severity", 0) or 0),
                "channel":  channel,
                "channel_es": _CHANNEL_ES.get(channel.split("+")[0], channel),
                "sev_q": float(t3.get("score_quality", 0) / t3.get("threshold_quality", 1)
                               if t3.get("alert_quality") and t3.get("threshold_quality", 0) > 0 else 0),
                "sev_v": float(t3.get("score_volume", 0) / t3.get("threshold_volume", 1)
                               if t3.get("alert_volume") and t3.get("threshold_volume", 0) > 0 else 0),
                "sev_s": float(t3.get("score_structural", 0) / t3.get("threshold_structural", 1)
                               if t3.get("alert_structural") and t3.get("threshold_structural", 0) > 0 else 0),
            }
            explanations["T3"] = []  # canal ya en tiers_day["T3"]

        if "T4" in tiers_present:
            rl = rl_lookup.get(d)
            if rl is not None:
                thr = rl["threshold_pct"]
                pct = rl["pct_anomalous"]
                # Si el period_df ya trae el veredicto consolidado (vía
                # merge_tier4_into_period), respétalo — incluye low_volume guard.
                if "t4_fires" in row.index:
                    fires = bool(row.get("t4_fires", False))
                    sev   = float(row.get("t4_severity", 0) or 0)
                    low_v = bool(row.get("t4_low_volume", False))
                else:
                    fires = pct > thr
                    sev   = float((pct / thr) if (fires and thr > 0) else 0)
                    low_v = False
                tiers_day["T4"] = {
                    "fires":      fires,
                    "severity":   sev,
                    "pct":        pct,
                    "threshold":  thr,
                    "top_col":    rl["top_col"],
                    "low_volume": low_v,
                    "n_total":    rl.get("n_total", 0),
                    "n_anomalous": rl.get("n_anomalous", 0),
                    "top_drivers": rl_drivers.get(d, []),
                }
                explanations["T4"] = _extract_t4_explanations(tiers_day["T4"])
            else:
                tiers_day["T4"] = {"fires": False, "severity": 0.0, "pct": None,
                                   "threshold": None, "top_col": "",
                                   "low_volume": False, "n_total": 0}
                explanations["T4"] = []

        # Verdict global del día (recomputable, pero usamos lo de period_df)
        firing_now = [n for n, t in tiers_day.items() if t["fires"]]
        n_firing   = len(firing_now)
        max_sev    = max((t["severity"] for t in tiers_day.values() if t["fires"]), default=0.0)
        anomaly    = n_firing >= 1

        days.append({
            "date":         d.strftime("%Y-%m-%d"),
            "anomaly":      anomaly,
            "n_firing":     n_firing,
            "n_total":      len(tiers_day),
            "max_severity": round(max_sev, 2),
            "tiers":        tiers_day,
            "explanations": explanations,
        })

    return json.dumps({
        "table":           table,
        "tiers_present":   tiers_present,
        "tier_meta":       {k: _TIER_META[k] for k in tiers_present},
        "days":            days,
    }, default=str, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS
# ─────────────────────────────────────────────────────────────────────────────
_HTML = '''
<div id="dh-{uid}">
<style>
#dh-{uid} {{ all:initial; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif;
  color:#0f172a; background:#ffffff; padding:20px; border-radius:12px;
  max-width:920px; line-height:1.45; font-size:13px; box-sizing:border-box;
  display:block; border:1px solid #e2e8f0; }}
#dh-{uid} * {{ box-sizing:border-box; margin:0; padding:0; }}

#dh-{uid} .hdr {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
  padding:20px 24px; margin-bottom:14px; }}
#dh-{uid} .lbl {{ font-size:10px; letter-spacing:2px; color:#64748b; font-weight:700;
  text-transform:uppercase; }}
#dh-{uid} .ttl-row {{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
  margin-top:4px; margin-bottom:2px; }}
#dh-{uid} h1 {{ font-size:22px; font-weight:700; color:#0f172a; }}
#dh-{uid} .sub {{ font-size:12px; color:#64748b; font-variant-numeric:tabular-nums;
  margin-bottom:18px; }}

#dh-{uid} .bdg {{ padding:3px 12px; border-radius:14px; font-size:10px; font-weight:700;
  letter-spacing:1px; text-transform:uppercase; }}
#dh-{uid} .bok {{ background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }}
#dh-{uid} .bwn {{ background:#fef3c7; color:#92400e; border:1px solid #fde68a; }}
#dh-{uid} .bal {{ background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }}

#dh-{uid} .kpis {{ display:flex; gap:36px; margin-top:6px; padding:14px 0; flex-wrap:wrap;
  border-top:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0; }}
#dh-{uid} .kpi-v {{ font-size:24px; font-weight:700; color:#0f172a;
  font-variant-numeric:tabular-nums; letter-spacing:-0.4px; }}
#dh-{uid} .kpi-l {{ font-size:9px; letter-spacing:1px; color:#64748b; font-weight:600;
  text-transform:uppercase; margin-top:2px; }}

#dh-{uid} .tstrip {{ margin-top:14px; }}
#dh-{uid} .tstrip-lbl {{ font-size:10px; color:#64748b; font-weight:600; margin-bottom:4px;
  letter-spacing:.5px; }}
#dh-{uid} .tstrip-bars {{ display:flex; gap:2px; height:34px; align-items:flex-end; }}
#dh-{uid} .tstrip-cell {{ flex:1; min-width:0; text-align:center; }}
#dh-{uid} .tstrip-bar {{ width:100%; border-radius:2px; transition:background .2s; }}
#dh-{uid} .tstrip-tic {{ font-size:8px; color:#94a3b8; margin-top:3px; }}

#dh-{uid} .section-hdr {{ display:flex; justify-content:space-between; align-items:center;
  margin-bottom:8px; padding:0 4px; }}
#dh-{uid} .tab-toggle {{ font-size:11px; color:#2563eb; cursor:pointer; font-weight:600;
  padding:3px 10px; border:1px solid #dbeafe; border-radius:5px; background:#eff6ff; }}
#dh-{uid} .tab-toggle:hover {{ background:#dbeafe; }}

#dh-{uid} .day-card {{ border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px;
  margin-bottom:5px; background:#ffffff; transition:all .15s; }}
#dh-{uid} .day-card:hover {{ border-color:#cbd5e1; }}
#dh-{uid} .day-card.anom .day-row {{ cursor:pointer; user-select:none; }}
#dh-{uid} .day-card.ok {{ padding:6px 14px; background:#fafbfc; }}
#dh-{uid} .expansion {{ user-select:text; }}

#dh-{uid} .day-row {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
#dh-{uid} .day-date {{ font-weight:600; color:#0f172a; font-variant-numeric:tabular-nums;
  font-size:13px; min-width:88px; }}
#dh-{uid} .day-dow {{ color:#64748b; font-size:11px; min-width:30px; }}
#dh-{uid} .day-status {{ margin-left:auto; font-size:11px; color:#64748b;
  font-variant-numeric:tabular-nums; }}
#dh-{uid} .day-chevron {{ color:#94a3b8; font-size:10px; transition:transform .15s;
  margin-left:4px; }}

#dh-{uid} .pills {{ display:flex; gap:6px; flex-wrap:wrap; }}
#dh-{uid} .pill {{ display:inline-flex; align-items:center; gap:5px; padding:3px 9px;
  border-radius:14px; font-size:11px; font-weight:600; border:1px solid;
  font-variant-numeric:tabular-nums; line-height:1.2; }}
#dh-{uid} .pill .pill-name {{ font-weight:700; letter-spacing:.3px; }}
#dh-{uid} .pill .pill-sev {{ font-weight:600; }}
#dh-{uid} .pill-off  {{ background:#f1f5f9; color:#94a3b8; border-color:#e2e8f0; }}
#dh-{uid} .pill-low  {{ background:#fef3c7; color:#92400e; border-color:#fde68a; }}
#dh-{uid} .pill-mid  {{ background:#ffedd5; color:#9a3412; border-color:#fed7aa; }}
#dh-{uid} .pill-high {{ background:#fee2e2; color:#991b1b; border-color:#fecaca; }}

#dh-{uid} .expansion {{ margin-top:10px; padding-top:10px; border-top:1px solid #e2e8f0;
  display:none; }}
#dh-{uid} .day-card.open .expansion {{ display:block; }}
#dh-{uid} .day-card.open .day-chevron {{ transform:rotate(180deg); }}

#dh-{uid} .tier-block {{ margin-bottom:8px; padding:8px 10px; border-radius:5px;
  background:#f8fafc; border-left:3px solid; }}
#dh-{uid} .tier-block-t1 {{ border-left-color:#2563eb; }}
#dh-{uid} .tier-block-t2 {{ border-left-color:#7c3aed; }}
#dh-{uid} .tier-block-t3 {{ border-left-color:#0891b2; }}
#dh-{uid} .tier-block-t4 {{ border-left-color:#db2777; }}
#dh-{uid} .tier-block-hdr {{ font-size:11px; font-weight:700; color:#0f172a;
  display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }}
#dh-{uid} .tier-block-meta {{ font-size:10px; color:#64748b;
  font-variant-numeric:tabular-nums; font-weight:500; }}
#dh-{uid} .tier-block-row {{ font-size:11.5px; color:#334155; padding:2px 0;
  font-variant-numeric:tabular-nums; }}
#dh-{uid} .tier-block-row .col {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  color:#0f172a; }}

#dh-{uid} .legend {{ font-size:10px; color:#64748b; margin-top:10px;
  padding:8px 12px; background:#f8fafc; border:1px solid #e2e8f0;
  border-radius:6px; line-height:1.7; }}
#dh-{uid} .legend b {{ color:#334155; }}
#dh-{uid} .swatch {{ display:inline-block; width:9px; height:9px; border-radius:50%;
  vertical-align:middle; margin-right:3px; }}
</style>

<div class="hdr">
  <div class="lbl">DATA HEALTH MONITOR</div>
  <div class="ttl-row">
    <h1 id="t-{uid}"></h1>
    <span id="badge-{uid}" class="bdg"></span>
  </div>
  <div class="sub" id="sub-{uid}"></div>
  <div class="kpis" id="kpis-{uid}"></div>
  <div class="tstrip">
    <div class="tstrip-lbl" id="tstrip-lbl-{uid}">SEVERIDAD MÁXIMA POR DÍA</div>
    <div class="tstrip-bars" id="tstrip-{uid}"></div>
  </div>
</div>

<div class="section-hdr">
  <div class="lbl" id="section-lbl-{uid}">DÍAS ANÓMALOS</div>
  <span class="tab-toggle" id="tt-{uid}">Ver todos los días</span>
</div>

<div id="daylist-{uid}"></div>

<div class="legend">
  <div><b>SEVERIDAD</b> = veces sobre el umbral del tier que dispara.
       <span class="swatch" style="background:#fbbf24"></span>1–2× leve
   &middot; <span class="swatch" style="background:#f97316"></span>2–5× media
   &middot; <span class="swatch" style="background:#dc2626"></span>5–10× alta
   &middot; <span class="swatch" style="background:#991b1b"></span>&gt;10× extrema</div>
  <div id="tier-legend-{uid}" style="margin-top:3px"></div>
  <div style="margin-top:3px"><b>VEREDICTO</b> = <b>Atención</b> si ≥1 tier supera su umbral &middot; <b>Alerta</b> si ≥3 tiers convergen (consenso).</div>
</div>
</div>

<script>
(function(){{
const D  = {data_json};
const u  = "{uid}";
const $  = s => document.querySelector("#dh-"+u+" "+s);
const $$ = s => document.querySelectorAll("#dh-"+u+" "+s);

const days = D.days;
const N    = days.length;
const A    = days.filter(d => d.anomaly).length;
const CONS3 = days.filter(d => d.n_firing >= 3).length;   // consenso ≥3 tiers
const CONS2 = days.filter(d => d.n_firing >= 2).length;   // consenso ≥2 tiers (referencia)
const MAX_SEV = days.reduce((m, d) => Math.max(m, d.max_severity), 0);

const DOW = ["Dom","Lun","Mar","Mié","Jue","Vie","Sáb"];
const MON = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"];
const parseDay = s => new Date(s + "T12:00:00");
const dowOf    = s => DOW[parseDay(s).getDay()];
const dayOf    = s => parseDay(s).getDate();
const monOf    = s => MON[parseDay(s).getMonth()];

// ── Header ────────────────────────────────────────────────────────────────
$("#t-"+u).textContent = D.table;
const sub = days.length
    ? days[0].date + " → " + days[N-1].date + " · " + N + " días · tiers: "
      + D.tiers_present.join(" · ")
    : "(sin datos)";
$("#sub-"+u).textContent = sub;

const badge = $("#badge-"+u);
if (A === 0) {{
  badge.className = "bdg bok"; badge.textContent = "Saludable";
}} else if (CONS3 >= 1) {{
  badge.className = "bdg bal"; badge.textContent = "Alerta";
}} else {{
  badge.className = "bdg bwn"; badge.textContent = "Atención";
}}

// ── KPIs ──────────────────────────────────────────────────────────────────
function kpi(v, l) {{
  return '<div><div class="kpi-v">'+v+'</div><div class="kpi-l">'+l+'</div></div>';
}}
$("#kpis-"+u).innerHTML =
    kpi(N, "Días evaluados")
  + kpi(A, "Anomalías")
  + kpi(CONS3, "Consenso ≥3")
  + kpi(MAX_SEV > 0 ? MAX_SEV.toFixed(1) + "×" : "—", "Severidad máx.");

// ── Color y altura por severidad (5 niveles discretos) ──────────────────
function pillClass(sev, fires) {{
  if (!fires) return "pill pill-off";
  if (sev < 2)  return "pill pill-low";
  if (sev < 5)  return "pill pill-mid";
  return "pill pill-high";
}}
// Tira diaria: 5 estados discretos (0% / 25% / 50% / 75% / 100%)
function stripLevel(sev, fires) {{
  if (!fires)      return {{h: 0.00, color: "#e2e8f0"}};   // estado 0
  if (sev < 2)     return {{h: 0.25, color: "#fbbf24"}};   // estado 1 — leve (amber)
  if (sev < 5)     return {{h: 0.50, color: "#f97316"}};   // estado 2 — medio (orange)
  if (sev < 10)    return {{h: 0.75, color: "#dc2626"}};   // estado 3 — alto (red)
  return                    {{h: 1.00, color: "#991b1b"}};   // estado 4 — extremo (dark red)
}}

// ── Tira diaria (severity strip, niveles cuantizados) ────────────────────
const tstrip = $("#tstrip-"+u);
let stripHtml = "";
days.forEach((d, i) => {{
  const sev    = d.max_severity || 0;
  const fires  = d.anomaly;
  const st     = stripLevel(sev, fires);
  // Altura: piso de 4px (visible aunque no haya disparo) + escala cuantizada
  const h      = Math.max(4, Math.round(st.h * 32));
  const tic    = (i === 0 || dayOf(d.date) === 1)
        ? monOf(d.date) + " " + dayOf(d.date)
        : dayOf(d.date);
  const title  = d.date + " — sev " + sev.toFixed(1) + "×, "
              + d.n_firing + "/" + d.n_total + " tiers";
  stripHtml += '<div class="tstrip-cell" title="'+title+'">'
            +   '<div class="tstrip-bar" style="height:'+h+'px;background:'+st.color+'"></div>'
            +   '<div class="tstrip-tic">'+tic+'</div>'
            + '</div>';
}});
tstrip.innerHTML = stripHtml;

// ── Leyenda dinámica de tiers ────────────────────────────────────────────
const tlEl = $("#tier-legend-"+u);
const tierMeta = D.tier_meta || {{}};
tlEl.innerHTML = '<b>TIERS EVALUADOS</b> · '
    + D.tiers_present.map(t => {{
        const m = tierMeta[t] || {{label: t, accent: "#64748b"}};
        return '<span style="color:'+m.accent+';font-weight:600">'+t+'</span> '+m.label;
      }}).join(" &middot; ");

// ── Lista de días ────────────────────────────────────────────────────────
let showAll = false;
let opened  = null;
const list  = $("#daylist-"+u);
const tt    = $("#tt-"+u);
const sl    = $("#section-lbl-"+u);

function pillHtml(t, info) {{
  const cls   = pillClass(info.severity, info.fires);
  const meta  = tierMeta[t] || {{short: t}};
  const sev   = info.fires ? info.severity.toFixed(1) + "×" : "—";
  let title   = t + " " + (meta.label || "");
  if (t === "T4") {{
    if (info.pct != null) title += " · " + (info.pct*100).toFixed(2) + "% filas";
    if (info.low_volume)  title += " · ⚠ volumen bajo, no dispara";
  }}
  if (info.fires) {{
    if (t === "T1") title += " · " + info.n_cols + " cols, max z=" + info.max_z.toFixed(1);
    if (t === "T2") title += " · z=" + info.z_score.toFixed(1);
    if (t === "T3") title += " · canal " + (info.channel || "?");
  }}
  return '<span class="'+cls+'" title="'+title.replace(/"/g, '&quot;')+'">'
      +    '<span class="pill-name">'+(meta.short || t)+'</span>'
      +    '<span class="pill-sev">'+sev+'</span>'
      +  '</span>';
}}

function expansionHtml(d) {{
  let h = '<div class="expansion">';

  // ── T1 ──
  const t1 = d.tiers.T1;
  if (t1 && t1.fires) {{
    h += '<div class="tier-block tier-block-t1">';
    h +=   '<div class="tier-block-hdr"><span>T1 Stats column-level</span>'
        +  '<span class="tier-block-meta">'+t1.n_cols+' cols flagged · max z='+t1.max_z.toFixed(1)
        +    ' · '+t1.severity.toFixed(1)+'×</span></div>';
    const expl = d.explanations.T1 || [];
    expl.slice(0, 6).forEach(e => {{
      h += '<div class="tier-block-row">→ '+escapeHtml(e.description)+'</div>';
    }});
    if (expl.length === 0) {{
      h += '<div class="tier-block-row">(sin desglose por columna)</div>';
    }}
    h += '</div>';
  }}

  // ── T2 ──
  const t2 = d.tiers.T2;
  if (t2 && t2.fires) {{
    h += '<div class="tier-block tier-block-t2">';
    h +=   '<div class="tier-block-hdr"><span>T2 PCA cross-column</span>'
        +  '<span class="tier-block-meta">z='+t2.z_score.toFixed(1)
        +    ' · '+t2.severity.toFixed(1)+'×</span></div>';
    const expl = d.explanations.T2 || [];
    if (expl.length === 0) {{
      h += '<div class="tier-block-row">Patrón global de correlaciones alterado.</div>';
    }} else {{
      h += '<div class="tier-block-row" style="color:#64748b;font-size:10px;margin-bottom:2px">'
        +  'Columnas con más contribución al error de reconstrucción PCA '
        +  '(su patrón conjunto con el resto se ha roto):</div>';
      expl.forEach(e => {{
        h += '<div class="tier-block-row">→ <span class="col">'+escapeHtml(e.column)+'</span> '
          +  escapeHtml(e.description.replace(e.column, "").trim().replace(/^—\\s*/, ""))+'</div>';
      }});
    }}
    h += '</div>';
  }}

  // ── T3 ──
  const t3 = d.tiers.T3;
  if (t3 && t3.fires) {{
    h += '<div class="tier-block tier-block-t3">';
    h +=   '<div class="tier-block-hdr"><span>T3 TranAD agregado</span>'
        +  '<span class="tier-block-meta">'+t3.severity.toFixed(1)+'× · canal '
        +    escapeHtml(t3.channel || "?")+'</span></div>';
    const CH_DESC = {{
      "Quality":    "calidad de datos (% nulos, vacíos, unknown por columna)",
      "Volume":     "volumen y cardinalidades (filas, valores únicos)",
      "Structural": "estructura de distribución (entropía, concentración, top-1, nº categorías)"
    }};
    const chanKey = (t3.channel || "").split("+")[0];
    if (CH_DESC[chanKey]) {{
      h += '<div class="tier-block-row" style="color:#64748b;font-size:10px;margin-bottom:2px">'
        +  'Canal '+escapeHtml(chanKey)+' = '+escapeHtml(CH_DESC[chanKey])+'.</div>';
    }}
    if (t3.sev_q > 0)
      h += '<div class="tier-block-row">→ Calidad: '+t3.sev_q.toFixed(1)+'× sobre umbral</div>';
    if (t3.sev_v > 0)
      h += '<div class="tier-block-row">→ Volumen: '+t3.sev_v.toFixed(1)+'× sobre umbral</div>';
    if (t3.sev_s > 0)
      h += '<div class="tier-block-row">→ Estructura: '+t3.sev_s.toFixed(1)+'× sobre umbral</div>';
    h += '</div>';
  }}

  // ── T4 ──
  const t4 = d.tiers.T4;
  if (t4 && t4.fires) {{
    h += '<div class="tier-block tier-block-t4">';
    const nA = t4.n_anomalous || 0;
    const nT = t4.n_total || 0;
    const counts = nT > 0 ? nA.toLocaleString()+' de '+nT.toLocaleString()+' filas' : '';
    h +=   '<div class="tier-block-hdr"><span>T4 Row-level AE</span>'
        +  '<span class="tier-block-meta">'+(t4.pct*100).toFixed(1)+'% '+counts
        +    ' · umbral '+(t4.threshold*100).toFixed(1)+'% · '+t4.severity.toFixed(1)+'×</span></div>';
    const drivers = t4.top_drivers || [];
    if (drivers.length === 0 && t4.top_col) {{
      h += '<div class="tier-block-row">→ <span class="col">'+escapeHtml(t4.top_col)+'</span> — columna con más error medio</div>';
    }} else if (drivers.length > 0) {{
      h += '<div class="tier-block-row" style="color:#64748b;font-size:10px;margin-bottom:2px">Top columnas por error relativo (ratio vs. baseline):</div>';
      drivers.slice(0, 5).forEach(dr => {{
        const col   = String(dr.col || dr.column || "");
        const ratio = Number(dr.ratio_vs_normal || 0);
        const tag   = ratio > 5 ? "🔴" : ratio > 2 ? "🟠" : "🟡";
        h += '<div class="tier-block-row">→ '+tag+' <span class="col">'+escapeHtml(col)
          +  '</span> — '+ratio.toFixed(1)+'× sobre lo normal</div>';
      }});
    }} else {{
      h += '<div class="tier-block-row">(sin desglose por columna disponible)</div>';
    }}
    h += '</div>';
  }}

  h += '</div>';
  return h;
}}

function escapeHtml(s) {{
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, m => ({{
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }})[m]);
}}

function render() {{
  const items = showAll ? days : days.filter(d => d.anomaly);
  sl.textContent = showAll ? "TODOS LOS DÍAS" : "DÍAS ANÓMALOS";
  tt.textContent = showAll ? "Solo anomalías" : "Ver todos los días";

  if (items.length === 0) {{
    list.innerHTML = '<div class="day-card ok"><div class="day-row">'
      + '<span style="color:#16a34a">✓</span>'
      + '<span style="color:#64748b">Sin anomalías en el periodo.</span></div></div>';
    return;
  }}

  let html = "";
  items.forEach((d, idx) => {{
    const isAnom = d.anomaly;
    const dayIdx = days.indexOf(d);  // estable para mantener "opened" al toggle
    const isOpen = opened === dayIdx;
    const klass  = "day-card " + (isAnom ? "anom" : "ok") + (isOpen ? " open" : "");

    let pillsHtml = '<div class="pills">';
    D.tiers_present.forEach(t => {{
      const info = d.tiers[t];
      if (info) pillsHtml += pillHtml(t, info);
    }});
    pillsHtml += '</div>';

    const status = isAnom
        ? ('<span class="day-status">'+d.n_firing+'/'+d.n_total+' tiers · sev '
           + d.max_severity.toFixed(1) + '×</span>'
           + '<span class="day-chevron">▾</span>')
        : '<span class="day-status">✓ normal</span>';

    html += '<div class="'+klass+'" data-idx="'+dayIdx+'">'
         +    '<div class="day-row">'
         +      '<span class="day-date">'+d.date+'</span>'
         +      '<span class="day-dow">'+dowOf(d.date)+'</span>'
         +      pillsHtml
         +      status
         +    '</div>'
         +    (isAnom && isOpen ? expansionHtml(d) : "")
         +  '</div>';
  }});
  list.innerHTML = html;

  // Hook clicks SOLO sobre el header del día (no la expansión, para que se
  // pueda seleccionar/copiar texto dentro sin que se cierre la card).
  $$(".day-card.anom .day-row").forEach(row => {{
    row.addEventListener("click", function() {{
      const card = this.closest(".day-card");
      const i = parseInt(card.getAttribute("data-idx"));
      opened = (opened === i) ? null : i;
      render();
    }});
  }});
  // Detener cualquier evento dentro de la expansión para que clicks y
  // selecciones de texto NO toggleen la card padre.
  $$(".expansion").forEach(ex => {{
    ["click", "mousedown", "mouseup", "contextmenu"].forEach(evt => {{
      ex.addEventListener(evt, function(e) {{ e.stopPropagation(); }});
    }});
  }});
}}

tt.addEventListener("click", function() {{ showAll = !showAll; opened = null; render(); }});
render();
}})();
</script>
'''


def display_dashboard(results_multi: dict,
                      table: Optional[str] = None,
                      results_row_level: Optional[dict] = None):
    """Renderiza el dashboard HTML inline en Jupyter.

    Args:
        results_multi: salida de runner.monitor() o run_multi_tier_mode().
                       Debe contener `period_df` y opcionalmente
                       `combined_reports` (lista de dicts) o `reports`,
                       `_tiers_run` (lista [1,2,3,4]) y `row_level`.
        table:         nombre para mostrar en el header. Si None, usa
                       results_multi["_display_name"] o "portabilidades".
        results_row_level: salida de runner.row_level(). Si no se pasa pero
                       results_multi["row_level"] existe, se usa ese.
    """
    from IPython.display import HTML, display as ipy_display

    period_df = results_multi.get("period_df")
    if period_df is None or len(period_df) == 0:
        ipy_display(HTML("<div style='padding:20px;border:1px solid #e2e8f0;border-radius:6px;"
                         "color:#64748b'>Sin datos para mostrar.</div>"))
        return

    combined = (results_multi.get("combined_reports")
                or results_multi.get("reports") or [])

    if table is None:
        table = results_multi.get("_display_name", "portabilidades")

    if results_row_level is None:
        results_row_level = results_multi.get("row_level")

    rl_summary = None
    rl_threshold = None
    rl_reports = None
    if results_row_level is not None:
        rl_summary   = results_row_level.get("summary_df")
        rl_threshold = results_row_level.get("_threshold")
        rl_reports   = results_row_level.get("reports")

    tiers_run = results_multi.get("_tiers_run")

    uid = uuid.uuid4().hex[:8]
    data_json = _build_data_json(
        period_df=period_df,
        combined_reports=combined,
        row_level_summary=rl_summary,
        row_level_threshold=rl_threshold,
        table=table,
        tiers_run=tiers_run,
        row_level_reports=rl_reports,
    )
    ipy_display(HTML(_HTML.format(uid=uid, data_json=data_json)))
