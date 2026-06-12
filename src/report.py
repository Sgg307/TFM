"""
report.py — Presentación de resultados del sistema de Data Health
==================================================================
Dos modos:
  format_executive()   → Salida limpia para presentar en la empresa
  format_developer()   → Salida verbosa para debugging
"""

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd

from core.labels import (
    CHANNEL_ES, STAT_ES, STAT_DIRECTION, DOW_ES, SEV,
    tier_tag, translate_channel, dow_label, fmt_pct,
)


_CONF_BAR_LEN = 10


def _conf_bar(conf: float) -> str:
    filled = round(conf * _CONF_BAR_LEN)
    return f"{'█' * filled}{'░' * (_CONF_BAR_LEN - filled)} {conf*100:.0f}%"


def _format_t1_column_detail(col_name: str, details_df: pd.DataFrame) -> List[str]:
    lines = []
    col_rows = details_df[(details_df["column"] == col_name) & (details_df["flagged"])]
    if col_rows.empty:
        return [f"    → {col_name}"]
    for _, r in col_rows.iterrows():
        stat, val, exp = r["stat"], r["value"], r["expected"]
        stat_label = STAT_ES.get(stat, stat)
        if pd.isna(val):
            lines.append(f"    → {col_name}: {stat_label} = NaN (esperado: {fmt_pct(exp)})")
        elif stat in ("pct_null", "pct_empty", "pct_unknown"):
            d = STAT_DIRECTION.get(stat, ("más", "menos"))
            direction = d[0] if val > exp else d[1]
            lines.append(f"    → {col_name}: {fmt_pct(val)} {stat_label} vs "
                         f"{fmt_pct(exp)} esperado ({direction} de lo normal)")
        elif stat == "entropy_norm":
            direction = STAT_DIRECTION[stat][0] if val > exp else STAT_DIRECTION[stat][1]
            lines.append(f"    → {col_name}: distribución {direction} de lo normal")
        elif stat == "hhi":
            direction = STAT_DIRECTION[stat][0] if val > exp else STAT_DIRECTION[stat][1]
            lines.append(f"    → {col_name}: {direction} de lo normal")
        elif stat == "top1_share":
            lines.append(f"    → {col_name}: valor más frecuente ocupa {fmt_pct(val)} "
                         f"vs {fmt_pct(exp)} esperado")
        elif stat == "n_cats":
            lines.append(f"    → {col_name}: {int(val)} categorías vs {int(exp)} esperadas")
        else:
            lines.append(f"    → {col_name}: {stat_label} inusual")
    return lines


def _format_t2_column_detail(top_features_df: pd.DataFrame) -> List[str]:
    """Explicaciones para columnas detectadas por PCA (T2).

    T2 top_features: feature=brand_donor__pct_null, column=brand_donor, stat=pct_null.
    Agrupa por columna y muestra qué stats están alterados.
    """
    lines = []
    if top_features_df.empty:
        return lines
    col_summary = {}
    for _, r in top_features_df.iterrows():
        col  = r.get("column", "")
        stat = r.get("stat", "")
        err  = r.get("recon_error", 0)
        pct  = r.get("pct_of_total", 0)
        if not col:
            continue
        if col not in col_summary:
            col_summary[col] = {"total_error": 0, "stats": [], "total_pct": 0}
        col_summary[col]["total_error"] += err
        col_summary[col]["total_pct"]   += pct
        if stat:
            col_summary[col]["stats"].append(STAT_ES.get(stat, stat))
    sorted_cols = sorted(col_summary.items(), key=lambda x: x[1]["total_error"], reverse=True)
    for col, info in sorted_cols[:6]:
        stats_str = ", ".join(dict.fromkeys(info["stats"]))
        pct_str   = f"{info['total_pct']:.0f}% del error"
        if stats_str:
            lines.append(f"    → {col}: {stats_str} alterados ({pct_str})")
        else:
            lines.append(f"    → {col}: patrón inusual ({pct_str})")
    return lines


def _describe_anomaly(row: dict, day_report: Optional[Dict] = None) -> List[str]:
    lines = []
    t3 = day_report.get("tier3") if day_report else None
    if t3 and t3.get("anomaly"):
        ch = t3.get("dominant_channel", "")
        if ch:
            lines.append(f"Problema detectado en: {translate_channel(ch)}")
    elif row.get("t3_anomaly"):
        ch = row.get("t3_channel", "")
        if ch:
            lines.append(f"Problema detectado en: {translate_channel(ch)}")

    t2 = (day_report.get("tier2") or {}) if day_report else {}
    if t2.get("anomaly") or row.get("t2_anomaly"):
        top_t2_cols = t2.get("top_columns")
        if isinstance(top_t2_cols, pd.DataFrame) and not top_t2_cols.empty:
            col_names = ", ".join(top_t2_cols.head(3)["column"])
            lines.append(f"Relación entre columnas alterada (más afectadas: {col_names})")
        else:
            lines.append("Relaciones entre columnas inusuales (patrón global alterado)")

    if day_report:
        col_lines = _extract_column_explanations(day_report)
        if col_lines:
            lines.append("Columnas afectadas:")
            lines.extend(col_lines)
    return lines


def _extract_column_explanations(day_report: Dict) -> List[str]:
    """Prioridad: T1 details_df > T2 top_features > T2 top_columns."""
    lines = []
    t1 = day_report.get("tier1") or {}
    top_t1 = t1.get("top_columns")
    if isinstance(top_t1, pd.DataFrame) and not top_t1.empty:
        details = t1.get("details_df", pd.DataFrame())
        flagged = top_t1[top_t1["n_flagged"] > 0] if "n_flagged" in top_t1.columns else top_t1
        for _, r in flagged.head(5).iterrows():
            col_name = r.get("column", "")
            if isinstance(details, pd.DataFrame) and not details.empty:
                lines.extend(_format_t1_column_detail(col_name, details))
            else:
                lines.append(f"    → {col_name}")
    if lines:
        return lines

    # Fallback: T2 top_features (column__stat con error PCA)
    t2 = day_report.get("tier2") or {}
    top_features = t2.get("top_features")
    if isinstance(top_features, pd.DataFrame) and not top_features.empty:
        lines = _format_t2_column_detail(top_features)
        if lines:
            return lines

    # Fallback: T2 top_columns (solo nombres)
    top_t2_cols = t2.get("top_columns")
    if isinstance(top_t2_cols, pd.DataFrame) and not top_t2_cols.empty:
        for _, r in top_t2_cols.head(5).iterrows():
            lines.append(f"    → {r['column']}")
    return lines


def _extract_top_columns(day_report: Dict) -> List[str]:
    cols = []
    t1 = day_report.get("tier1") or {}
    top_t1 = t1.get("top_columns")
    if isinstance(top_t1, pd.DataFrame) and not top_t1.empty:
        flagged = top_t1[top_t1["n_flagged"] > 0] if "n_flagged" in top_t1.columns else top_t1
        for _, r in flagged.head(5).iterrows():
            if r.get("column", ""):
                cols.append(r["column"])
    if not cols:
        t2 = day_report.get("tier2") or {}
        top_t2 = t2.get("top_columns")
        if isinstance(top_t2, pd.DataFrame) and not top_t2.empty:
            cols = list(top_t2.head(5)["column"])
    return cols


# ─────────────────────────────────────────────────────────────────────────────
def format_executive(period_df, day_reports=None, table="portabilidades",
                     show_normal_days=False):
    n_days = len(period_df)
    n_anom = int(period_df["anomaly"].sum())
    n_ok   = n_days - n_anom
    dates  = pd.to_datetime(period_df["date"])
    d_start, d_end = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")

    if n_anom == 0:
        health_icon, health_label = "✅", "SALUDABLE"
    elif n_anom <= 2:
        health_icon, health_label = "⚠️", "ATENCIÓN"
    else:
        health_icon, health_label = "🚨", "ALERTA"

    report_lookup = {pd.Timestamp(r.get("date")): r for r in (day_reports or [])}
    w = 62
    lines = ["", f"{'═'*(w+4)}",
             f"  {health_icon}  DATA HEALTH REPORT", f"  Tabla: {table}",
             f"  Periodo: {d_start} → {d_end} ({n_days} días)", f"{'═'*(w+4)}", ""]
    if n_anom == 0:
        lines.append(f"  ✅ {n_ok} días evaluados — sin anomalías detectadas.")
    else:
        lines.append(f"  ✅ {n_ok} días normales  ·  ⚠️ {n_anom} anomalías "
                     f"({100*n_anom/n_days:.0f}%)")
        n_conf = int((period_df.get("n_tiers", pd.Series(dtype=int)) >= 2).sum())
        n_t3   = int(period_df.get("t3_anomaly", pd.Series(dtype=bool)).sum())
        n_t2   = int(period_df.get("t2_anomaly", pd.Series(dtype=bool)).sum())
        parts = []
        if n_conf:                parts.append(f"{n_conf} confirmadas (múltiples detectores)")
        if n_t3 - n_conf > 0:     parts.append(f"{n_t3 - n_conf} solo modelo")
        if n_t2 - n_conf > 0:     parts.append(f"{n_t2 - n_conf} solo correlación")
        if parts:
            lines.append(f"  Detecciones: {' · '.join(parts)}")
    lines.append("")
    lines.append(f"  ─── Timeline {'─'*(w-12)}")

    tc = []
    for _, row in period_df.sort_values("date").iterrows():
        if row.get("anomaly"):
            c = row.get("confidence", 0)
            tc.append("🔴" if c >= 0.8 else "🟡" if c >= 0.4 else "🟠")
        else:
            tc.append("·")
    sd    = period_df.sort_values("date")
    first = pd.Timestamp(sd.iloc[0]["date"]).strftime("%m/%d")
    last  = pd.Timestamp(sd.iloc[-1]["date"]).strftime("%m/%d")
    lines.append(f"  {' '.join(tc)}")
    lines.append(f"  {first}{' '*max(0, len(' '.join(tc))-len(first)-len(last))}{last}")
    lines.append("")

    anomalies = period_df[period_df["anomaly"]].sort_values("date")
    if not anomalies.empty:
        lines.append(f"  ─── Anomalías detectadas {'─'*(w-24)}")
        lines.append("")
        for _, row in anomalies.iterrows():
            d     = pd.Timestamp(row["date"])
            conf  = row.get("confidence", 0)
            atype = row.get("type", "none")
            icon, label = SEV.get(atype, ("⚪", atype.upper()))
            lines.append(f"  {icon} {d.strftime('%Y-%m-%d')} ({dow_label(d)})")
            lines.append(f"     Confianza: {_conf_bar(conf)}")
            lines.append(f"     Tipo: {label}  │  Detectores: {tier_tag(row.get('tiers_firing',''))}")
            dr = report_lookup.get(d)
            for dl in _describe_anomaly(row.to_dict(), dr):
                lines.append(f"     {dl}")
            lines.append("")
    if show_normal_days:
        normal = period_df[~period_df["anomaly"]].sort_values("date")
        if not normal.empty:
            lines.append(f"  ─── Días normales {'─'*(w-17)}")
            ds = [pd.Timestamp(r["date"]).strftime("%m/%d") for _, r in normal.iterrows()]
            for i in range(0, len(ds), 10):
                lines.append(f"     {'  '.join(ds[i:i+10])}")
            lines.append("")
    lines += [
        f"  {'─'*w}",
        "  Confirmado = múltiples detectores coinciden en anomalía",
        "  Agregado = detectado por el modelo global",
        "  Relacional = correlaciones entre columnas rotas",
        "",
    ]
    return "\n".join(lines)


def format_developer(period_df, day_reports, table="portabilidades"):
    from core.alert_fusion import format_full_report, summarize_period
    return "\n".join(
        [format_full_report(r) for r in day_reports] + [summarize_period(period_df)]
    )


def format_day_detail(combined: Dict) -> str:
    d     = pd.Timestamp(combined["date"])
    conf  = combined.get("confidence", 0)
    atype = combined.get("type", "none")
    icon, label = SEV.get(atype, ("⚪", atype.upper()))
    w = 62
    lines = [
        f"┌{'─'*w}┐",
        f"│  {icon} {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})"
        f"{' '*max(0, w-20-len(d.strftime('%A')))}│",
        f"│  Confianza: {_conf_bar(conf)}{' '*max(0, w-14-len(_conf_bar(conf)))}│",
        f"│  Tipo: {label}{' '*max(0, w-9-len(label))}│",
        f"├{'─'*w}┤",
    ]
    tf = combined.get("tiers_firing", [])
    if isinstance(tf, str):
        tf = [t.strip() for t in tf.split(",") if t.strip()]
    if tf:
        s = f"  Detectado por: {tier_tag(tf)}"
        lines.append(f"│{s}{' '*max(0, w-len(s))}│")
    r3 = combined.get("tier3") or {}
    if r3:
        if r3.get("anomaly"):
            t = f"  Área afectada: {translate_channel(r3.get('dominant_channel', '?'))}"
            lines.append(f"│{t}{' '*max(0, w-len(t))}│")
        else:
            lines.append(f"│  Modelo global: sin anomalía{' '*(w-30)}│")
    r2 = combined.get("tier2") or {}
    if r2.get("anomaly"):
        tc = r2.get("top_columns")
        if isinstance(tc, pd.DataFrame) and not tc.empty:
            t = f"  Correlaciones alteradas: {', '.join(tc.head(3)['column'])}"
        else:
            t = "  Correlaciones entre columnas: alteradas"
        lines.append(f"│{t}{' '*max(0, w-len(t))}│")
    elif r2:
        t = "  Correlaciones entre columnas: normales"
        lines.append(f"│{t}{' '*max(0, w-len(t))}│")
    cl = _extract_column_explanations(combined)
    if cl:
        lines.append(f"├{'─'*w}┤")
        h = "  DIAGNÓSTICO — columnas afectadas:"
        lines.append(f"│{h}{' '*max(0, w-len(h))}│")
        for c in cl:
            if len(c) > w - 2:
                c = c[:w-5] + "..."
            lines.append(f"│{c}{' '*max(0, w-len(c))}│")
    lines.append(f"└{'─'*w}┘")
    return "\n".join(lines)


def format_period_table(period_df):
    lines = [
        "",
        f"  {'Fecha':<12} {'DOW':^5} {'':^3} {'Conf':^6} "
        f"{'Tipo':<14} {'Detectores':<20} {'Cols':>5}",
        f"  {'─'*12} {'─'*5} {'─'*3} {'─'*6} {'─'*14} {'─'*20} {'─'*5}",
    ]
    for _, row in period_df.sort_values("date").iterrows():
        d_ts    = pd.Timestamp(row["date"])
        is_anom = row.get("anomaly", False)
        if is_anom:
            icon, label = SEV.get(row.get("type", "none"), ("⚪", row.get("type", "")))
            cs = f"{row.get('confidence', 0)*100:.0f}%"
        else:
            icon, label, cs = "✅", "", ""
        lines.append(
            f"  {d_ts.strftime('%Y-%m-%d'):<12} {dow_label(d_ts):^5} {icon:^3} "
            f"{cs:^6} {label:<14} {tier_tag(row.get('tiers_firing', '')):<20} "
            f"{row.get('t1_n_flagged', 0):>5}"
        )
    lines.append("")
    return "\n".join(lines)


def export_for_dashboard(period_df, day_reports, table):
    rl = {pd.Timestamp(r["date"]): r for r in (day_reports or [])}
    rows = []
    for _, row in period_df.sort_values("date").iterrows():
        d  = pd.Timestamp(row["date"])
        dr = rl.get(d)
        rows.append({
            "date":       d.strftime("%Y-%m-%d"),
            "anomaly":    bool(row.get("anomaly", False)),
            "confidence": float(row.get("confidence", 0)),
            "type":       str(row.get("type", "none")),
            "tiers":      tier_tag(row.get("tiers_firing", "")),
            "t3_channel": translate_channel(str(row.get("t3_channel", ""))),
            "t1_cols":    int(row.get("t1_n_flagged", 0)),
            "topCols":    _extract_top_columns(dr) if dr else [],
        })
    return rows
