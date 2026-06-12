"""labels.py — Constantes de presentación compartidas.

Tradicionalmente duplicadas entre alert_fusion, report y dashboard.
Mantener un único punto de verdad evita drift de etiquetas cuando se añade
un tier nuevo o se traducen los canales del modelo.
"""

from __future__ import annotations


# ── Días de la semana ────────────────────────────────────────────────────────
DOW_ES = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}


# ── Canales TranAD (Tier 3) ──────────────────────────────────────────────────
CHANNEL_ES = {
    "Quality":                    "Calidad de datos (nulos, vacíos)",
    "Volume":                     "Volumen de registros",
    "Structural":                 "Estructura y distribución",
    "Quality+Volume":             "Calidad + Volumen",
    "Quality+Structural":         "Calidad + Estructura",
    "Volume+Structural":          "Volumen + Estructura",
    "Quality+Volume+Structural":  "Calidad + Volumen + Estructura",
}


# ── Stats column-level (Tier 1 + 2) ──────────────────────────────────────────
STAT_ES = {
    "pct_null":     "nulos",
    "pct_empty":    "vacíos",
    "pct_unknown":  "unknowns",
    "entropy_norm": "diversidad",
    "hhi":          "concentración",
    "top1_share":   "valor dominante",
    "n_cats":       "nº categorías",
}

STAT_DIRECTION = {
    "pct_null":     ("más", "menos"),
    "pct_empty":    ("más", "menos"),
    "pct_unknown":  ("más", "menos"),
    "entropy_norm": ("más diversa",      "menos diversa"),
    "hhi":          ("más concentrada",  "menos concentrada"),
    "top1_share":   ("más dominante",    "menos dominante"),
    "n_cats":       ("más categorías",   "menos categorías"),
}


# ── Etiquetas de tier en reportes (alert_fusion.combine_day → "tiers_firing") ─
TIER_DISPLAY = {
    "Tier1_Column": "Stats",
    "Tier2_PCA":    "PCA",
    "Tier3_TranAD": "TranAD",
    "Tier4_RowAE":  "RowAE",
}


# ── Severidad (icono, etiqueta) por tipo de anomalía ─────────────────────────
SEV = {
    "confirmed":    ("🔴", "CONFIRMADO"),
    "aggregate":    ("🟠", "AGREGADO"),
    "relational":   ("🟡", "RELACIONAL"),
    "concentrated": ("🟡", "CONCENTRADO"),
    "none":         ("⚪", "—"),
}


# ── Helpers comunes ──────────────────────────────────────────────────────────

def tier_tag(tiers_str_or_list) -> str:
    """Convierte una lista/CSV de identificadores de tier a una etiqueta legible.

    >>> tier_tag(["Tier2_PCA", "Tier3_TranAD"])
    'PCA + TranAD'
    >>> tier_tag("Tier1_Column,Tier2_PCA")
    'Stats + PCA'
    """
    if isinstance(tiers_str_or_list, list):
        items = tiers_str_or_list
    else:
        items = [t.strip() for t in str(tiers_str_or_list).split(",") if t.strip()]
    parts = [TIER_DISPLAY.get(t, t) for t in items]
    return " + ".join(parts) if parts else "—"


def translate_channel(ch: str) -> str:
    """Traducción del canal TranAD para presentación."""
    return CHANNEL_ES.get(ch, ch)


def dow_label(date) -> str:
    """Etiqueta corta del día de la semana para una fecha."""
    import pandas as pd
    return DOW_ES.get(pd.Timestamp(date).dayofweek, "?")


def fmt_pct(v: float) -> str:
    """Porcentaje con resolución adaptativa según magnitud."""
    if abs(v) < 0.01:
        return f"{v*100:.2f}%"
    return f"{v*100:.1f}%"
