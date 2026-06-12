"""
column_attribution.py — Atribución column-level (Nivel 2 de explicabilidad)
Nivel 1: feature_universal → %_contribución
Nivel 2: feature_universal → columnas_raw_responsables

MODIFICADO: integra ArrayStructAggregator para columnas ARRAY<STRUCT>. Si
cfg["array_struct_features"] está presente, esas columnas se procesan por el
agregador (que las convierte en pseudo-columnas con stats categóricas sobre
el flatten del día). El resto del archivo es idéntico.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
import numpy as np, pandas as pd

log = logging.getLogger(__name__)

_CAT_MAX_CARDINALITY = 500
_UNKNOWN_VALUES = {"UNKNOWN", "N/A", "NULL", "NONE", ""}
_BASELINE_DAYS_DEFAULT = 30
_STAT_COLS = ["pct_null", "pct_empty", "pct_unknown", "entropy_norm", "hhi", "top1_share", "n_cats", "n_rows"]


# ── Helpers de stats ──────────────────────────────────────────────────────────

def _entropy_norm(s: pd.Series) -> float:
    p = s.value_counts(normalize=True).values
    if len(p) <= 1: return 0.0
    raw = float(-np.sum(p * np.log(p + 1e-12)))
    return raw / float(np.log(len(p)))

def _hhi(s: pd.Series) -> float:
    if s.empty: return 1.0
    return float((s.value_counts(normalize=True).values ** 2).sum())

def _is_catlike(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s): return False
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s): return True
    if pd.api.types.is_string_dtype(s): return True
    return False


def _stats_for_one_column(col_values: pd.Series, n_rows: int) -> Dict[str, float]:
    if n_rows == 0: return {k: 0.0 for k in _STAT_COLS}
    pct_null = float(col_values.isna().mean())
    is_str = pd.api.types.is_string_dtype(col_values)
    pct_empty = float((col_values == "").mean()) if is_str else 0.0
    pct_unknown = float(col_values.dropna().astype(str).str.upper().isin(_UNKNOWN_VALUES).sum()) / n_rows if is_str else 0.0

    clean = col_values.dropna()
    is_cat = _is_catlike(col_values)
    if not is_cat and pd.api.types.is_numeric_dtype(col_values) and not pd.api.types.is_bool_dtype(col_values):
        if len(clean) > 0 and int(clean.nunique()) <= _CAT_MAX_CARDINALITY: is_cat = True

    entropy = hhi = top1 = n_cats = np.nan
    if is_cat and len(clean) > 0:
        nu = int(clean.nunique())
        if 0 < nu <= _CAT_MAX_CARDINALITY:
            vc = clean.value_counts(normalize=True)
            entropy, hhi, top1, n_cats = _entropy_norm(clean), _hhi(clean), float(vc.iloc[0]), float(nu)

    return {"pct_null": pct_null, "pct_empty": pct_empty, "pct_unknown": pct_unknown,
            "entropy_norm": entropy, "hhi": hhi, "top1_share": top1, "n_cats": n_cats, "n_rows": float(n_rows)}


# ── Persistencia / cómputo ────────────────────────────────────────────────────

def _per_column_stats_path(cfg: dict) -> Optional[Path]:
    raw = (cfg.get("paths") or {}).get("raw_data")
    return Path(raw).parent / "per_column_stats.parquet" if raw else None

def _resolve_date_col(cfg: dict) -> str:
    return cfg["bq"].get("date_col") or cfg["bq"].get("snapshot_col", "snapshot_date")


def _maybe_load_array_aggregator(df: pd.DataFrame, cfg: dict):
    """Devuelve (aggregator, array_cols_a_excluir_del_loop_normal).

    Si cfg["array_struct_features"] está presente:
      - intenta cargar aggregator persistido en cfg["paths"]["array_struct_aggregator"]
      - si no existe, fit on-the-fly (las day-stats no dependen del vocab fitted)
    Si la clave no existe, devuelve (None, []).
    """
    if not cfg.get("array_struct_features"):
        return None, []
    try:
        from core.array_struct_aggregator import ArrayStructAggregator
    except ImportError as exc:
        log.warning(f"[ATTR] No se pudo importar ArrayStructAggregator: {exc}")
        return None, []

    agg_path = (cfg.get("paths") or {}).get("array_struct_aggregator")
    aggregator = None
    if agg_path and Path(agg_path).exists():
        try:
            aggregator = ArrayStructAggregator.load(Path(agg_path))
        except Exception as exc:
            log.warning(f"[ATTR] Aggregator no carga ({agg_path}): {exc}")
            aggregator = None
    if aggregator is None:
        log.info("[ATTR] Aggregator no persistido — fit on-the-fly para day-stats")
        aggregator = ArrayStructAggregator().fit(df, cfg)

    return aggregator, aggregator.list_array_cols()


def compute_per_column_stats(raw_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    date_col = _resolve_date_col(cfg)
    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_date"] = df[date_col].dt.date

    # ── ArrayStructAggregator: detecta + carga/fit ──
    aggregator, array_cols = _maybe_load_array_aggregator(df, cfg)

    # Excluir columnas array del loop normal (las trataría mal _stats_for_one_column)
    cols = [c for c in df.columns if c not in (date_col, "_date") and c not in array_cols]

    rows = []
    for date in sorted(df["_date"].unique()):
        day = df[df["_date"] == date]
        n = len(day)
        for col in cols:
            s = _stats_for_one_column(day[col], n)
            s["date"], s["column"] = pd.Timestamp(date), col
            rows.append(s)
        # ── Pseudo-columnas de array_struct (entropy/hhi/top1/... sobre flatten) ──
        if aggregator is not None:
            for s in aggregator.compute_day_stats(day):
                col = s.pop("column")
                s["date"], s["column"] = pd.Timestamp(date), col
                rows.append(s)

    return pd.DataFrame(rows)[["date","column"]+_STAT_COLS].sort_values(["date","column"]).reset_index(drop=True)


def build_per_column_stats_history(raw_df: pd.DataFrame, cfg: dict, force=False) -> pd.DataFrame:
    path = _per_column_stats_path(cfg)
    if path.exists() and not force:
        log.info(f"[ATTR] Cargando desde {path}")
        return pd.read_parquet(path)
    out = compute_per_column_stats(raw_df, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info(f"[ATTR] {len(out):,} filas → {path}")
    return out


# ── Atribución: factory unificada ─────────────────────────────────────────────

def _baseline_window(per_col_df, target_date, days):
    start = target_date - pd.Timedelta(days=days)
    return per_col_df[(per_col_df["date"] >= start) & (per_col_df["date"] < target_date)]

def _day_slice(per_col_df, date):
    return per_col_df[per_col_df["date"] == date].set_index("column")

def _baseline_medians(bdf):
    return bdf.groupby("column")[_STAT_COLS].median(numeric_only=True)


def _rank_by_delta(day, base, stat, direction="up", filter_mask=None, top_k=5, min_delta=1e-4):
    """Core: ranking de columnas por delta de un stat vs baseline."""
    today_vals = day[stat] if stat in day.columns else pd.Series(dtype=float)
    base_vals = base[stat] if stat in base.columns else pd.Series(dtype=float)
    both = pd.concat([today_vals.rename("today"), base_vals.rename("base")], axis=1)
    if stat in ("pct_null","pct_empty","pct_unknown"): both["base"] = both["base"].fillna(0.0)
    both["today"] = both["today"].fillna(0.0) if stat in ("pct_null","pct_empty","pct_unknown") else both["today"]
    both["delta"] = both["today"] - both["base"]
    both["score"] = both["delta"].clip(lower=0) if direction == "up" else ((-both["delta"]).clip(lower=0) if direction == "down" else both["delta"].abs())
    if filter_mask is not None: both = both[filter_mask.reindex(both.index, fill_value=False)]
    both = both[both["score"] > min_delta].sort_values("score", ascending=False).head(top_k)
    return [{"column": str(c), "weight": float(r["score"]),
             "details": {"stat": stat, "today": float(r["today"]), "baseline": float(r["base"]), "delta": float(r["delta"])}}
            for c, r in both.iterrows()]


# Factory functions — all delegate to _rank_by_delta with filters
def _attrib_stat(stat, direction="up"):
    return lambda day, base, top_k: _rank_by_delta(day, base, stat, direction, top_k=top_k)

def _attrib_max(stat):
    def _f(day, base, top_k):
        vals = day[stat].dropna().sort_values(ascending=False).head(top_k)
        base_vals = base[stat] if stat in base.columns else pd.Series(dtype=float)
        return [{"column": str(c), "weight": float(v),
                 "details": {"stat": stat, "today": float(v),
                             "baseline": float(base_vals.get(c, 0)), "delta": float(v - base_vals.get(c, 0))}}
                for c, v in vals.items()]
    return _f

def _attrib_min(stat):
    def _f(day, base, top_k):
        vals = day[stat].dropna().sort_values(ascending=True).head(top_k)
        base_vals = base[stat] if stat in base.columns else pd.Series(dtype=float)
        return [{"column": str(c), "weight": float(abs(base_vals.get(c, 1.0) - v)),
                 "details": {"stat": stat, "today": float(v),
                             "baseline": float(base_vals.get(c, 1.0)), "delta": float(v - base_vals.get(c, 1.0))}}
                for c, v in vals.items()]
    return _f

def _attrib_pct(stat, pct, direction="up"):
    def _f(day, base, top_k):
        vals = day[stat].dropna()
        if vals.empty: return []
        q = float(np.percentile(vals.values, pct))
        mask = day[stat] >= q if pct >= 50 else day[stat] <= q
        return _rank_by_delta(day, base, stat, direction, filter_mask=mask, top_k=top_k)
    return _f

def _attrib_gini(stat):
    def _f(day, base, top_k):
        vals = day[stat].dropna()
        if vals.empty: return []
        med = float(vals.median())
        dev = (day[stat] - med).abs()
        base_vals = base[stat] if stat in base.columns else pd.Series(dtype=float)
        both = pd.concat([day[stat].rename("today"), base_vals.rename("base").fillna(0), dev.rename("dev")], axis=1)
        both = both.dropna(subset=["today"]).sort_values("dev", ascending=False).head(top_k)
        return [{"column": str(c), "weight": float(r["dev"]),
                 "details": {"stat": stat, "today": float(r["today"]), "baseline": float(r["base"]), "delta": float(r["today"]-r["base"])}}
                for c, r in both.iterrows()]
    return _f

def _attrib_threshold(stat, thr):
    def _f(day, base, top_k):
        tv = day[stat] if stat in day.columns else pd.Series(dtype=float)
        bv = (base[stat] if stat in base.columns else pd.Series(dtype=float)).fillna(0)
        ab = bv.reindex(tv.index).fillna(0)
        crossed = tv[tv > thr]
        newly = crossed[ab.reindex(crossed.index).fillna(0) <= thr].sort_values(ascending=False).head(top_k)
        return [{"column": str(c), "weight": float(v-thr),
                 "details": {"stat": stat, "today": float(v), "baseline": float(ab.get(c,0)), "delta": float(v-ab.get(c,0))}}
                for c, v in newly.items()]
    return _f

def _attrib_zero_null(day, base, top_k):
    tv = day["pct_null"]
    bv = (base["pct_null"].fillna(0) if "pct_null" in base.columns else pd.Series(dtype=float)).reindex(tv.index).fillna(0)
    cols = tv[(bv == 0) & (tv > 0)].sort_values(ascending=False).head(top_k)
    return [{"column": str(c), "weight": float(v),
             "details": {"stat": "pct_null", "today": float(v), "baseline": 0., "delta": float(v)}}
            for c, v in cols.items()]


# ── Registro de reglas ────────────────────────────────────────────────────────

ATTRIBUTION_RULES: Dict[str, Callable] = {
    "qual_mean_pct_null":        _attrib_stat("pct_null"), "qual_std_pct_null": _attrib_gini("pct_null"),
    "qual_p25_pct_null":         _attrib_pct("pct_null",25), "qual_p75_pct_null": _attrib_pct("pct_null",75),
    "qual_p95_pct_null":         _attrib_pct("pct_null",95), "qual_max_pct_null": _attrib_max("pct_null"),
    "qual_gini_pct_null":        _attrib_gini("pct_null"),
    "qual_pct_cols_over10_null": _attrib_threshold("pct_null",0.10),
    "qual_pct_cols_fully_null":  _attrib_threshold("pct_null",0.999),
    "qual_pct_cols_zero_null":   _attrib_zero_null,
    "qual_mean_pct_empty":       _attrib_stat("pct_empty"), "qual_max_pct_empty": _attrib_max("pct_empty"),
    "qual_mean_pct_unknown":     _attrib_stat("pct_unknown"), "qual_max_pct_unknown": _attrib_max("pct_unknown"),
    "qual_n_cols_any_null":      _attrib_threshold("pct_null",0.0),
    "dist_mean_entropy_cat":  _attrib_stat("entropy_norm","down"), "dist_std_entropy_cat": _attrib_gini("entropy_norm"),
    "dist_p25_entropy_cat":   _attrib_pct("entropy_norm",25,"down"), "dist_p75_entropy_cat": _attrib_pct("entropy_norm",75,"down"),
    "dist_min_entropy_cat":   _attrib_min("entropy_norm"),
    "dist_mean_hhi":          _attrib_stat("hhi"), "dist_max_hhi": _attrib_max("hhi"),
    "dist_p25_hhi":           _attrib_pct("hhi",25), "dist_p75_hhi": _attrib_pct("hhi",75),
    "dist_mean_top1_share":   _attrib_stat("top1_share"), "dist_max_top1_share": _attrib_max("top1_share"),
    "dist_mean_n_cats":       _attrib_stat("n_cats","both"), "dist_std_n_cats": _attrib_gini("n_cats"),
    "dyn_delta_mean_pct_null":    _attrib_stat("pct_null"), "dyn_delta_max_pct_null": _attrib_max("pct_null"),
    "dyn_delta_gini_pct_null":    _attrib_gini("pct_null"), "dyn_delta_mean_entropy_cat": _attrib_stat("entropy_norm","down"),
    "dyn_delta_mean_hhi":         _attrib_stat("hhi"), "dyn_delta_mean_top1_share": _attrib_stat("top1_share"),
    "dyn_delta_mean_pct_empty":   _attrib_stat("pct_empty"), "dyn_delta_mean_pct_unknown": _attrib_stat("pct_unknown"),
    "dyn_delta_n_cats":           _attrib_stat("n_cats","both"),
}

_COH_COMPONENTS = {
    "coh_quality_score":      ["qual_mean_pct_null","qual_mean_pct_empty","qual_mean_pct_unknown","qual_pct_cols_fully_null"],
    "coh_null_x_volume":      ["qual_mean_pct_null"], "coh_null_volume_trend": ["qual_mean_pct_null"],
    "coh_pct_cols_degraded":  ["qual_mean_pct_null","qual_mean_pct_empty","qual_mean_pct_unknown"],
    "coh_distribution_drift": ["dist_mean_entropy_cat","dist_mean_hhi","dist_mean_top1_share"],
    "coh_null_spread_ratio":  ["qual_max_pct_null","qual_mean_pct_null"],
    "coh_cat_stability_score":["dist_mean_entropy_cat","dist_mean_hhi"],
}


# ── Atribución principal ─────────────────────────────────────────────────────

def attribute_features_to_columns(top_features, feature_weights, date, per_col_df, baseline_days=30, top_k_per_feat=5):
    target_date = pd.Timestamp(date).normalize()
    day = _day_slice(per_col_df, target_date)
    if day.empty: return {}

    bw = _baseline_window(per_col_df, target_date, baseline_days)
    base = _baseline_medians(bw) if not bw.empty else pd.DataFrame()
    d7 = _day_slice(per_col_df, target_date - pd.Timedelta(days=7))

    out = {}
    for feat, w in zip(top_features, feature_weights):
        entry = {"weight": float(w), "columns": [], "cover": "none"}

        if feat in ATTRIBUTION_RULES:
            rule = ATTRIBUTION_RULES[feat]
            b = d7 if feat.startswith("dyn_delta_") else base
            if not b.empty:
                try:
                    entry["columns"] = rule(day, b, top_k=top_k_per_feat)
                    entry["cover"] = "direct"
                except Exception as exc:
                    log.debug(f"[ATTR] Regla directa falló para '{feat}': {exc}")

        elif feat in _COH_COMPONENTS:
            agg = {}
            for comp in _COH_COMPONENTS[feat]:
                if comp in ATTRIBUTION_RULES:
                    try:
                        for c in ATTRIBUTION_RULES[comp](day, base, top_k=top_k_per_feat):
                            if c["column"] in agg: agg[c["column"]]["weight"] += c["weight"]
                            else: agg[c["column"]] = dict(c)
                    except Exception as exc:
                        log.debug(f"[ATTR] Regla indirecta '{comp}' falló para '{feat}': {exc}")
            merged = sorted(agg.values(), key=lambda x: x["weight"], reverse=True)[:top_k_per_feat]
            entry["columns"], entry["cover"] = merged, ("indirect" if merged else "none")

        out[feat] = entry
    return out


def rank_columns_across_features(attribution, top_k=5, min_features=1):
    agg = {}
    for feat, info in attribution.items():
        if info["cover"] == "none": continue
        fw = float(info["weight"])
        for c in info["columns"]:
            cn = c["column"]
            b = agg.setdefault(cn, {"column": cn, "total_weight": 0., "features": [], "stats_today": {}, "stats_baseline": {}, "deltas": {}})
            b["total_weight"] += float(c["weight"]) * (fw / 100 + 1e-6)
            b["features"].append(feat)
            d = c["details"]
            b["stats_today"][d["stat"]] = d["today"]
            b["stats_baseline"][d["stat"]] = d["baseline"]
            b["deltas"][d["stat"]] = d["delta"]
    ranked = sorted(agg.values(), key=lambda x: x["total_weight"], reverse=True)
    return [r for r in ranked if len(r["features"]) >= min_features][:top_k]


_STAT_LABELS = {"pct_null":"% null","pct_empty":"% empty","pct_unknown":"% unknown",
                "entropy_norm":"entropy","hhi":"HHI","top1_share":"top1_share","n_cats":"n_cats","n_rows":"n_rows"}

def format_attribution_report(column_ranking, baseline_days=_BASELINE_DAYS_DEFAULT, max_stats=5):
    if not column_ranking: return "  (Sin atribución column-level disponible)"
    lines = [f"  Baseline: mediana de los {baseline_days} días previos a D.", ""]
    for col in column_ranking:
        lines.append(f"  {col['column']}")
        for stat, delta in sorted(col["deltas"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_stats]:
            lbl, t, b = _STAT_LABELS.get(stat, stat), col["stats_today"].get(stat, float("nan")), col["stats_baseline"].get(stat, float("nan"))
            if stat in ("pct_null","pct_empty","pct_unknown","top1_share","hhi","entropy_norm"):
                lines.append(f"    · {lbl:<12} {t*100:6.2f}%   (baseline {b*100:6.2f}%)   Δ {delta*100:+6.2f}pp")
            else:
                lines.append(f"    · {lbl:<12} {t:8.2f}   (baseline {b:8.2f})   Δ {delta:+8.2f}")
        lines.append(f"    Features: {', '.join(sorted(set(col['features'])))}")
        lines.append("")
    return "\n".join(lines)
