"""stress_test.py — Inyección de anomalías sintéticas en espacio escalado.

Valida la sensibilidad de Tier 3 (TranAD) corrompiendo el vector de un día
sano en el espacio de features escaladas y comprobando que el modelo lo
detecta. Para validar T1/T2/T4 con corrupciones reales sobre filas usar
`fake_error.make_fake_error_table` + `runner.inject`.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CLIP_MAX, CLIP_MIN = 5.0, -5.0


# ── Tabla de inyección: (feature, direction, multiplier) ─────────────────────
_INJECT_SPEC = {
    "quality_spike": [
        ("qual_max_pct_null", "max", 1.), ("qual_p95_pct_null", "max", 1.),
        ("qual_gini_pct_null", "max", 1.), ("qual_pct_cols_fully_null", "max", .5),
        ("dyn_delta_max_pct_null", "max", 1.), ("coh_null_spread_ratio", "max", 1.),
        ("coh_quality_score", "max", .4),
    ],
    "quality_global": [
        ("qual_mean_pct_null", "max", 1.), ("qual_std_pct_null", "max", .3),
        ("qual_p75_pct_null", "max", 1.), ("qual_pct_cols_over10_null", "max", 1.),
        ("qual_mean_pct_empty", "max", 1.), ("qual_max_pct_empty", "max", 1.),
        ("qual_mean_pct_unknown", "max", 1.), ("coh_pct_cols_degraded", "max", 1.),
        ("coh_quality_score", "max", 1.), ("dyn_delta_mean_pct_null", "max", 1.),
        ("dyn_delta_mean_pct_empty", "max", 1.),
    ],
    "cat_disappear": [
        ("dyn_pct_disappeared_cat_values", "max", 1.), ("dyn_n_disappeared_cat_values", "max", 1.),
        ("dist_max_hhi", "max", 1.), ("dist_p75_hhi", "max", .7),
        ("dist_min_entropy_cat", "min", 1.), ("dyn_delta_mean_hhi", "max", 1.),
        ("dyn_delta_mean_top1_share", "max", 1.), ("coh_distribution_drift", "max", 1.),
        ("coh_cat_stability_score", "min", 1.),
    ],
    "cat_appear": [
        ("dyn_pct_new_cat_values", "max", 1.), ("dyn_n_new_cat_values", "max", 1.),
        ("dyn_delta_mean_entropy_cat", "max", 1.), ("dyn_delta_n_cats", "max", 1.),
        ("coh_distribution_drift", "max", 1.), ("dist_mean_entropy_cat", "max", .5),
    ],
    "volume_drop": [
        ("vol_log_row_count", "min", 1.), ("vol_row_count_raw", "min", 1.),
        ("dyn_row_count_delta_pct", "min", 1.), ("vol_row_count_7d_slope", "min", .5),
    ],
    "schema_change": [
        ("schema_n_new_cols", "max", 1.), ("schema_n_total_cols", "max", .3),
    ],
    "distribution_shift": [
        ("coh_distribution_drift", "max", 1.), ("dyn_delta_mean_entropy_cat", "max", 1.),
        ("dyn_delta_mean_hhi", "max", 1.), ("dyn_delta_mean_top1_share", "max", 1.),
    ],
}

_DEFAULT_INTENSITY = {
    "quality_global":     0.60,
    "volume_drop":        0.70,
    "distribution_shift": 0.80,
}

_LABELS = {
    "quality_spike":      "Spike de nulos en columna específica",
    "quality_global":     "Degradación global de calidad",
    "cat_disappear":      "Valor categórico desaparece",
    "cat_appear":         "Valor nuevo aparece de golpe",
    "volume_drop":        "Caída de volumen del pipeline",
    "schema_change":      "Columnas nuevas en esquema",
    "distribution_shift": "Cambio brusco de distribución",
}


def _val(intensity, direction):
    if intensity is None:
        return CLIP_MAX if direction == "max" else CLIP_MIN
    base = CLIP_MAX if direction == "max" else CLIP_MIN
    return float(np.clip(intensity * base, CLIP_MIN, CLIP_MAX))


def _inject(scaled_array, feature_cols, day_idx, spec, intensity):
    arr = scaled_array.copy()
    for feat, direction, mult in spec:
        try:
            idx = feature_cols.index(feat)
            arr[day_idx, idx] = np.clip(_val(intensity, direction) * mult, CLIP_MIN, CLIP_MAX)
        except ValueError:
            pass
    return arr


def _find_clean_day(idx, alerts_df, cfg, direction=-1):
    if not cfg.get("stress_test", {}).get("skip_existing_anomalies", True):
        return idx
    for _ in range(cfg.get("stress_test", {}).get("max_clean_day_search", 30)):
        if 0 <= idx < len(alerts_df) and not alerts_df.iloc[idx]["anomaly"]:
            return idx
        idx += direction
    return idx


# ── Motor principal ──────────────────────────────────────────────────────────

def run_stress_tests(tests, scaled_data, features_df, feature_cols, model,
                     alerts_fn, thresholds, cfg):
    from core.model import compute_all_errors
    import logging as _lg

    seq_len = cfg.get("tranad", {}).get("seq_len", 14)
    dates = features_df["date"].reset_index(drop=True)

    err_w1_b, err_w2_b = compute_all_errors(model, scaled_data, cfg)
    dates_sc = dates.iloc[seq_len:].reset_index(drop=True)

    _sl = _lg.getLogger("scoring"); _sl.setLevel(_lg.WARNING)
    alerts_base, _ = alerts_fn(err_w1_b, err_w2_b, dates_sc, feature_cols, thresholds, cfg)
    _sl.setLevel(_lg.INFO)

    print(f"\n{'═'*65}\n  🧪 STRESS TEST — {len(tests)} tests\n{'═'*65}")

    results = []
    for date_ref, test_type, intensity in tests:
        label = _LABELS.get(test_type, test_type)
        int_d = f"{intensity:.0%}" if isinstance(intensity, float) else "máximo"
        print(f"\n  ▶ {label} ({int_d})")

        # Resolver índice del día
        if isinstance(date_ref, str):
            m = dates[dates >= date_ref]
            raw_idx = m.index[0] if not m.empty else len(dates) - 1
        else:
            raw_idx = len(dates) + date_ref if date_ref < 0 else int(date_ref)

        clean_idx = _find_clean_day(raw_idx - seq_len, alerts_base, cfg)
        day_idx = clean_idx + seq_len
        if day_idx >= len(dates):
            day_idx = len(dates) - 1
        d = dates.iloc[day_idx]
        print(f"    Día: {d.date() if hasattr(d, 'date') else d}")

        fi = intensity if isinstance(intensity, float) else None
        if test_type in _INJECT_SPEC:
            corrupted = _inject(scaled_data, feature_cols, day_idx,
                                _INJECT_SPEC[test_type],
                                fi or _DEFAULT_INTENSITY.get(test_type))
        else:
            print(f"    ⚠️ Tipo desconocido: {test_type}")
            continue

        err_w1_c, err_w2_c = compute_all_errors(model, corrupted, cfg)
        adj = day_idx - seq_len
        if adj < 0 or adj >= len(err_w1_c):
            results.append({"test": test_type, "detected": None})
            continue

        _sl.setLevel(_lg.WARNING)
        alerts_c, _ = alerts_fn(err_w1_c, err_w2_c, dates_sc, feature_cols, thresholds, cfg)
        _sl.setLevel(_lg.INFO)

        row = alerts_c.iloc[adj]
        det  = bool(row["anomaly"])
        conf = max(float(row["confidence_quality"]),
                   float(row["confidence_volume"]),
                   float(row["confidence_structural"]))
        icon = "✅" if det else "❌"
        print(f"    {icon} conf={conf:.2f}x | Q={row['score_quality']:.4f} "
              f"V={row['score_volume']:.4f} S={row['score_structural']:.4f} | "
              f"{row['dominant_channel']}")

        results.append({
            "test":             test_type,
            "label":            label,
            "intensity":        str(intensity),
            "target_date":      str(dates.iloc[day_idx]),
            "detected":         det,
            "confidence":       round(conf, 3),
            "dominant_channel": str(row["dominant_channel"]),
            "score_quality":    round(float(row["score_quality"]),    6),
            "score_volume":     round(float(row["score_volume"]),     6),
            "score_structural": round(float(row["score_structural"]), 6),
        })

    df = pd.DataFrame(results)
    valid = df[df["detected"].notna()]
    n_det = int(valid["detected"].sum()) if len(valid) else 0
    if len(valid) > 0 and valid["detected"].all():
        status = "✅ VÁLIDO"
    elif len(valid) > 0 and valid["detected"].any():
        status = "⚠️ PARCIAL"
    else:
        status = "🔴 CIEGO"
    print(f"\n{'═'*65}\n  📊 {n_det}/{len(valid)} detectados — {status}\n{'═'*65}\n")
    return df


def run_stress_test(scaled_data, features_df, feature_cols, model,
                    alerts_fn, thresholds, cfg):
    """Versión 'standard' compacta (3 escenarios canónicos)."""
    offset = cfg.get("stress_test", {}).get("target_date_offset", -7)
    return run_stress_tests(
        [(offset, "quality_spike", 0.80),
         (offset, "volume_drop",   0.70),
         (offset, "cat_disappear", None)],
        scaled_data, features_df, feature_cols, model, alerts_fn, thresholds, cfg,
    )
