"""
main.py — Entry points del sistema de Data Health
==================================================

API pública:
  run_multi_tier_mode(table, tiers=[1,2,3], ...) → resultados del pipeline

Notas de diseño:
  - El parámetro `tiers` permite ejecutar solo un subconjunto. Por ejemplo:
       run_multi_tier_mode(tiers=[1, 2])    # sin TranAD
       run_multi_tier_mode(tiers=[3])       # solo TranAD
  - Cada tier es independiente: se carga su artefacto desde `cfg["paths"]`
    si existe (y `force_retrain=False`), o se ajusta desde cero si no.
  - Tier 4 (row-level AE) no se ejecuta aquí — usa `RowLevelPipeline`
    directamente o `runner.row_level()`.
"""

import sys
import logging
import pickle

import numpy as np
import pandas as pd
import torch

from config import get_cfg
from pipelines.base import run_feature_pipeline
from core.column_attribution import build_per_column_stats_history
from core.model import train_tranad, compute_all_errors, TranAD
from core.scoring import (
    compute_thresholds, compute_dynamic_thresholds, compute_dow_volume_baseline,
    generate_alerts,
)
from pipelines.column_level import (
    run_column_level_pipeline, _load_feature_contract,
    _pivot_long_to_wide, _excluded_columns,
)
from core.statistical_detector import ColumnStatisticalDetector
from core.pca_detector import CrossColumnPCADetector
from core.alert_fusion import score_period, combine_day

logging.basicConfig(format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("main")

torch.set_float32_matmul_precision("medium")


# ─────────────────────────────────────────────────────────────────────────────
def _run_table_pipeline(table: str, cfg: dict):
    """Despacha al feature pipeline correspondiente."""
    if table == "discounts":
        from pipelines.discounts import run_pipeline
        return run_pipeline(cfg)

    # Default: pipeline genérico para tablas evento (portabilidades, error_mvt…)
    raw_df, scaled_df, feature_cols = run_feature_pipeline(cfg)
    per_col_df = build_per_column_stats_history(
        raw_df, cfg, force=cfg.get("force_features", False),
    )
    return raw_df, scaled_df, feature_cols, per_col_df


# ─────────────────────────────────────────────────────────────────────────────
def run_multi_tier_mode(
    table: str = "portabilidades",
    tiers: list = None,
    force_retrain: bool = False,
    force_refeatures: bool = False,
    explain_top_k: int = 3,
    cfg_override: dict = None,
    inference_mode: bool = False,
):
    """Pipeline Multi-Tier con tiers seleccionables.

    Args:
        table:            nombre lógico de la tabla (clave en config.TABLES)
        tiers:            subset de [1, 2, 3] a ejecutar. None → [1, 2, 3].
        force_retrain:    refitea artefactos aunque existan en disco.
        force_refeatures: recomputa features aunque exista cache.
        explain_top_k:    nº de días anómalos para generar reporte combinado.
        cfg_override:     si se pasa, usa este cfg en lugar de get_cfg(table).
                          NO implica inferencia: úsalo para inyectar un cfg
                          mutado (deepcopy en bootstrap, paths efímeros, etc.).
        inference_mode:   activa rama de inferencia (T3 carga checkpoint en
                          lugar de entrenar; thresholds dinámicos; T2 carga
                          contrato frozen). Default False (training).

    Returns:
        dict con period_df, tier1, tier2, tier3_results, reports, etc.
        Las claves de tiers desactivados llegan como None.
    """
    if tiers is None:
        tiers = [1, 2, 3]
    tiers = [int(t) for t in tiers]
    do_t1 = 1 in tiers
    do_t2 = 2 in tiers
    do_t3 = 3 in tiers

    # ── 0. Config ────────────────────────────────────────────────────────────
    CFG = cfg_override if cfg_override is not None else get_cfg(table)

    CFG["force_features"] = force_refeatures
    CFG["force_retrain"]  = force_retrain

    log.info(f"\n{'='*60}\n  MULTI-TIER — tabla: {table} "
             f"{'(INFERENCIA)' if inference_mode else '(STANDARD)'}  "
             f"tiers={tiers}\n{'='*60}")

    # ── 1. Feature pipeline (siempre — per_col_df lo usan T1 y T2) ───────────
    raw_df, scaled_df_univ, feature_cols_univ, per_col_df = _run_table_pipeline(table, CFG)

    dates_all = sorted(per_col_df["date"].unique())
    n_days    = len(dates_all)
    excluded  = _excluded_columns(CFG)

    train_split = CFG.get("tranad", {}).get("train_split", 0.95)
    n_train     = int(n_days * train_split)
    train_mask  = np.array([True] * n_train + [False] * (n_days - n_train))

    # ── 2. Column-level features (solo si T2) ────────────────────────────────
    scaled_data_cl: np.ndarray = None
    dates_cl                  = None
    feature_cols_cl           = None
    if do_t2:
        if inference_mode:
            contract_path = CFG["paths"]["column_level_features"]
            scaler_path   = CFG["paths"]["column_level_scaler"]
            expected_features = _load_feature_contract(contract_path)
            with open(scaler_path, "rb") as f:
                scaler_cl = pickle.load(f)
            wide = _pivot_long_to_wide(per_col_df, excluded)
            wide = wide.reindex(columns=expected_features).ffill().fillna(0.0)
            scaled_data_cl  = np.clip(scaler_cl.transform(wide), -5.0, 5.0).astype(np.float32)
            dates_cl        = pd.DatetimeIndex(wide.index)
            feature_cols_cl = expected_features
            log.info(f"[COL_LEVEL] Inferencia: {wide.shape[0]} días × "
                     f"{len(expected_features)} features")
        else:
            scaled_df_cl, scaled_data_cl, feature_cols_cl = run_column_level_pipeline(
                raw_df, CFG, per_col_df=per_col_df,
            )
            dates_cl = pd.DatetimeIndex(scaled_df_cl["date"])

        n_train_cl = int(len(dates_cl) * train_split)
        train_mask_cl = np.array([True] * n_train_cl + [False] * (len(dates_cl) - n_train_cl))

    # ── 3. Tier 1 — ColumnStatisticalDetector ────────────────────────────────
    tier1 = None
    if do_t1:
        tier1_path = CFG["paths"]["tier1_baselines"]
        hp_t1 = CFG.get("tier1", {})
        tier1 = ColumnStatisticalDetector(
            z_threshold     = hp_t1.get("z_threshold",     4.0),
            min_flagged_pct = hp_t1.get("min_flagged_pct", 0.10),
            min_history     = hp_t1.get("min_history",     8),
            min_rows        = hp_t1.get("min_rows",        100),
        )
        if tier1_path.exists() and not force_retrain:
            tier1.load(tier1_path)
        else:
            tier1.fit(per_col_df, train_mask, excluded)
            tier1.save(tier1_path)

    # ── 4. Tier 2 — CrossColumnPCADetector ───────────────────────────────────
    tier2 = None
    if do_t2:
        tier2_path = CFG["paths"]["tier2_pca"]
        hp_t2 = CFG.get("tier2", {})
        tier2 = CrossColumnPCADetector(
            explained_variance = hp_t2.get("explained_variance", 0.95),
            z_threshold        = hp_t2.get("z_threshold",        4.0),
        )
        if tier2_path.exists() and not force_retrain:
            tier2.load(tier2_path)
        else:
            tier2.fit(scaled_data_cl, dates_cl, train_mask_cl, feature_cols_cl)
            tier2.save(tier2_path)

    # ── 5. Tier 3 — TranAD ───────────────────────────────────────────────────
    tier3_alerts   = None
    model          = None
    errors_w1      = None
    errors_w2      = None
    channel_scores = None
    thresholds     = None
    dates_scored   = None
    if do_t3:
        log.info("[MT] Tier 3 — TranAD...")
        scaled_data_univ = scaled_df_univ[
            [c for c in feature_cols_univ if c in scaled_df_univ.columns]
        ].values
        dates_univ = pd.DatetimeIndex(scaled_df_univ["date"])
        seq_len = CFG["tranad"]["seq_len"]

        if inference_mode:
            ckpt = CFG["paths"]["tranad_model"]
            log.info(f"[MT] Cargando TranAD desde {ckpt}")
            model = TranAD.load_from_checkpoint(str(ckpt), weights_only=False)
            model.eval()
        else:
            tranad_split = CFG.get("tranad", {}).get("train_split", 0.95)
            n_train_t3 = int(len(scaled_data_univ) * tranad_split)
            train_mask_t3 = np.array([True] * n_train_t3
                                     + [False] * (len(scaled_data_univ) - n_train_t3))
            model = train_tranad(scaled_data_univ, feature_cols_univ, CFG, train_mask_t3)

        errors_w1, errors_w2 = compute_all_errors(model, scaled_data_univ, CFG)
        dates_scored = dates_univ[seq_len:]
        n_eval = CFG.get("_eval_days", None)
        if inference_mode:
            thresholds = compute_dynamic_thresholds(
                errors_w1, errors_w2, feature_cols_univ, CFG,
                k_sigma=CFG.get("tranad", {}).get("k_sigma", 5.0),
                n_eval=n_eval,
            )
        else:
            tranad_split = CFG.get("tranad", {}).get("train_split", 0.95)
            n_train_t3 = int(len(scaled_data_univ) * tranad_split)
            train_mask_t3 = np.array([True] * n_train_t3
                                     + [False] * (len(scaled_data_univ) - n_train_t3))
            thresholds = compute_thresholds(
                errors_w1, errors_w2, feature_cols_univ, train_mask_t3, seq_len, CFG,
            )

        # Row counts + baseline DOW de volumen
        _rc = per_col_df.groupby("date")["n_rows"].first()
        row_counts = np.array([float(_rc.get(pd.Timestamp(d), np.nan))
                               for d in dates_scored])
        dow_vol_baseline = compute_dow_volume_baseline(
            per_col_df, n_eval=n_eval if inference_mode else None,
        )

        tier3_alerts, channel_scores = generate_alerts(
            errors_w1, errors_w2, dates_scored, feature_cols_univ, thresholds, CFG,
            row_counts=row_counts, dow_volume_baseline=dow_vol_baseline,
        )

    # ── 6. Evaluación combinada ──────────────────────────────────────────────

    if inference_mode:
        start_ts = pd.Timestamp(CFG["bq"]["start_date"])
        end_ts   = pd.Timestamp(CFG["bq"]["end_date"])

        eval_dates = [
            d for d in dates_all
            if start_ts <= pd.Timestamp(d) <= end_ts
        ]

    else:
        eval_dates = dates_all[n_train:]

        if not eval_dates:
            eval_dates = dates_all[-30:]

    period_df = score_period(
        per_col_df,
        scaled_data_cl if do_t2 else None,
        dates_cl       if do_t2 else None,
        tier1, tier2, eval_dates, excluded, tier3_alerts,
        cfg=CFG,
    )

    # ── 7. Reportes combinados para días anómalos top-K ──────────────────────
    reports = []
    anomaly_days = period_df[period_df["anomaly"]] \
        .sort_values("max_severity", ascending=False)

    for _, row in anomaly_days.head(explain_top_k).iterrows():
        target_date = row["date"]
        r1 = tier1.score_day(per_col_df, target_date, excluded) if do_t1 else None

        r2 = None
        if do_t2:
            day_idx = next((j for j, d in enumerate(dates_cl)
                            if pd.Timestamp(d) == pd.Timestamp(target_date)), None)
            r2 = (tier2.score_day(scaled_data_cl[day_idx], target_date)
                  if day_idx is not None
                  else {"anomaly": False, "z_score": 0, "total_error": 0})

        r3 = None
        if do_t3 and tier3_alerts is not None:
            t3_row = tier3_alerts[tier3_alerts["date"] == pd.Timestamp(target_date)]
            r3 = t3_row.iloc[0].to_dict() if not t3_row.empty else None

        reports.append(combine_day(target_date, r1, r2, r3, cfg=CFG))

    return {
        "period_df":     period_df,
        "tier1":         tier1,
        "tier2":         tier2,
        "per_col_df":    per_col_df,
        "scaled_data":   scaled_data_cl,
        "feature_cols":  feature_cols_cl,
        "dates":         dates_cl,
        "excluded_cols": excluded,
        "tier3_results": {
            "alerts_df":      tier3_alerts,
            "model":          model,
            "errors_w1":      errors_w1,
            "errors_w2":      errors_w2,
            "thresholds":     thresholds,
            "channel_scores": channel_scores,
            "feature_cols":   feature_cols_univ if do_t3 else None,
            "scaled_data":    scaled_data_univ if do_t3 else None,
            "dates":          dates_univ       if do_t3 else None,
            "dates_scored":   dates_scored,
        } if do_t3 else None,
        "tiers_run":     tiers,
        "reports":       reports,
    }
