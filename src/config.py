"""
config.py — Configuración multi-tabla del Sistema de Data Health
================================================================
Añadir una tabla nueva:
  1. Crear CFG_<TABLA> con "bq" y opcionalmente "tranad" / "scoring".
  2. Registrarla en TABLES al final.
  3. Crear src/pipelines/<tabla>.py.
"""

from pathlib import Path

BASE_DIR   = Path.cwd()
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
PLOTS_DIR  = BASE_DIR / "plots"

for d in [MODELS_DIR, DATA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _paths(table_name: str) -> dict:
    """Genera el dict de paths para una tabla, creando los directorios."""
    m = MODELS_DIR / table_name
    d = DATA_DIR   / table_name
    p = PLOTS_DIR  / table_name
    for dr in [m, d, p]:
        dr.mkdir(parents=True, exist_ok=True)
    return {
        # Datos
        "raw_data":              d / "raw_features.parquet",
        "scaled_data":           d / "scaled_features.parquet",
        "column_level_data":     d / "column_level_scaled.parquet",
        # Tier 3 — TranAD
        "tranad_model":          m / "tranad_best.ckpt",
        "scaler_quality":        m / "scaler_quality.pkl",
        "scaler_structural":     m / "scaler_structural.pkl",
        # Tier 1 — Estadístico DOW-aware
        "tier1_baselines":       m / "tier1_baselines.pkl",
        # Tier 2 — PCA cross-column
        "tier2_pca":             m / "tier2_pca_per_dow.pkl",
        "column_level_scaler":   m / "scaler_column_level.pkl",
        "column_level_features": m / "column_level_features.json",
        # Tier 4 — Row-Level AE
        "row_level_encoder":         m / "row_level_encoder.pkl",
        "row_level_model":           m / "row_level_best.pt",
        "row_level_vae_model":       m / "row_level_vae_best.pt",
        "row_level_thresholds":      m / "row_level_thresholds.json",
        "row_level_vae_thresholds":  m / "row_level_vae_thresholds.json",
        "array_struct_aggregator":   m / "array_struct_aggregator.pkl",
        # Visualizaciones
        "plots_dir":             p,
    }


# ── Defaults compartidos ─────────────────────────────────────────────────────
_MAX_EPOCHS    = 150
_PHASE1_EPOCHS = int(_MAX_EPOCHS * 0.4)
_PHASE2_EPOCHS = _MAX_EPOCHS - _PHASE1_EPOCHS

_DEFAULT_TRANAD = {
    "seq_len":       14,
    "d_model":       128,
    "n_heads":       4,
    "n_layers":      3,
    "dropout":       0.2,
    "lr":            1e-4,
    "batch_size":    16,
    "max_epochs":    _MAX_EPOCHS,
    "phase1_epochs": _PHASE1_EPOCHS,
    "phase2_epochs": _PHASE2_EPOCHS,
    "patience":      40,
    "train_split":   0.95,
    "k_sigma":       5.0,                # multiplicador MAD para umbrales dinámicos en inferencia
}

_DEFAULT_SCORING = {
    "threshold_percentile":        99.5,
    "quality_threshold_factor":    1.0,
    "volume_threshold_factor":     1.0,
    "structural_threshold_factor": 1.2,
    "min_rows_valid":              10,
    "min_reliable_rows":           200,   # bajo este nº, Quality/Structural se suprimen (Tier 3)
    "volume_collapse_k":           4.0,   # k·σ(MAD) por DOW para definir colapso de volumen
}

_DEFAULT_STRESS = {
    "active":                  True,
    "target_date_offset":      -7,
    "skip_existing_anomalies": True,
    "max_clean_day_search":    30,
}

_DEFAULT_EXPLAINABILITY = {
    "enable_shap":                True,
    "top_k_contributors":         5,
    "silence_check":              True,
    "attribution_baseline_days":  30,
    "attribution_top_k_columns":  5,
}

_DEFAULT_TIER1 = {
    "z_threshold":     4.0,
    "min_concentrated_cols": 2, # T1 fires si n_flagged >= 2
    "min_flagged_pct": 0.10,
    "min_history":     8,
    "min_rows":        100,    # guarda de volumen mínimo (días por debajo no se puntúan)
}

_DEFAULT_TIER2 = {
    "explained_variance": 0.95,
    "z_threshold":        4.0,    # decisión basada en z-score (no en p99 × factor)
}

_DEFAULT_ROW_LEVEL = {
    # Encoding
    "max_cardinality":    500,

    # Arquitectura
    "bottleneck_dim":     64,
    "encoder_layers":     [256, 128],
    "decoder_layers":     [128, 256],
    "dropout":            0.3,

    # Training
    "lr":                 1e-3,
    "batch_size":         4096,
    "max_epochs":         30,
    "patience":           5,
    "alpha_loss":         0.5,     # peso MSE vs CE

    # VAE
    "beta_kl":            1.0,

    # Scoring — calibración dinámica (median + K×MAD sobre ventana de inferencia,
    # análogo a Tier 3). El percentile se usa solo para el baseline_row guardado.
    "score_percentile":   99.0,
    "k_sigma":            5.0,
    "z_threshold_warn":   4.0,
    "z_threshold_crit":   6.0,

    # Datos
    "sample_pct_train":   10,
    "sample_pct_infer":   100,     # 100% para inferencia diaria (1 partición, barato)
    "train_start":        "2024-01-01",
    "train_end":          "2025-06-30",
    "val_start":          "2025-06-30",
    "val_end":            "2025-10-01",
    "eval_start":         "2025-10-01",
}


# ─────────────────────────────────────────────────────────────────────────────
# TABLA: portabilidades
# ─────────────────────────────────────────────────────────────────────────────

CFG_PORTABILIDADES = {
    "bq": {
        "project_id":      "mm-business-analysis",
        "dataset":         "SEMANTIC_PORTABILITY",
        "table":           "semantic_portability_sergio_catedra",
        "date_col":        "request_date",
        "phone_col":       "phone_nm",
        "category_cols":   [
            "operator_donor", "operator_receiver",
            "brand_donor",    "brand_receiver",
            "status",         "request_type",
        ],
        "numeric_cols":    [],
        "sample_hash_col": "phone_nm",
        "sample_pct":      10,
        "start_date":      "2023-01-01",
        "end_date":        "2025-12-31",
        "train_end":       "2025-10-31",
        "test_start":      "2025-11-01",
        "exclude_cols": [
            # Funcionales (descartadas por auditoría operativa)
            "billing_type_receiver", "brand_receiver", "bundle_id_donor",
            # PII — RGPD art. 4(1) / AI Act art. 10
            "customer_id_donor", "customer_id_receiver",
            "customer_gid_donor", "customer_gid_receiver",
            "account_id_donor", "account_id_receiver",
            "service_id_donor", "service_id_receiver",
            "zip_code_donor",
            "dealer_receiver", "subdealer_receiver",
        ],
    },
    "macro_diff": {
        "sample_pct":   10,
        "sample_key":   None,
        "exclude_cols": [],
        "start_date":   None,
        "end_date":     None,
    },
    "paths":          _paths("portabilidades"),
    "tranad":         dict(_DEFAULT_TRANAD),
    "scoring":        dict(_DEFAULT_SCORING),
    "stress_test":    dict(_DEFAULT_STRESS),
    "explainability": dict(_DEFAULT_EXPLAINABILITY),
    "tier1":          dict(_DEFAULT_TIER1),
    "tier2":          dict(_DEFAULT_TIER2),
    "force_download": False,
    "force_features": False,
    "force_retrain":  False,
    "verbose":        True,
    "row_level":      {**_DEFAULT_ROW_LEVEL, "exclude_cols": ["billing_type_receiver", "brand_receiver", "bundle_id_donor", 
        "customer_id_donor", "customer_id_receiver", "customer_gid_donor", "customer_gid_receiver",
        "account_id_donor", "account_id_receiver", "service_id_donor", "service_id_receiver", "zip_code_donor",
        "dealer_receiver", "subdealer_receiver",],
                      },
}


# ─────────────────────────────────────────────────────────────────────────────
# TABLA: discounts (semantic_discounts)
# ─────────────────────────────────────────────────────────────────────────────

CFG_DISCOUNTS = {
    "bq": {
        "project_id":       "mm-business-analysis",
        "dataset":          "SEMANTIC_DISCOUNTS",
        "table":            "semantic_discounts",
        "date_col":         None,
        "init_date_col":    "init_date",
        "end_date_col":     "max_end_date",
        "snapshot_col":     "snapshot_date",
        "pk_col":           "clave_pk",
        "category_cols":    [],
        "numeric_cols":     [],
        "sample_hash_col":  "clave_pk",
        "sample_pct":       10,
        "start_date":       "2021-01-01",
        "end_date":         "2023-12-31",
        "train_end":        "2023-11-30",
        "test_start":       "2023-12-01",
        "exclude_cols": [
            # Estructurales: >90% null permanente, son señal cero
            "campaign_name_commercial_profile", "commercial_profile",
            "dealer_code", "binding_type_name", "initial_penalty_amount",
            # Cobertura parcial alta (~75%): cobertura limitada estructural
            "canal", "servicio",
            # Maduración a meses (45%): requieren 4-5 meses de lag — fuera del SLA "fresh"
            "total_fee", "ARPU_GIVEN", "monthly_discount_value",
            "benefit_value", "ARPU_REAL", "sfid_id",],
    },
    "macro_diff": {
        "sample_pct":   10,
        "sample_key":   None,
        "exclude_cols": [],
        "start_date":   None,
        "end_date":     None,
    },
    "paths":          _paths("discounts"),
    "tranad":         {**_DEFAULT_TRANAD, "batch_size": 16},
    "scoring":        dict(_DEFAULT_SCORING),
    "stress_test":    dict(_DEFAULT_STRESS),
    "explainability": dict(_DEFAULT_EXPLAINABILITY),
    "tier1":          dict(_DEFAULT_TIER1),
    "tier2":          dict(_DEFAULT_TIER2),
    "force_download": False,
    "force_features": False,
    "force_retrain":  False,
    "verbose":        True,
    "row_level": {
        **_DEFAULT_ROW_LEVEL,
        "train_start": "2022-01-01",
        "train_end":   "2023-09-30",
        "val_start":   "2023-10-01",
        "val_end":     "2023-12-31",
        "sample_pct_infer": 10,
        # Excludes específicos del row-level (contrato MATURE):
        # Mantenemos DENTRO: total_fee, ARPU_GIVEN, monthly_discount_value,
        # benefit_value, ARPU_REAL → son el contenido de negocio que el AE debe ver.
        "exclude_cols": [
            "campaign_name_commercial_profile", "commercial_profile",
            "dealer_code", "binding_type_name", "initial_penalty_amount",
            "canal", "servicio",
            "clave_pk", "customer_id", "customer_gid", "account_id", "sfid_id",
            "discount_id", "discount_name",
            "min_end_date", "max_end_date", "end_date_planned",
            "account_id", "customer_gid", "customer_id"
        ],
    },
    "array_struct_features": {
        "funcional": {
            "categorical_tokens": {
                "service_type":   {"path": ["service_type"]},
                "technology_ds":  {"path": ["technology_ds"]},
                "line_type_ds":   {"path": ["line_type_ds"]},
                "bundle_type":    {"path": ["bundle_type"], "null_as_token": True},
                "segment_ds":     {"path": ["crm", "segment_ds"]},
                "sub_segment_ds": {"path": ["crm", "sub_segment_ds"]},
            },
            "numeric_aggs": {
                "tariff_fee":             {"path": ["tariff_fee"],                       "aggs": ["mean", "max"]},
                "monthly_discount_value": {"path": ["monthly_discount_value"],           "aggs": ["mean", "max"]},
                "ARPU_REAL":              {"path": ["ARPU_REAL"],                        "aggs": ["mean", "max"]},
                "ARPU_given":             {"path": ["ARPU_given"],                       "aggs": ["mean", "max"]},
                "imp_without_tax":        {"path": ["crm", "billed", "imp_without_tax"], "aggs": ["mean", "sum"]},
            },
            "length_features": [
                {"name": "len_outer",           "path": [],                "mode": "count"},
                {"name": "total_len_crm",       "path": ["crm"],           "mode": "count"},
                {"name": "total_len_billed",    "path": ["crm", "billed"], "mode": "count"},
                {"name": "pct_crm_with_billed", "path": ["crm"],           "mode": "pct_nonempty_child", "child": "billed"},
            ],
        }
    },
}






CFG_PORTABILIDADES_FRESH = {
    **CFG_PORTABILIDADES,
    "bq": {
        **CFG_PORTABILIDADES["bq"],
        "exclude_cols": [
            # 3 EXCLUDE (auditoría)
            "billing_type_receiver", "brand_receiver", "bundle_id_donor",
            # 41 REVIEW (lag 6d) — fuera del modelo fresh
            "transfer_date", "transfer_date_ts", "cus_bundle_type_receiver",
            "customer_gid_receiver", "line_type_receiver", "tariff_id_receiver",
            "tariff_ds_receiver", "sub_segment_receiver", "bundle_type_receiver",
            "bundle_name_receiver", "brand_aggr_receiver", "confirmation_date",
            "confirmation_date_ts", "bundle_id_receiver", "bundle_name_donor",
            "zip_code_donor", "sub_segment_n2_receiver", "bundle_type_donor",
            "sub_segment_donor", "tariff_id_donor", "tariff_ds_donor",
            "line_type_donor", "customer_gid_donor", "cus_bundle_type_donor",
            "sub_brand_donor", "comm_brand_donor", "customer_id_donor",
            "account_id_donor", "service_id_donor", "datasource_donor",
            "segment_donor", "brand_donor", "service_id_receiver",
            "segment_receiver", "datasource_receiver", "sub_brand_receiver",
            "comm_brand_receiver", "customer_id_receiver", "account_id_receiver",
            "dealer_receiver", "subdealer_receiver",
        ],
    },
    "paths": _paths("portabilidades_fresh"),
    "row_level": {
        **CFG_PORTABILIDADES["row_level"], 
        "k_sigma":          4.0,    # K en median + K·MAD; sube a 6 para más estricto
        "min_threshold_pct": 0.05,  # piso absoluto del umbral day-level (5%)
        "low_volume_pct":    0.01,  # día con <1% del volumen mediano → ruido
        "low_volume_abs":    200,   # o día con <200 filas en absoluto → ruido
        # Mismas exclusiones también para el row-level encoder
        "exclude_cols": (
            CFG_PORTABILIDADES["row_level"].get("exclude_cols", [])
            # Y todas las REVIEW también para que el AE row-level sea coherente
            + ["billing_type_receiver", "brand_receiver", "bundle_id_donor"]
            + ["transfer_date", "transfer_date_ts", "cus_bundle_type_receiver",
               "customer_gid_receiver", "line_type_receiver", "tariff_id_receiver",
               "tariff_ds_receiver", "sub_segment_receiver", "bundle_type_receiver",
               "bundle_name_receiver", "brand_aggr_receiver", "confirmation_date",
               "confirmation_date_ts", "bundle_id_receiver", "bundle_name_donor",
               "zip_code_donor", "sub_segment_n2_receiver", "bundle_type_donor",
               "sub_segment_donor", "tariff_id_donor", "tariff_ds_donor",
               "line_type_donor", "customer_gid_donor", "cus_bundle_type_donor",
               "sub_brand_donor", "comm_brand_donor", "customer_id_donor",
               "account_id_donor", "service_id_donor", "datasource_donor",
               "segment_donor", "brand_donor", "service_id_receiver",
               "segment_receiver", "datasource_receiver", "sub_brand_receiver",
               "comm_brand_receiver", "customer_id_receiver", "account_id_receiver",
               "dealer_receiver", "subdealer_receiver"]
        ),
    },
}



# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

TABLES: dict[str, dict] = {
    "portabilidades":       CFG_PORTABILIDADES,
    "portabilidades_fresh": CFG_PORTABILIDADES_FRESH,
    "discounts":            CFG_DISCOUNTS,
}


def get_cfg(table: str) -> dict:
    if table not in TABLES:
        raise KeyError(f"Tabla '{table}' no registrada. Disponibles: {sorted(TABLES)}")
    return TABLES[table]


# Alias para compatibilidad
CFG = CFG_PORTABILIDADES


if __name__ == "__main__":
    for name, cfg in TABLES.items():
        print(f"\n✅ {name}")
        print(f"   BQ    : {cfg['bq']['dataset']}.{cfg['bq']['table']}")
        print(f"   Modelo: {cfg['paths']['tranad_model']}")
