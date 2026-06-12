"""
row_level_pipeline.py — Orquestador del pipeline row-level (Tier 4)
====================================================================

MODIFICADO: integra ArrayStructAggregator. Si cfg["array_struct_features"]
está presente:
  - en train(): fit + transform_per_row antes del SchemaEncoder.fit, y
    drop de las columnas ARRAY originales (incompatibles con el encoder).
  - en save()/load(): persiste/carga el agregador junto al encoder.
  - en score_single_day() / score_range_dynamic(): aplica transform_per_row
    al day_df antes de delegar al scorer.

Si la clave no existe (ej. portabilidades), el comportamiento es idéntico
al anterior. No-op.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from core.schema_encoder import SchemaEncoder
from core.row_level_model import TabularAE, TabularVAE, RowLevelTrainer
from core.row_level_scoring import RowLevelScorer, DayReport

log = logging.getLogger(__name__)


class RowLevelPipeline:
    """Pipeline completo: descarga → (agg) → encode → train → calibrate → score."""

    def __init__(self, cfg: dict, model_type: str = "ae"):
        self.cfg        = cfg
        self.model_type = model_type
        self.rl         = cfg.get("row_level", {})
        self.bq         = cfg["bq"]
        self.paths      = cfg["paths"]

        self.encoder:    Optional[SchemaEncoder] = None
        self.model:      Optional[TabularAE]     = None
        self.scorer:     Optional[RowLevelScorer] = None
        self.aggregator                          = None      # ArrayStructAggregator | None
        self._raw_df:    Optional[pd.DataFrame]  = None

    # ── Paths ──────────────────────────────────────────────────────────────────

    @property
    def encoder_path(self) -> Path:
        return Path(self.paths["row_level_encoder"])

    @property
    def model_path(self) -> Path:
        key = "row_level_vae_model" if self.model_type == "vae" else "row_level_model"
        return Path(self.paths[key])

    @property
    def thresholds_path(self) -> Path:
        key = ("row_level_vae_thresholds" if self.model_type == "vae"
               else "row_level_thresholds")
        return Path(self.paths[key])

    @property
    def aggregator_path(self) -> Path:
        """Path para persistir el ArrayStructAggregator. Default: junto al encoder."""
        p = self.paths.get("array_struct_aggregator")
        return Path(p) if p else self.encoder_path.parent / "array_struct_aggregator.pkl"

    # ── Aggregator helper ──────────────────────────────────────────────────────

    def _apply_aggregator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transform_per_row y dropea columnas ARRAY originales."""
        if self.aggregator is None:
            return df
        df = self.aggregator.transform_per_row(df)
        for col in self.aggregator.list_array_cols():
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    # ── Descarga ───────────────────────────────────────────────────────────────

    def download(self, force: bool = False) -> pd.DataFrame:
        """Descarga los datos raw desde BQ (o caché parquet)."""
        cache_path = Path(str(self.paths["raw_data"]).replace(
            "raw_features", "row_level_raw"
        ))

        if cache_path.exists() and not force:
            log.info(f"[RL-PIPE] Caché: {cache_path}")
            self._raw_df = pd.read_parquet(cache_path)
            return self._raw_df

        from google.cloud import bigquery
        client = bigquery.Client(project=self.bq["project_id"])

        date_col = self.bq.get("date_col") or self.bq.get("init_date_col")
        if not date_col:
            raise RuntimeError("cfg.bq necesita date_col o init_date_col")
        fqn = f"`{self.bq['project_id']}.{self.bq['dataset']}.{self.bq['table']}`"

        start = self.rl.get("train_start", self.bq.get("start_date", "2024-01-01"))
        end   = self.bq.get("end_date", "2025-12-31")

        h   = self.bq.get("sample_hash_col")
        pct = self.rl.get("sample_pct_train", self.bq.get("sample_pct", 10))

        where_parts = [f"{date_col} BETWEEN '{start}' AND '{end}'"]
        if h and pct < 100:
            where_parts.append(
                f"MOD(ABS(FARM_FINGERPRINT(CAST({h} AS STRING))), 100) < {pct}"
            )
        where = " AND ".join(where_parts)
        query = f"SELECT * FROM {fqn} WHERE {where}"
        log.info(f"[RL-PIPE] Descargando desde BQ ({pct}% sample)...")
        df = client.query(query).to_dataframe(progress_bar_type="tqdm")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info(f"[RL-PIPE] {len(df):,} filas → {cache_path}")

        self._raw_df = df
        return df

    # ── Train ──────────────────────────────────────────────────────────────────

    def train(self, force_download: bool = False) -> Dict:
        """Pipeline completo: descarga → (agg) → encode → split → train → calibrate."""
        t0 = time.time()

        # 1. Descarga
        df = self.download(force=force_download)
        date_col = self.bq.get("date_col") or self.bq.get("init_date_col")
        if not date_col:
            raise RuntimeError("cfg.bq necesita date_col o init_date_col")

        # 2. ArrayStructAggregator (si la tabla lo declara)
        if self.cfg.get("array_struct_features"):
            from core.array_struct_aggregator import ArrayStructAggregator
            log.info("[RL-PIPE] Fitting ArrayStructAggregator sobre df de training...")
            self.aggregator = ArrayStructAggregator().fit(df, self.cfg)
            log.info(f"[RL-PIPE]\n{self.aggregator.describe()}")
            df = self._apply_aggregator(df)

        # 3. Encode (fit sobre un sample del raw post-aggregator)
        log.info("[RL-PIPE] Fitting SchemaEncoder...")
        self.encoder = SchemaEncoder()
        sample_size = min(500_000, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        self.encoder.fit(sample, self.cfg)
        log.info(f"\n{self.encoder.describe()}\n")

        # 4. Split temporal
        df["_date"] = pd.to_datetime(df[date_col]).dt.date
        train_end = self.rl.get("train_end", self.bq.get("train_end", "2025-06-30"))
        val_start = self.rl.get("val_start", train_end)
        val_end   = self.rl.get("val_end",   self.bq.get("test_start", "2025-10-01"))

        train_mask = df["_date"] <= pd.Timestamp(train_end).date()
        val_mask   = ((df["_date"] >  pd.Timestamp(val_start).date()) &
                      (df["_date"] <= pd.Timestamp(val_end).date()))

        train_df = df[train_mask].drop(columns=["_date"])
        val_df   = df[val_mask].drop(columns=["_date"])
        log.info(f"[RL-PIPE] Train: {len(train_df):,} filas | Val: {len(val_df):,} filas")

        # 5. Crear modelo
        model_kwargs = {
            "bottleneck_dim": self.rl.get("bottleneck_dim", 64),
            "encoder_layers": self.rl.get("encoder_layers", [256, 128]),
            "decoder_layers": self.rl.get("decoder_layers", [128, 256]),
            "dropout":        self.rl.get("dropout", 0.3),
        }

        if self.model_type == "vae":
            self.model = TabularVAE.from_encoder(
                self.encoder, beta_kl=self.rl.get("beta_kl", 1.0), **model_kwargs)
        else:
            self.model = TabularAE.from_encoder(self.encoder, **model_kwargs)

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"[RL-PIPE] Modelo: {type(self.model).__name__} ({n_params:,} parámetros)")

        # 6. Entrenar
        trainer = RowLevelTrainer(self.model, self.encoder, self.cfg)
        history = trainer.fit(train_df, val_df)

        # 7. Calibrar baseline estático
        self.scorer = RowLevelScorer(self.model, self.encoder, self.cfg)
        thresholds = self.scorer.calibrate(val_df)

        dt = time.time() - t0
        log.info(f"[RL-PIPE] Training completo en {dt:.0f}s")

        return {
            "history": history,
            "encoder_summary": self.encoder.describe(),
            "thresholds": thresholds,
            "n_params": n_params,
            "train_time_s": dt,
        }

    # ── Persistencia ───────────────────────────────────────────────────────────

    def save(self):
        """Guarda aggregator, encoder, modelo y umbrales en los paths del config."""
        if self.aggregator is not None:
            self.aggregator.save(self.aggregator_path)
        self.encoder.save(self.encoder_path)
        trainer = RowLevelTrainer(self.model, self.encoder, self.cfg)
        trainer.save(self.model_path, encoder_path=None)
        self.scorer.save_thresholds(self.thresholds_path)
        log.info(f"[RL-PIPE] Todo guardado en {self.model_path.parent}")

    def load(self):
        """Carga aggregator (si existe), encoder, modelo y umbrales."""
        # Aggregator: cargar si el cfg lo declara y el archivo existe
        if self.cfg.get("array_struct_features") and self.aggregator_path.exists():
            from core.array_struct_aggregator import ArrayStructAggregator
            self.aggregator = ArrayStructAggregator.load(self.aggregator_path)

        self.encoder = SchemaEncoder.load(self.encoder_path)

        checkpoint = torch.load(self.model_path, map_location="cpu")
        ModelClass = TabularVAE if self.model_type == "vae" else TabularAE
        self.model = ModelClass(
            cat_specs=checkpoint["cat_specs"],
            n_num_features=checkpoint["n_num_features"],
            bottleneck_dim=checkpoint["bottleneck_dim"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.scorer = RowLevelScorer(self.model, self.encoder, self.cfg)
        self.scorer.load_thresholds(self.thresholds_path)

        log.info(f"[RL-PIPE] Cargado desde {self.model_path.parent}")

    # ── Inferencia ─────────────────────────────────────────────────────────────

    def score_single_day(self, day_df: pd.DataFrame, date: str,
                         threshold: Optional[float] = None) -> DayReport:
        """Score un único día. Requiere `load()` previo."""
        if self.scorer is None:
            raise RuntimeError("Llamar load() o train() antes de score_single_day()")
        day_df = self._apply_aggregator(day_df)
        return self.scorer.score_day(day_df, pd.Timestamp(date), threshold=threshold)

    def score_range_dynamic(self, df: pd.DataFrame, date_col: str,
                            n_eval: int) -> Dict[str, object]:
        """Score un rango de días con umbral dinámico sobre la ventana."""
        if self.scorer is None:
            raise RuntimeError("Llamar load() o train() antes de score_range_dynamic()")

        df = self._apply_aggregator(df)

        scores_by_day = self.scorer.score_rows_by_day(df, date_col)
        thr = self.scorer.compute_dynamic_threshold(scores_by_day, n_eval=n_eval)

        days_sorted = sorted(scores_by_day.keys())
        eval_days   = days_sorted[-n_eval:] if n_eval and n_eval < len(days_sorted) else days_sorted

        df_copy = df.copy()
        df_copy["_date"] = pd.to_datetime(df_copy[date_col]).dt.normalize()
        reports = []
        for d in eval_days:
            day_df = df_copy[df_copy["_date"] == d].drop(columns=["_date"])
            reports.append(self.scorer.score_day(day_df, d, threshold=thr))

        return {"reports": reports, "threshold": thr, "scores_by_day": scores_by_day}
