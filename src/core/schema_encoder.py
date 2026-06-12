"""
schema_encoder.py — Encoding genérico de filas tabulares para autoencoder.

Auto-descubre el esquema de un DataFrame (cat/num/skip), construye vocabularios
para categóricas, ajusta un RobustScaler para numéricas y transforma filas a
tensores. Agnóstico a la tabla: toda la personalización va vía cfg.

    encoder = SchemaEncoder().fit(df_sample, cfg)
    batch   = encoder.transform(day_df)
    encoder.save(path); encoder = SchemaEncoder.load(path)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

MAX_CARDINALITY = 500          # Máximo de categorías antes de truncar a top-N
MIN_IQR = 0.01                 # Floor del scaler: evita amplificación en cols casi-constantes
CLIP_RANGE = (-5.0, 5.0)       # Clipping post-escalado
NULL_FRACTION_SKIP = 0.95      # Excluir columnas con >95% nulos
UNK_TOKEN = "<UNK>"            # Índice 0: valor desconocido
PAD_TOKEN = "<PAD>"            # Índice 1: nulo

_SKIP_DTYPES = {"datetime64", "timedelta64"}

class PrivacyGuardError(RuntimeError):
    """Se eleva cuando SchemaEncoder detecta columnas PII en el DataFrame
    de entrada que no están excluidas en cfg.

    Conforme a RGPD art. 25 (Data Protection by Design and by Default) y
    AI Act art. 10 (Data Governance for High-Risk AI Systems): el sistema
    debe rechazar activamente datos personales no autorizados, no confiar
    en la disciplina del operador.
    """
    pass

@dataclass
class EncodedBatch:
    """Resultado de encoder.transform()."""
    cat_tensors: Dict[str, torch.LongTensor]   # col → [batch] índices
    num_tensor: torch.FloatTensor               # [batch, n_num_cols]
    cat_col_names: List[str]
    num_col_names: List[str]
    n_rows: int


@dataclass
class CatColumnMeta:
    """Vocabulario y dim de embedding de una columna categórica."""
    col_name: str
    vocab: Dict[str, int]       # valor → índice (0=UNK, 1=PAD, 2..N=valores)
    n_categories: int           # len(vocab), incluye UNK+PAD
    embedding_dim: int


class SchemaEncoder:
    """Encoder genérico de filas tabulares: auto-discovery → vocabs → scaler → tensores."""

    def __init__(self):
        self.cat_metas: Dict[str, CatColumnMeta] = {}
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.excluded_cols: Set[str] = set()
        self.scaler: Optional[RobustScaler] = None
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, cfg: dict) -> "SchemaEncoder":
        """Descubre esquema, construye vocabs y ajusta scaler desde un sample representativo."""
        bq = cfg.get("bq", {})
        rl = cfg.get("row_level", {})

        self.excluded_cols = self._build_exclusion_set(bq, rl)
        log.info(f"[ENCODER] Exclusiones explícitas: {sorted(self.excluded_cols)}")
        # Hard guard de privacidad — RGPD 
        self._check_pii_guard(df, cfg, self.excluded_cols)

        candidates = [c for c in df.columns if c not in self.excluded_cols]
        cat_candidates, num_candidates, auto_skipped = [], [], []

        for col in candidates:
            series = df[col]
            if any(t in str(series.dtype) for t in _SKIP_DTYPES):
                auto_skipped.append((col, "datetime/timedelta"))
                continue
            null_frac = series.isna().mean()
            if null_frac > NULL_FRACTION_SKIP:
                auto_skipped.append((col, f"null_frac={null_frac:.2%}"))
                continue
            n_unique = series.nunique(dropna=True)
            if n_unique <= 1:
                auto_skipped.append((col, f"constant (nunique={n_unique})"))
                continue

            if self._is_categorical(series, n_unique, rl):
                cat_candidates.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                num_candidates.append(col)
            else:
                auto_skipped.append((col, f"dtype={series.dtype}, no cat/num"))

        if auto_skipped:
            log.info(f"[ENCODER] Auto-excluidas: {auto_skipped}")

        self.cat_cols = sorted(cat_candidates)
        self.cat_metas = {col: self._build_vocab(df[col], col, rl) for col in self.cat_cols}

        self.num_cols = sorted(num_candidates)
        if self.num_cols:
            self.scaler = RobustScaler()
            num_data = df[self.num_cols].copy()
            for c in self.num_cols:                       # el scaler no admite NaN
                med = num_data[c].median()
                num_data[c] = num_data[c].fillna(med if pd.notna(med) else 0.0)
            self.scaler.fit(num_data.values)
            self.scaler.scale_ = np.maximum(self.scaler.scale_, MIN_IQR)

        self._fitted = True
        total_emb_dim = sum(m.embedding_dim for m in self.cat_metas.values())
        log.info(f"[ENCODER] Fitted: {len(self.cat_cols)} cat (emb_total={total_emb_dim}), "
                 f"{len(self.num_cols)} num, {len(self.excluded_cols) + len(auto_skipped)} excluidas")
        return self

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> EncodedBatch:
        """Transforma filas raw a tensores (categóricas→índices, numéricas→float escalado)."""
        assert self._fitted, "Llamar a fit() antes de transform()"
        n = len(df)

        cat_tensors = {}
        for col in self.cat_cols:
            meta = self.cat_metas[col]
            if col not in df.columns:                     # columna ausente → todo PAD
                cat_tensors[col] = torch.ones(n, dtype=torch.long) * meta.vocab[PAD_TOKEN]
                continue
            series = df[col].astype(str).fillna(PAD_TOKEN)
            indices = series.map(lambda v: meta.vocab.get(v, meta.vocab[UNK_TOKEN]))
            cat_tensors[col] = torch.tensor(indices.values, dtype=torch.long)

        if self.num_cols:
            num_data = df[self.num_cols].copy() if all(c in df.columns for c in self.num_cols) else pd.DataFrame()
            if num_data.empty:
                num_tensor = torch.zeros(n, len(self.num_cols))
            else:
                for c in self.num_cols:
                    med = num_data[c].median()
                    num_data[c] = num_data[c].fillna(med if pd.notna(med) else 0.0)
                scaled = np.clip(self.scaler.transform(num_data.values), *CLIP_RANGE)
                num_tensor = torch.tensor(scaled, dtype=torch.float32)
        else:
            num_tensor = torch.zeros(n, 0)

        return EncodedBatch(cat_tensors, num_tensor, self.cat_cols, self.num_cols, n)

    # ── Dimensiones para el modelo ─────────────────────────────────────────────

    @property
    def total_input_dim(self) -> int:
        return sum(m.embedding_dim for m in self.cat_metas.values()) + len(self.num_cols)

    @property
    def cat_specs(self) -> List[Tuple[str, int, int]]:
        """(col_name, n_categories, embedding_dim) para construir embeddings."""
        return [(m.col_name, m.n_categories, m.embedding_dim) for m in self.cat_metas.values()]

    @property
    def n_num_features(self) -> int:
        return len(self.num_cols)

    # ── Persistencia ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cat_metas": {col: vars(m) for col, m in self.cat_metas.items()},
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "excluded_cols": list(self.excluded_cols),
            "scaler": self.scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"[ENCODER] Guardado → {path}")

    @classmethod
    def load(cls, path: Path) -> "SchemaEncoder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        enc = cls()
        enc.num_cols = state["num_cols"]
        enc.cat_cols = state["cat_cols"]
        enc.excluded_cols = set(state["excluded_cols"])
        enc.scaler = state["scaler"]
        enc.cat_metas = {col: CatColumnMeta(**meta) for col, meta in state["cat_metas"].items()}
        enc._fitted = True
        return enc

    def describe(self) -> str:
        lines = [f"SchemaEncoder: {len(self.cat_cols)} cat + {len(self.num_cols)} num "
                 f"= {self.total_input_dim} dims de entrada", "", "Categóricas:"]
        for col in self.cat_cols:
            m = self.cat_metas[col]
            lines.append(f"  {col:<35} vocab={m.n_categories:>4}  emb_dim={m.embedding_dim:>3}")
        lines += ["", f"Numéricas: {', '.join(self.num_cols[:10])}"]
        if len(self.num_cols) > 10:
            lines.append(f"  ... y {len(self.num_cols) - 10} más")
        return "\n".join(lines)

    # ── Internos ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_exclusion_set(bq_cfg: dict, rl_cfg: dict = None) -> Set[str]:
        """Columnas a excluir: fechas, identidad/privacidad y exclusiones manuales."""
        excluded = set()
        for key in ("date_col", "init_date_col", "end_date_col", "snapshot_col",
                    "sample_hash_col", "phone_col", "pk_col"):
            v = bq_cfg.get(key)
            if v:
                excluded.add(v)
        excluded.update((rl_cfg or {}).get("exclude_cols", []) or [])
        return excluded


    @staticmethod
    def _check_pii_guard(df: pd.DataFrame, cfg: dict, excluded: Set[str]) -> None:
        """Hard guard: rechaza fit() si hay PII canónica sin excluir.

        Resuelve la tabla desde cfg["paths"]["tranad_model"].parent.name y
        cruza df.columns contra audit_privacy.PII_COLUMNS[table]. Cualquier
        intersección no excluida → PrivacyGuardError.
        """
        paths = cfg.get("paths", {}) or {}
        tranad_path = paths.get("tranad_model")
        if tranad_path is None:
            log.warning("[PRIVACY] table_key no resoluble desde cfg.paths — guard inactivo")
            return
        table_key = Path(tranad_path).parent.name

        try:
            from audit_privacy import PII_COLUMNS
        except ImportError:
            log.warning("[PRIVACY] audit_privacy no importable — guard inactivo")
            return

        pii_set = set(PII_COLUMNS.get(table_key, []))
        if not pii_set:
            log.info(f"[PRIVACY] Tabla '{table_key}' sin PII canónica declarada")
            return

        leaked = (set(df.columns) & pii_set) - excluded
        if leaked:
            raise PrivacyGuardError(
                f"\n[PRIVACY GUARD] PII detectada sin exclusión configurada para "
                f"tabla '{table_key}':\n"
                f"    {sorted(leaked)}\n\n"
                f"Añádelas a cfg['bq']['exclude_cols'] y cfg['row_level']['exclude_cols']\n"
                f"en config.py antes de fit().\n"
                f"Referencia normativa: RGPD art. 25, AI Act art. 10.\n"
            )

        log.info(
            f"[PRIVACY] Guard OK — '{table_key}': "
            f"{len(pii_set & excluded)}/{len(pii_set)} PII canónicas excluidas"
        )


    
    @staticmethod
    def _is_categorical(series: pd.Series, n_unique: int, rl_cfg: dict) -> bool:
        max_card = rl_cfg.get("max_cardinality", MAX_CARDINALITY)
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            return n_unique <= max_card
        if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_numeric_dtype(series) and n_unique <= 20:   # numérica de baja cardinalidad
            return True
        return False

    @staticmethod
    def _build_vocab(series: pd.Series, col_name: str, rl_cfg: dict) -> CatColumnMeta:
        max_card = rl_cfg.get("max_cardinality", MAX_CARDINALITY)
        vc = series.dropna().astype(str).value_counts()
        if len(vc) > max_card:
            log.info(f"[ENCODER] {col_name}: truncado de {len(vc)} a {max_card} categorías")
            top_vals = vc.head(max_card).index.tolist()
        else:
            top_vals = vc.index.tolist()

        vocab = {UNK_TOKEN: 0, PAD_TOKEN: 1}              # resto: 2..N+1
        for i, val in enumerate(top_vals):
            vocab[val] = i + 2

        n_categories = len(vocab)
        embedding_dim = max(min(50, (n_categories + 1) // 2), 2)
        return CatColumnMeta(col_name, vocab, n_categories, embedding_dim)
