"""
array_struct_aggregator.py — Agregador genérico para columnas ARRAY<STRUCT<...>>
================================================================================

Convierte cada columna ARRAY<STRUCT> declarada en cfg["array_struct_features"]
en un set estable de features:

  - per-row : multi-hot (categorical_tokens) + agregados numéricos (numeric_aggs)
              + lengths estructurales (length_features) → consumido por T4 vía
              SchemaEncoder (las columnas binarias entran como numéricas).

  - per-day : stats categóricas sobre el flatten del día, con el mismo
              contrato que _stats_for_one_column → consumido por T1/T2 vía
              column_attribution. Una pseudo-columna por (array_col, token).

Soporta arbitrariamente profundo via `path: List[str]`. La columna array
declarada (ej. "funcional") es el contenedor exterior; el path es la ruta
dentro del struct (ej. ["crm", "segment_ds"]).

Genérico: cada tabla declara sus campos en cfg. Si la clave no existe,
es no-op completo. Cero cambios en la lógica de las demás tablas.

Estructura del cfg esperada:

    "array_struct_features": {
        "funcional": {                          # nombre de la columna ARRAY
            "categorical_tokens": {
                "service_type":   {"path": ["service_type"]},
                "bundle_type":    {"path": ["bundle_type"], "null_as_token": True},
                "segment_ds":     {"path": ["crm", "segment_ds"]},
                ...
            },
            "numeric_aggs": {
                "tariff_fee":       {"path": ["tariff_fee"],                       "aggs": ["mean", "max"]},
                "imp_without_tax":  {"path": ["crm", "billed", "imp_without_tax"], "aggs": ["mean", "sum"]},
                ...
            },
            "length_features": [
                {"name": "len_outer",           "path": [],                "mode": "count"},
                {"name": "total_len_crm",       "path": ["crm"],           "mode": "count"},
                {"name": "total_len_billed",    "path": ["crm", "billed"], "mode": "count"},
                {"name": "pct_crm_with_billed", "path": ["crm"],           "mode": "pct_nonempty_child", "child": "billed"},
            ],
        }
    }

Vocabularios:
    Por defecto vocab="full" (todos los valores distintos vistos en fit, +
    NULL si null_as_token=True). Para campos de cardinalidad alta, usar
    vocab="top_n", top_n=N, include_other=True.

Uso:
    agg = ArrayStructAggregator().fit(df_train, cfg)
    agg.save(path)
    df_with_features = agg.transform_per_row(df)
    day_stats = agg.compute_day_stats(day_df)   # lista de dicts para column_attribution
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NULL_TOKEN  = "__NULL__"
_OTHER_TOKEN = "__OTHER__"

_DAY_STAT_COLS = ["pct_null", "pct_empty", "pct_unknown",
                  "entropy_norm", "hhi", "top1_share", "n_cats", "n_rows"]


# ── Helpers de navegación estructural ─────────────────────────────────────────

def _is_scalar(x: Any) -> bool:
    """True si x es escalar (no contenedor)."""
    return np.isscalar(x) or x is None or isinstance(x, (str, bytes, np.datetime64))


def _is_null(x: Any) -> bool:
    """pd.isna seguro frente a no-escalares."""
    if x is None:
        return True
    if _is_scalar(x):
        try:
            return bool(pd.isna(x))
        except (TypeError, ValueError):
            return False
    return False


def _safe_iter(x: Any) -> list:
    """Devuelve x como lista iterable, robusto frente a None/NaN/escalares/dict/str."""
    if x is None:
        return []
    if _is_scalar(x):
        return []
    if isinstance(x, (str, bytes, dict)):
        return []
    try:
        return list(x)
    except Exception:
        return []


def _safe_len(x: Any) -> int:
    if x is None or _is_scalar(x):
        return 0
    if isinstance(x, (str, bytes, dict)):
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def _get_field(elem: Any, field: str) -> Any:
    """Acceso a campo de STRUCT — dict, namedtuple, pyarrow Struct o atributo."""
    if elem is None:
        return None
    if isinstance(elem, dict):
        return elem.get(field)
    return getattr(elem, field, None)


def _walk_struct(elem: Any, path: List[str]) -> list:
    """Recorre recursivamente desde un STRUCT siguiendo el path.

    Devuelve lista plana de valores terminales. Si en algún nivel encuentra
    una lista, expande sobre todos sus elementos.
    """
    if elem is None:
        return []
    if not path:
        return [elem]
    head, tail = path[0], path[1:]
    val = _get_field(elem, head)
    if val is None:
        return []
    items = _safe_iter(val)
    if items:                                         # val es contenedor → expandir
        out = []
        for sub in items:
            out.extend(_walk_struct(sub, tail))
        return out
    if not tail:                                      # val es escalar terminal
        return [val]
    return _walk_struct(val, tail)                    # val es struct → seguir bajando


def _flatten_array_terminals(arr: Any, path: List[str]) -> list:
    """Recibe el valor de una columna ARRAY<STRUCT> y devuelve la lista plana
    de valores terminales siguiendo el path dentro de cada struct."""
    items = _safe_iter(arr)
    if not items:
        return []
    if not path:
        return items
    out = []
    for elem in items:
        out.extend(_walk_struct(elem, path))
    return out


def _collect_subarray_lengths(elem: Any, path: List[str], out: list) -> None:
    """Recorre hasta el penúltimo nivel y emite len(último_array) por struct
    final encontrado. Usado por length_features con mode='count' sobre paths
    que apuntan a sub-arrays."""
    if elem is None or not path:
        return
    head, tail = path[0], path[1:]
    val = _get_field(elem, head)
    if val is None:
        if not tail:
            out.append(0)
        return
    if not tail:
        out.append(_safe_len(val))                    # último nivel: longitud del array
        return
    items = _safe_iter(val)
    if items:
        for sub in items:
            _collect_subarray_lengths(sub, tail, out)
    else:
        _collect_subarray_lengths(val, tail, out)


def _array_subarray_lengths(arr: Any, path: List[str]) -> list:
    """Recibe valor de columna ARRAY<STRUCT> y devuelve lista de longitudes
    del sub-array en `path` por cada struct del array exterior (y recursivo
    si path atraviesa más arrays)."""
    items = _safe_iter(arr)
    if not items:
        return []
    if not path:
        return [_safe_len(arr)]
    out: list = []
    for elem in items:
        _collect_subarray_lengths(elem, path, out)
    return out


# ── Stats helpers (replican column_attribution para coherencia) ──────────────

def _entropy_norm(s: pd.Series) -> float:
    p = s.value_counts(normalize=True).values
    if len(p) <= 1:
        return 0.0
    raw = float(-np.sum(p * np.log(p + 1e-12)))
    return raw / float(np.log(len(p)))


def _hhi(s: pd.Series) -> float:
    if s.empty:
        return 1.0
    return float((s.value_counts(normalize=True).values ** 2).sum())


# ── Clase principal ──────────────────────────────────────────────────────────

class ArrayStructAggregator:
    """Convierte columnas ARRAY<STRUCT> en features estables.

    Estado fitted:
        cfg_snapshot           : config usada en fit (subdict array_struct_features)
        vocabs                 : {array_col: {token_name: [vocab]}}
        feature_names_per_row  : columnas emitidas por transform_per_row, en orden
        pseudo_column_names    : pseudo-cols emitidas por compute_day_stats
    """

    def __init__(self):
        self.cfg_snapshot: Dict[str, Any] = {}
        self.vocabs: Dict[str, Dict[str, List[str]]] = {}
        self.feature_names_per_row: List[str] = []
        self.pseudo_column_names: List[str] = []
        self._fitted = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, cfg: dict) -> "ArrayStructAggregator":
        asf = cfg.get("array_struct_features", {}) or {}
        self.cfg_snapshot = dict(asf)
        self.vocabs = {}
        feature_names: List[str] = []
        pseudo_cols: List[str] = []

        if not asf:
            log.info("[AGG] cfg.array_struct_features vacío → no-op")
            self._fitted = True
            return self

        for array_col, col_cfg in asf.items():
            if array_col not in df.columns:
                log.warning(f"[AGG] Columna '{array_col}' no presente en df, se omite")
                continue

            self.vocabs[array_col] = {}

            # ── categorical_tokens: construir vocab ──
            for tok_name, tok_cfg in (col_cfg.get("categorical_tokens") or {}).items():
                path = tok_cfg["path"]
                vocab_mode    = tok_cfg.get("vocab", "full")
                null_as_token = tok_cfg.get("null_as_token", False)
                include_other = tok_cfg.get("include_other", vocab_mode == "top_n")

                counter: Counter = Counter()
                for arr in df[array_col].values:
                    vals = _flatten_array_terminals(arr, path)
                    for v in vals:
                        if _is_null(v):
                            if null_as_token:
                                counter[_NULL_TOKEN] += 1
                            continue
                        counter[str(v)] += 1

                if vocab_mode == "full":
                    vocab = sorted(k for k in counter if k != _OTHER_TOKEN)
                else:
                    top_n = int(tok_cfg.get("top_n", 30))
                    vocab = [k for k, _ in counter.most_common(top_n) if k != _OTHER_TOKEN]
                if include_other:
                    vocab.append(_OTHER_TOKEN)

                self.vocabs[array_col][tok_name] = vocab

                for v in vocab:
                    feature_names.append(f"{array_col}__{tok_name}__{v}")
                pseudo_cols.append(f"{array_col}__{tok_name}")

                log.info(f"[AGG] {array_col}.{tok_name}: vocab={len(vocab)} "
                         f"(modo={vocab_mode}, null_as_token={null_as_token})")

            # ── numeric_aggs: solo registrar nombres ──
            for num_name, num_cfg in (col_cfg.get("numeric_aggs") or {}).items():
                for agg in num_cfg.get("aggs", ["mean"]):
                    feature_names.append(f"{array_col}__{num_name}__{agg}")

            # ── length_features ──
            for len_cfg in (col_cfg.get("length_features") or []):
                feature_names.append(f"{array_col}__{len_cfg['name']}")

        self.feature_names_per_row = feature_names
        self.pseudo_column_names   = pseudo_cols
        self._fitted = True
        log.info(f"[AGG] Fitted: {len(feature_names)} features per-row, "
                 f"{len(pseudo_cols)} pseudo-cols")
        return self

    # ── transform_per_row ────────────────────────────────────────────────────

    def transform_per_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade columnas de features per-row al df. Devuelve un nuevo df.

        Si el agregador no fue fitted o cfg_snapshot está vacío, devuelve el
        df sin cambios.
        """
        assert self._fitted, "fit() antes de transform_per_row()"
        if not self.cfg_snapshot:
            return df

        result = df.copy()

        for array_col, col_cfg in self.cfg_snapshot.items():
            if array_col not in df.columns:
                # Llenar con ceros las features esperadas (robustez en inferencia)
                for fname in self.feature_names_per_row:
                    if fname.startswith(f"{array_col}__"):
                        result[fname] = 0.0
                continue

            arrays = df[array_col].values
            n = len(arrays)

            # 1. categorical_tokens → multi-hot
            for tok_name, tok_cfg in (col_cfg.get("categorical_tokens") or {}).items():
                path = tok_cfg["path"]
                null_as_token = tok_cfg.get("null_as_token", False)
                vocab     = self.vocabs[array_col][tok_name]
                vocab_set = set(vocab)
                has_other = _OTHER_TOKEN in vocab_set

                hot = {f"{array_col}__{tok_name}__{v}": np.zeros(n, dtype=np.float32)
                       for v in vocab}

                for i, arr in enumerate(arrays):
                    vals = _flatten_array_terminals(arr, path)
                    if not vals:
                        continue
                    for v in vals:
                        if _is_null(v):
                            if null_as_token and _NULL_TOKEN in vocab_set:
                                hot[f"{array_col}__{tok_name}__{_NULL_TOKEN}"][i] = 1.0
                            continue
                        key = str(v)
                        col_name = f"{array_col}__{tok_name}__{key}"
                        if col_name in hot:
                            hot[col_name][i] = 1.0
                        elif has_other:
                            hot[f"{array_col}__{tok_name}__{_OTHER_TOKEN}"][i] = 1.0

                for col_name, vec in hot.items():
                    result[col_name] = vec

            # 2. numeric_aggs
            for num_name, num_cfg in (col_cfg.get("numeric_aggs") or {}).items():
                path = num_cfg["path"]
                aggs = num_cfg.get("aggs", ["mean"])

                vecs = {agg: np.full(n, np.nan, dtype=np.float32) for agg in aggs}

                for i, arr in enumerate(arrays):
                    vals = _flatten_array_terminals(arr, path)
                    clean = []
                    for v in vals:
                        if _is_null(v):
                            continue
                        try:
                            clean.append(float(v))
                        except (TypeError, ValueError):
                            continue
                    if not clean:
                        continue
                    a = np.asarray(clean, dtype=np.float64)
                    for agg in aggs:
                        if   agg == "mean": vecs[agg][i] = float(a.mean())
                        elif agg == "max":  vecs[agg][i] = float(a.max())
                        elif agg == "min":  vecs[agg][i] = float(a.min())
                        elif agg == "sum":  vecs[agg][i] = float(a.sum())
                        elif agg == "std":  vecs[agg][i] = float(a.std())
                        elif agg == "count":vecs[agg][i] = float(len(a))

                for agg, vec in vecs.items():
                    result[f"{array_col}__{num_name}__{agg}"] = vec

            # 3. length_features
            for len_cfg in (col_cfg.get("length_features") or []):
                name = len_cfg["name"]
                path = len_cfg["path"]
                mode = len_cfg.get("mode", "count")
                col_name = f"{array_col}__{name}"

                vec = np.zeros(n, dtype=np.float32)

                if mode == "count":
                    for i, arr in enumerate(arrays):
                        if not path:
                            vec[i] = float(_safe_len(arr))
                        else:
                            sublens = _array_subarray_lengths(arr, path)
                            vec[i] = float(sum(sublens))

                elif mode == "pct_nonempty_child":
                    child = len_cfg["child"]
                    for i, arr in enumerate(arrays):
                        items = _safe_iter(arr)
                        if not items:
                            continue
                        parents = []
                        for elem in items:
                            parents.extend(_walk_struct(elem, path))
                        if not parents:
                            continue
                        nonempty = sum(1 for p in parents
                                       if _safe_len(_get_field(p, child)) > 0)
                        vec[i] = float(nonempty) / float(len(parents))

                else:
                    log.warning(f"[AGG] length_features mode desconocido: {mode}")

                result[col_name] = vec

        return result

    # ── compute_day_stats (consumido por column_attribution) ─────────────────

    def compute_day_stats(self, day_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Para cada pseudo-columna categórica, stats sobre el flatten del día.

        Devuelve lista de dicts con claves:
            'column' + _DAY_STAT_COLS
        listos para concatenar al output de compute_per_column_stats.
        """
        assert self._fitted, "fit() antes de compute_day_stats()"
        out: List[Dict[str, Any]] = []
        n_rows = len(day_df)

        for array_col, col_cfg in self.cfg_snapshot.items():
            if array_col not in day_df.columns:
                continue

            for tok_name, tok_cfg in (col_cfg.get("categorical_tokens") or {}).items():
                path = tok_cfg["path"]

                all_tokens: List[str] = []
                n_rows_with_null  = 0
                n_rows_with_empty = 0

                for arr in day_df[array_col].values:
                    items = _safe_iter(arr)
                    if not items:
                        n_rows_with_empty += 1
                        continue
                    vals = _flatten_array_terminals(arr, path)
                    if not vals:
                        n_rows_with_null += 1
                        continue
                    row_had_value = False
                    for v in vals:
                        if _is_null(v):
                            continue
                        all_tokens.append(str(v))
                        row_had_value = True
                    if not row_had_value:
                        n_rows_with_null += 1

                pseudo_col = f"{array_col}__{tok_name}"

                if n_rows == 0 or not all_tokens:
                    out.append({
                        "column":       pseudo_col,
                        "pct_null":     (float(n_rows_with_null) / float(n_rows)) if n_rows else 0.0,
                        "pct_empty":    (float(n_rows_with_empty) / float(n_rows)) if n_rows else 0.0,
                        "pct_unknown":  0.0,
                        "entropy_norm": np.nan,
                        "hhi":          np.nan,
                        "top1_share":   np.nan,
                        "n_cats":       np.nan,
                        "n_rows":       float(n_rows),
                    })
                    continue

                ts = pd.Series(all_tokens)
                vc = ts.value_counts(normalize=True)
                out.append({
                    "column":       pseudo_col,
                    "pct_null":     float(n_rows_with_null)  / float(n_rows),
                    "pct_empty":    float(n_rows_with_empty) / float(n_rows),
                    "pct_unknown":  0.0,
                    "entropy_norm": _entropy_norm(ts),
                    "hhi":          _hhi(ts),
                    "top1_share":   float(vc.iloc[0]),
                    "n_cats":       float(ts.nunique()),
                    "n_rows":       float(n_rows),
                })

        return out

    # ── Persistencia ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cfg_snapshot":          self.cfg_snapshot,
            "vocabs":                self.vocabs,
            "feature_names_per_row": self.feature_names_per_row,
            "pseudo_column_names":   self.pseudo_column_names,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"[AGG] Guardado → {path}")

    @classmethod
    def load(cls, path: Path) -> "ArrayStructAggregator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        a = cls()
        a.cfg_snapshot          = state["cfg_snapshot"]
        a.vocabs                = state["vocabs"]
        a.feature_names_per_row = state["feature_names_per_row"]
        a.pseudo_column_names   = state["pseudo_column_names"]
        a._fitted = True
        log.info(f"[AGG] Cargado desde {path}: "
                 f"{len(a.feature_names_per_row)} features per-row")
        return a

    # ── Inspección ───────────────────────────────────────────────────────────

    def list_array_cols(self) -> List[str]:
        return list(self.cfg_snapshot.keys())

    def describe(self) -> str:
        lines = [f"ArrayStructAggregator: {len(self.feature_names_per_row)} features per-row, "
                 f"{len(self.pseudo_column_names)} pseudo-columnas day-stats"]
        for array_col in self.cfg_snapshot:
            lines.append(f"  {array_col}:")
            for tok_name, vocab in self.vocabs.get(array_col, {}).items():
                lines.append(f"    {tok_name:<20} vocab={len(vocab):>4} → "
                             f"{vocab[:4]}{'...' if len(vocab) > 4 else ''}")
        return "\n".join(lines)
