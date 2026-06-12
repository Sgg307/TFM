"""
fake_error.py — Inyector de incidentes a nivel RAW
===================================================
Crea una tabla BigQuery `<tabla>_fake_error`: una copia de una ventana de
datos SANOS donde UN día concreto ha sido corrompido a nivel de fila. Como
es una tabla BQ real, se pasa por el pipeline de producción tal cual
(runner.monitor / runner.row_level), ejercitando los CUATRO tiers — no solo
TranAD como el stress test en espacio escalado.

Filosofía:
  - La corrupción es sobre filas reales (NULL de verdad, valor que desaparece,
    volumen submuestreado, correlación rota), no sobre estadísticos agregados.
  - Cada tipo de corrupción mapea a un tier que DEBE cazarla → tabla de consenso.
  - La ventana de contexto (días previos) queda sana → lag-7, baselines DOW y
    umbrales dinámicos funcionan igual que en producción.

Uso (desde el notebook, vía runner.inject — ver patch en dev_tools):
    from fake_error import make_fake_error_table
    info = make_fake_error_table(
        cfg, target_date="2025-10-15",
        injections=[("null_spike", {"col": "brand_donor", "pct": 0.4})],
    )

Tipos de inyección:
  ("null_spike",       {"col", "pct"})           → NULL en pct% de filas de col
  ("category_drop",    {"col", "value"=None})    → un valor categórico desaparece
  ("category_inject",  {"col", "pct", "value"})  → valor nuevo nunca visto
  ("volume_drop",      {"keep"})                 → se queda solo keep% de filas
  ("correlation_break",{"col"})                  → baraja col (marginal intacta)

IMPORTANTE sobre muestreo:
  La tabla _fake_error se escribe ya con el muestreo del cfg (10% por hash),
  igual que ve el pipeline al entrenar. Por eso, al puntuarla después, hay que
  poner sample_pct=100 (no re-muestrear). runner.inject() lo hace por ti.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_INJECTED_VALUE = "__INJECTED_NEW__"

# Qué tier DEBE cazar cada tipo (para la tabla de consenso del TFM).
# Clave: una corrupción en UNA columna se diluye en el vector agregado de
# TranAD (T3) → la cazan los detectores a nivel columna/fila (T1/T4), no T3.
# T3 solo entra cuando el efecto es GLOBAL (volumen, o nulos en muchas cols).
_EXPECTED_TIERS = {
    "null_spike":        ["T1 (columna)", "T4 (fila)"],   # 1 col → se diluye en T3
    "category_drop":     ["T1 (columna)", "T2 (PCA)", "T4 (fila)"],
    "category_inject":   ["T1 (columna)", "T2 (PCA)", "T4 (fila)"],
    "volume_drop":       ["T1 (Volumen)", "T3 (Volume)", "(T4 ciego — esperado)"],
    "correlation_break": ["T2 (PCA)"],                     # marginal intacta → solo PCA
}


# ─────────────────────────────────────────────────────────────────────────────
# Corrupciones (operan in-place sobre el subconjunto del día objetivo)
# ─────────────────────────────────────────────────────────────────────────────

def _corrupt_null_spike(day: pd.DataFrame, col: str, pct: float, rng) -> str:
    n = len(day)
    k = int(n * pct)
    if col not in day.columns or k == 0:
        return f"null_spike SALTADO (col '{col}' ausente o pct=0)"
    idx = rng.choice(day.index, size=k, replace=False)
    day.loc[idx, col] = None
    return f"null_spike: '{col}' → NULL en {k:,}/{n:,} filas ({pct:.0%})"


def _corrupt_category_drop(day: pd.DataFrame, col: str, value, rng) -> str:
    if col not in day.columns:
        return f"category_drop SALTADO (col '{col}' ausente)"
    vc = day[col].value_counts(dropna=True)
    if vc.empty:
        return f"category_drop SALTADO (col '{col}' sin valores)"
    if value is None:
        value = vc.index[0]                       # el valor más frecuente
    # reasignar las filas de ese valor al SIGUIENTE más frecuente
    replacement = next((v for v in vc.index if v != value), value)
    mask = day[col] == value
    n_aff = int(mask.sum())
    day.loc[mask, col] = replacement
    return (f"category_drop: '{col}'='{value}' desaparece "
            f"({n_aff:,} filas → '{replacement}'); volumen intacto")


def _corrupt_category_inject(day: pd.DataFrame, col: str, pct: float,
                             value: str, rng) -> str:
    n = len(day)
    k = int(n * pct)
    if col not in day.columns or k == 0:
        return f"category_inject SALTADO (col '{col}' ausente o pct=0)"
    idx = rng.choice(day.index, size=k, replace=False)
    day.loc[idx, col] = value
    return f"category_inject: '{col}' → '{value}' en {k:,}/{n:,} filas ({pct:.0%})"


def _corrupt_correlation_break(day: pd.DataFrame, col: str, rng) -> str:
    if col not in day.columns:
        return f"correlation_break SALTADO (col '{col}' ausente)"
    shuffled = day[col].sample(frac=1.0, random_state=int(rng.integers(1e9))).values
    day[col] = shuffled
    return (f"correlation_break: '{col}' barajada (distribución marginal "
            f"intacta, relación con otras columnas rota)")


# ─────────────────────────────────────────────────────────────────────────────
# Constructor de la tabla fake
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeErrorInfo:
    dest_table: str
    dest_fqn: str
    target_date: str
    window_start: str
    window_end: str
    n_rows_total: int
    n_rows_target_before: int
    n_rows_target_after: int
    manifest: List[Dict[str, Any]] = field(default_factory=list)
    expected_tiers: List[str] = field(default_factory=list)


def _build_window_query(bq: dict, start: str, end: str) -> str:
    fqn = f"`{bq['project_id']}.{bq['dataset']}.{bq['table']}`"
    date_col = bq["date_col"]
    h = bq.get("sample_hash_col")
    pct = bq.get("sample_pct", 100)
    where = [f"{date_col} BETWEEN '{start}' AND '{end}'"]
    if h and pct < 100:
        where.append(
            f"MOD(ABS(FARM_FINGERPRINT(CAST({h} AS STRING))), 100) < {pct}"
        )
    return f"SELECT * FROM {fqn} WHERE {' AND '.join(where)}"


def make_fake_error_table(
    cfg: dict,
    target_date: Optional[str] = None,
    injections: Optional[List[Tuple[str, dict]]] = None,
    window_days: int = 50,
    dest_table: Optional[str] = None,
    dest_suffix: str = "fake_error",
    upload: bool = True,
    seed: int = 42,
) -> FakeErrorInfo:
    """
    Crea una tabla _fake_error en BQ con un día corrompido.

    Args:
        cfg:          Config de la tabla (debe tener bq.date_col != None).
        target_date:  Día a corromper ("YYYY-MM-DD"). None = MAX(date) del origen.
        injections:   Lista de (tipo, params). Si None → un null_spike de demo.
        window_days:  Días de contexto sano antes/incluyendo el target.
        dest_table:   Nombre destino. None = "<tabla>_<dest_suffix>".
        upload:       False = dry-run, no escribe en BQ (devuelve solo el plan).
        seed:         Semilla para reproducibilidad.

    Returns: FakeErrorInfo
    """
    from google.cloud import bigquery

    bq = cfg["bq"]
    date_col = bq.get("date_col")
    if date_col is None:
        raise ValueError(
            "make_fake_error_table requiere bq.date_col != None. "
            "Las tablas snapshot (discounts) necesitan inyección por reconstrucción "
            "de activos — no soportado todavía."
        )

    client = bigquery.Client(project=bq["project_id"])

    # ── Resolver target y ventana ────────────────────────────────────────
    if target_date is None:
        fqn = f"`{bq['project_id']}.{bq['dataset']}.{bq['table']}`"
        row = client.query(
            f"SELECT DATE(MAX({date_col})) AS d FROM {fqn}"
        ).to_dataframe()
        target_date = row["d"].iloc[0].strftime("%Y-%m-%d")

    target_ts = pd.Timestamp(target_date)
    start_ts = target_ts - pd.Timedelta(days=window_days - 1)
    start, end = start_ts.strftime("%Y-%m-%d"), target_ts.strftime("%Y-%m-%d")

    if dest_table is None:
        dest_table = f"{bq['table']}_{dest_suffix}"
    dest_fqn = f"{bq['project_id']}.{bq['dataset']}.{dest_table}"

    # ── Descargar ventana sana ───────────────────────────────────────────
    log.info(f"[FAKE] Descargando ventana sana {start} → {end} "
             f"({window_days}d, {bq.get('sample_pct',100)}% sample)...")
    df = client.query(_build_window_query(bq, start, end)).to_dataframe(
        progress_bar_type="tqdm"
    )
    log.info(f"[FAKE] {len(df):,} filas descargadas")

    df["_d"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    target_mask = df["_d"] == target_date
    n_before = int(target_mask.sum())
    if n_before == 0:
        raise ValueError(f"El día {target_date} no tiene filas en la ventana.")

    # ── Aplicar inyecciones al día objetivo ──────────────────────────────
    if injections is None:
        first_cat = (bq.get("category_cols") or [c for c in df.columns
                     if df[c].dtype == object and c != "_d"])[0]
        injections = [("null_spike", {"col": first_cat, "pct": 0.4})]

    rng = np.random.default_rng(seed)
    day = df[target_mask].copy()
    manifest, expected = [], []

    for kind, params in injections:
        if kind == "null_spike":
            desc = _corrupt_null_spike(day, params["col"], params.get("pct", 0.3), rng)
        elif kind == "category_drop":
            desc = _corrupt_category_drop(day, params["col"], params.get("value"), rng)
        elif kind == "category_inject":
            desc = _corrupt_category_inject(
                day, params["col"], params.get("pct", 0.2),
                params.get("value", _INJECTED_VALUE), rng)
        elif kind == "volume_drop":
            keep = params.get("keep", 0.4)
            day = day.sample(frac=keep, random_state=seed)
            desc = f"volume_drop: día reducido a {keep:.0%} ({len(day):,} filas)"
        elif kind == "correlation_break":
            desc = _corrupt_correlation_break(day, params["col"], rng)
        else:
            desc = f"⚠️ tipo desconocido: {kind}"
        manifest.append({"kind": kind, "params": params, "effect": desc})
        expected += _EXPECTED_TIERS.get(kind, [])
        log.info(f"[FAKE] {desc}")

    n_after = len(day)

    # Reensamblar: contexto sano + día corrompido
    healthy = df[~target_mask].drop(columns=["_d"])
    day = day.drop(columns=["_d"])
    out = pd.concat([healthy, day], ignore_index=True)

    info = FakeErrorInfo(
        dest_table=dest_table, dest_fqn=dest_fqn, target_date=target_date,
        window_start=start, window_end=end, n_rows_total=len(out),
        n_rows_target_before=n_before, n_rows_target_after=n_after,
        manifest=manifest, expected_tiers=sorted(set(expected)),
    )

    # ── Subir a BQ ───────────────────────────────────────────────────────
    if upload:
        log.info(f"[FAKE] Subiendo {len(out):,} filas → {dest_fqn} (WRITE_TRUNCATE)...")
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        client.load_table_from_dataframe(out, dest_fqn, job_config=job_config).result()
        log.info(f"[FAKE] Tabla creada: {dest_fqn}")
    else:
        log.info("[FAKE] dry-run (upload=False): no se ha escrito nada en BQ.")

    return info