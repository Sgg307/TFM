# Telco Anomaly — Sistema multi-tier de detección de anomalías sobre tablas BigQuery

Sistema de monitorización de calidad de datos para tablas grandes en BigQuery. Combina cuatro detectores complementarios (T1 estadístico per-columna, T2 PCA cross-column, T3 TranAD agregado, T4 autoencoder a nivel de fila) y un módulo de fusión que reporta consenso día a día.

Originalmente desarrollado sobre las tablas `semantic_portability` y `semantic_discounts` de Orange España (TFM ETSISI–UPM / Cátedra +Orange).

---

## Filosofía

El sistema opera en dos niveles complementarios:

- **Nivel agregado (T3 TranAD)** — aprende cómo evoluciona la tabla día a día como un todo. Los vectores día contienen solo estadísticos universales de distribución; no nombres de columna ni valores concretos. Esto permite reutilizar el mismo modelo y código entre tablas.
- **Nivel column-level (T1 + T2)** — monitoriza cada columna individualmente y sus correlaciones. Compara cada día contra el histórico del mismo día de la semana (DOW-aware), resolviendo la estacionalidad semanal por construcción. Atribuye anomalías a columnas concretas con un mensaje legible.
- **Nivel row-level (T4)** — autoencoder tabular con embeddings de entidad sobre la totalidad de filas (cientos de millones). Detecta filas individualmente inusuales y atribuye el error a la columna que peor reconstruye.

Principios:
- **Ciclicidad resuelta por construcción** — T1/T2 comparan lunes con lunes, martes con martes; T3 usa lag-7 differencing.
- **Multi-tabla desde el diseño** — todo parametrizado por `cfg`. Cada tabla tiene sus propios artefactos en `models/<tabla>/`.
- **Consenso multi-tier** — cada tier captura un tipo distinto de anomalía; la confianza viene del número de tiers que disparan.
- **Explicabilidad nativa** — T1 señala columnas, T2 señala componentes PCA con loadings, T3 reporta canal (Quality/Volume/Structural), T4 reporta drivers por fila.

---

## Arquitectura

```
                       ┌─────────────────────────┐
                       │   BigQuery (raw rows)   │
                       └────────────┬────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
   ┌────────────▼─────────────┐         ┌───────────────▼──────────────┐
   │ pipelines.base           │         │ core.row_level_pipeline      │
   │  → 72 features universal │         │  → schema_encoder + AE       │
   │  → per_col_df (long)     │         │  → 1 score por fila          │
   └─────┬──────────┬─────────┘         └───────────────┬──────────────┘
         │          │                                   │
   ┌─────▼────┐ ┌───▼────────────────┐                  │
   │ TIER 3   │ │ pipelines.         │                  │
   │ TranAD   │ │  column_level      │                  │
   │ (modelo  │ │  → wide + scale    │                  │
   │  agreg.) │ └───┬────────────────┘                  │
   └────┬─────┘     │                                   │
        │     ┌─────▼─────┐  ┌────────────┐             │
        │     │ TIER 1    │  │ TIER 2     │             │
        │     │ z-DOW per │  │ PCA per    │             │
        │     │ columna   │  │ DOW        │             │
        │     └─────┬─────┘  └────┬───────┘             │
        │           │             │                     │
        │     ┌─────▼─────────────▼─────────────────────▼───────┐
        └────►│           core.alert_fusion                     │
              │   anomaly = T1 OR T2 OR T3 OR T4                │
              │   severity = score/threshold por tier           │
              │   n_firing, confidence, tiers_firing            │
              └─────────────────┬───────────────────────────────┘
                                │
                       ┌────────▼────────┐
                       │ dashboard.py    │
                       │ report.py       │
                       └─────────────────┘
```

---

## Estructura del repositorio

```
telco-anomaly/
├── README.md                       ← este archivo
├── ADDING_A_NEW_TABLE.md           ← guía para añadir una tabla nueva
├── requirements.txt
│
├── src/
│   ├── config.py                   ← CFG_<TABLA> + get_cfg(table)
│   ├── main.py                     ← run_multi_tier_mode (orquestador T1+T2+T3)
│   ├── dev_tools.py                ← Runner: bootstrap / monitor / verify / inject / stress
│   ├── stress_test.py              ← Inyecciones sintéticas en espacio escalado
│   ├── test_suites.py              ← Catálogo de suites de stress por tabla
│   ├── report.py                   ← Formateo texto plano de resultados
│   ├── dashboard.py                ← HTML dashboard inline (Jupyter)
│   ├── fake_error.py               ← Inyector de incidentes a nivel raw (BQ)
│   ├── column_maturity_audit.py    ← Auditoría KEEP/REVIEW/EXCLUDE por columna
│   ├── array_struct_aggregator.py  ← Soporte ARRAY<STRUCT> anidado
│   │
│   ├── pipelines/
│   │   ├── base.py                 ← 72 features universales + scaler RobustScaler
│   │   ├── column_level.py         ← pivot long→wide + scaler para T2
│   │   ├── portabilidades.py       ← wrapper de base.py para tabla evento
│   │   └── discounts.py            ← snapshots diarios para tabla sin date_col
│   │
│   └── core/
│       ├── model.py                ← TranAD (PyTorch Lightning) — T3
│       ├── scoring.py              ← Umbrales (p99.5 estático + median+K·MAD dinámico)
│       ├── column_attribution.py   ← per-column stats (T1 + T2)
│       ├── statistical_detector.py ← T1 — z-score DOW-aware por columna
│       ├── pca_detector.py         ← T2 — PCA per-DOW cross-column
│       ├── alert_fusion.py         ← fusión multi-tier (consenso, no heurística)
│       ├── explainability.py       ← SHAP + atribución columna + timeline Plotly
│       ├── labels.py               ← etiquetas humanas compartidas
│       ├── schema_encoder.py       ← T4 — encoder tabular (cat embeddings + num)
│       ├── row_level_model.py      ← T4 — TabularAE (PyTorch)
│       ├── row_level_scoring.py    ← T4 — scoring + umbrales dinámicos
│       └── row_level_pipeline.py   ← T4 — pipeline de entrenamiento e inferencia
│
├── models/                         ← artefactos entrenados (gitignored)
│   └── <tabla>/
│       ├── tier1_baselines.pkl
│       ├── tier2_pca_per_dow.pkl
│       ├── scaler_column_level.pkl
│       ├── column_level_features.json
│       ├── tranad_best.ckpt
│       ├── scaler_quality.pkl
│       ├── scaler_structural.pkl
│       ├── row_level_encoder.pkl
│       ├── row_level_best.pt
│       ├── row_level_thresholds.json
│       └── array_struct_aggregator.pkl   (solo si la tabla lo declara)
│
├── data/                           ← parquets de cache (gitignored)
│   └── <tabla>/
│       ├── raw_features.parquet
│       ├── scaled_features.parquet
│       ├── per_column_stats.parquet
│       └── column_level_scaled.parquet
│
└── plots/                          ← visualizaciones (gitignored)
    └── <tabla>/
```

---

## Requisitos

- **Python** 3.10+
- **GPU NVIDIA con CUDA** para entrenar T3 (TranAD) y T4 (autoencoder). Inferencia funciona en CPU pero es lenta.
- **Acceso a BigQuery** vía Application Default Credentials (`gcloud auth application-default login`).
- Validado sobre Vertex AI Workbench (n1-highmem-8, T4 GPU, 52 GB RAM).

Dependencias en `requirements.txt`. Lo crítico: `torch >= 2.0`, `pytorch-lightning`, `google-cloud-bigquery`, `pandas == 2.3.x`, `pyarrow`, `scikit-learn`, `scipy`, `plotly`, `matplotlib`, `shap`.

---

## Quickstart

### Instalar

```bash
git clone <repo-url> telco-anomaly
cd telco-anomaly
pip install -r requirements.txt

# Autenticación BQ
gcloud auth application-default login
```

### Entrenar una tabla ya configurada

Desde Jupyter / IPython:

```python
import sys; sys.path.insert(0, "src/")
from dev_tools import runner

# Bootstrap completo: descarga + features + entrena T1, T2, T3, T4
runner.bootstrap("portabilidades")

# Solo un subconjunto de tiers
runner.bootstrap("portabilidades", with_tier3=False)   # solo T1+T2+T4
runner.bootstrap("discounts",      with_row_level=False)  # solo T1+T2+T3
```

Crea los artefactos en `models/<tabla>/`.

### Inferir sobre una ventana reciente

```python
# Últimos 20 días, todos los tiers, sobre la tabla del cfg
runner.monitor(eval_days=20)

# Sobre OTRA tabla de BQ (mismo esquema), solo T1+T2
runner.monitor(bq_table="otra_tabla", eval_days=20, tiers=[1, 2])

# Hasta una fecha concreta
runner.monitor(eval_days=30, end_date="2025-12-01")
```

Imprime un resumen por consola. Para ver detalle:

```python
from dashboard import display_dashboard
display_dashboard(runner.results_multi,
                  results_row_level=runner.results_row_level)
```

### Healthcheck

```python
# FP rate sobre datos sanos + stress test de detección sintética
runner.verify("portabilidades")
```

### Inyectar un incidente sintético (validación operativa)

Crea una tabla `<tabla>_fake_error` en BQ con UN día corrompido y la pasa por los cuatro tiers:

```python
runner.inject(
    table_type="portabilidades",
    target_date="2025-10-15",
    injections=[("null_spike", {"col": "brand_donor", "pct": 0.4})],
)
```

Tipos de inyección disponibles en `fake_error.py`:
- `null_spike` — NULL en X% de filas de una columna
- `category_drop` — un valor categórico desaparece
- `category_inject` — valor nunca visto antes
- `volume_drop` — keep% de filas
- `correlation_break` — baraja una columna (marginal intacta, correlación rota)

### Auditar madurez de columnas (recomendado antes de bootstrap)

Identifica columnas que tardan en estabilizarse (latencia de relleno) y recomienda excluirlas:

```python
report = runner.audit_maturity("portabilidades", end_date="2026-04-01")
print(report.recommended_excludes())
```

Las columnas REVIEW/EXCLUDE deben copiarse a `bq.exclude_cols` en el cfg antes de re-entrenar — si no, T1/T2 generan FP por columnas que aún no están llenas.

---

## Comandos clave del Runner

| Comando | Qué hace |
|---|---|
| `runner.bootstrap(table)` | Entrena de cero los tiers seleccionados (T1, T2, T3, T4) y persiste artefactos. |
| `runner.monitor(eval_days=N)` | Inferencia multi-tier sobre los últimos N días. Carga artefactos. |
| `runner.row_level(eval_days=N)` | Solo T4 con umbral dinámico (median+K·MAD sobre la ventana). |
| `runner.verify(table)` | Healthcheck: FP rate + stress test. |
| `runner.stress(suite="standard")` | Ejecuta una suite de stress tests del catálogo. |
| `runner.inject(...)` | Crea tabla `<tabla>_fake_error` en BQ con incidente sintético y la evalúa. |
| `runner.audit_maturity(table)` | Reporte de columnas KEEP/REVIEW/EXCLUDE por latencia de relleno. |
| `runner.inspect(date)` | Detalle de un día concreto del último `monitor()`. |

Todos aceptan `verbose=True` para output detallado y `reload=True` para hot-reload de los módulos en Jupyter.

---

## Cómo funciona cada tier

### T1 — Statistical detector (z-score DOW-aware)

Para cada columna y cada día de la semana, mantiene una baseline (mediana + MAD) de los estadísticos (`pct_null`, `pct_empty`, `pct_unknown`, `entropy_norm`, `hhi`, `top1_share`, `n_cats`). Cada día puntúa cuántas desviaciones se aparta cada (columna, stat) del histórico del mismo DOW.

**Dispara cuando** ≥`min_concentrated_cols` columnas superan `z_threshold` (defaults: 2 columnas, z≥4.0).
**Salida explicable:** lista de columnas con la stat y la dirección del desvío.

### T2 — PCA cross-column

Para cada día de la semana, ajusta un PCA sobre el vector ancho `[cols × stats]`. Cada día puntúa el error de reconstrucción (filtrado por z-score del DOW correspondiente).

**Dispara cuando** el z-score del error de reconstrucción supera `z_threshold` (default 4.0).
**Salida explicable:** componentes con mayor contribución + columnas con mayor peso en esos componentes.

### T3 — TranAD agregado

Transformer encoder-decoder (Tuli et al., 2022) sobre los 72 features universales del día. Loss en dos fases: reconstrucción directa (W1) y reconstrucción condicionada al error (W2). Anomaly score = `mean(|x-W1|² + |x-W2|²)` por feature, agrupado en tres canales (Quality, Volume, Structural).

**Inferencia:** umbrales dinámicos `median + K·MAD` sobre la ventana de evaluación (NO los umbrales absolutos de training — TranAD reconstruye los datos de training demasiado bien y genera FPR cercano al 100%).
**Salida explicable:** canal que dispara + top features contribuyentes + atribución a columnas raw.

### T4 — Row-level autoencoder

`TabularAE` con embeddings de entidad para columnas categóricas (cardinalidad ≤ `max_cardinality`, resto se hashea a `<UNK>`) y normalización robusta para numéricas. Loss = α·MSE(num) + (1-α)·CE(cat). Score por fila = error de reconstrucción.

**Inferencia:** un umbral por fila calibrado dinámicamente (`median + K·MAD` sobre la ventana). Un día se marca anómalo si `pct_anomalous > threshold_pct`.
**Salida explicable:** drivers por fila + agregación a columnas con mayor ratio sobre normal.

### Fusión

```python
anomaly      = any(tier disparado)
n_firing     = nº de tiers disparando
max_severity = max(score_tier / threshold_tier)
confidence   = n_firing / n_total_evaluados
```

NO se suma a un score heurístico. Si un tier no se evalúa (`tiers=[1,2]`) NO entra en el denominador.

---

## Outputs persistidos

Tras `runner.bootstrap("<tabla>")`, en `models/<tabla>/`:

- `tier1_baselines.pkl` — mediana + MAD por (columna, stat, DOW)
- `tier2_pca_per_dow.pkl` — 7 PCAs (uno por DOW) + error stats
- `scaler_column_level.pkl` + `column_level_features.json` — escalador frozen + lista de columnas (contract para inferencia)
- `tranad_best.ckpt` — modelo Lightning (incluye hparams)
- `scaler_quality.pkl`, `scaler_structural.pkl` — escaladores RobustScaler para grupos Q/S
- `row_level_encoder.pkl` — SchemaEncoder con vocabs y normalización
- `row_level_best.pt` — pesos del TabularAE
- `row_level_thresholds.json` — umbrales calibrados sobre val
- `array_struct_aggregator.pkl` — solo si la tabla declara `array_struct_features`

Las `data/<tabla>/*.parquet` son cache; se regeneran si se borran (con `force=True` o si no existen).

---

## Limitaciones conocidas

- **TranAD es un tercer voto, no el protagonista.** A granularidad diaria con ~1800 muestras de entrenamiento, no supera consistentemente a los baselines estadísticos. KernelPCA RBF con cero parámetros entrenables alcanza AUC-PR equivalente. El sistema está diseñado de modo que T3 aporte un voto adicional, no que cargue la decisión.
- **T4 es ciego a caídas de volumen** y a cambios en la distribución de frecuencias por categoría. Complementa, no reemplaza, a T1.
- **Los umbrales absolutos no generalizan** para modelos de reconstrucción. La inferencia siempre usa calibración dinámica (`median + K·MAD`) sobre la ventana actual.
- **El borde temporal es la zona difícil.** Días con cobertura de columnas inmaduras generan FP si no se excluyen del entrenamiento. Pasar por `runner.audit_maturity` antes de entrenar.
- **No es un sustituto del juicio operativo.** Una alerta no es una incidencia hasta que un humano la valida.

---

## Ver también

- **[`ADDING_A_NEW_TABLE.md`](./ADDING_A_NEW_TABLE.md)** — guía paso a paso para añadir una tabla nueva al sistema (privacidad, ARRAY<STRUCT>, snapshot vs evento, etc.).
