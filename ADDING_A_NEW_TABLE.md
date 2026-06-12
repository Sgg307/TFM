# Cómo añadir una tabla nueva al sistema

Esta guía cubre todo lo necesario para conectar el sistema a una tabla BigQuery nueva. La arquitectura es **multi-tabla por diseño**: el motor (detectores, scoring, fusión) no cambia entre tablas, lo único específico vive en `config.py` y (opcionalmente) un wrapper en `pipelines/`.

Sigue los pasos en orden. Cada uno es de unos minutos.

---

## Resumen del proceso

1. **Auditar el esquema** — distinguir tabla evento vs snapshot, identificar PII, mapear ARRAY<STRUCT> anidados.
2. **Decidir muestreo** — qué columna usar para hash y a qué porcentaje.
3. **Auditar madurez de columnas** — qué columnas excluir por latencia de relleno.
4. **Escribir `CFG_<TABLA>` en `config.py`** — clave principal del sistema.
5. **Registrarla en `TABLES`**.
6. **Crear `pipelines/<tabla>.py`** (solo si tabla snapshot, evento usa el wrapper de `portabilidades.py`).
7. **Entrenar** con `runner.bootstrap("<tabla>")`.
8. **Verificar** con `runner.verify("<tabla>")`.

---

## 1. Auditar el esquema

Antes de configurar nada, contesta cuatro preguntas sobre tu tabla:

### 1a. ¿Es evento o snapshot?

- **Evento** — cada fila ocurre en un instante. Tiene una columna `date_col` única.
  - Ejemplo: `portabilidades` (cada fila es una solicitud de portabilidad con `request_date`).
  - Pipeline: usa `pipelines/base.py` directamente vía un wrapper simple (3 líneas, como `portabilidades.py`).

- **Snapshot** — cada fila representa un estado válido en un rango. Tiene `init_date` y `end_date` (o equivalentes); para evaluar el día D filtras `init_date ≤ D ≤ end_date`.
  - Ejemplo: `discounts` (cada fila es un descuento activo entre `init_date` y `max_end_date`).
  - Pipeline: necesitas un módulo dedicado en `pipelines/` (puedes copiar `discounts.py` como plantilla).

### 1b. ¿Qué columnas son PII / sensibles?

**Crítico — revisar antes de cualquier entrenamiento.** El sistema entrena modelos sobre los datos; cualquier columna con PII contamina los artefactos pickled y los pesos del autoencoder.

Identifica y **excluye** todas las columnas que cumplan:

- Identificadores directos: `customer_id`, `customer_gid`, `account_id`, `service_id`, `phone_nm`, email, DNI.
- Cuasi-identificadores que permiten reidentificación: códigos postales completos, IDs de comercial (`dealer`, `sfid`), IDs internos que mapean a personas físicas.
- Cualquier campo con datos personales sensibles (RGPD art. 9): salud, ideología, biometría, etc.

Estas columnas van **siempre** en `bq.exclude_cols` (apartado 4 más abajo) y también en `row_level.exclude_cols`. El sistema tiene un `PrivacyGuardError` en `SchemaEncoder.fit()` que aborta si detecta columnas con patrones PII conocidos sin excluir, pero **no confíes** en él como única defensa: declara explícitamente.

> **Buena práctica:** mantén un registro central de columnas PII por tabla y revísalo con el equipo de privacidad antes de entrenar. RGPD art. 4(1) y AI Act art. 10 imponen requisitos sobre los datos de entrenamiento.

### 1c. ¿Hay columnas con tipo ARRAY<STRUCT<...>>?

BigQuery permite columnas anidadas como:

```sql
funcional ARRAY<STRUCT<
  service_type  STRING,
  technology_ds STRING,
  tariff_fee    FLOAT64,
  crm STRUCT<
    segment_ds     STRING,
    sub_segment_ds STRING,
    billed ARRAY<STRUCT<imp_without_tax FLOAT64>>
  >
>>
```

Si tu tabla tiene alguna, el sistema las soporta vía `array_struct_aggregator.py`. Tienes que declararlas explícitamente en el cfg (apartado 4d). Si no, las columnas anidadas se ignoran silenciosamente (no rompen, pero pierdes la señal).

### 1d. ¿Hay columnas que tardan en estabilizarse?

Muchas tablas tienen columnas que se rellenan con días o meses de retraso (ARPUs derivados, columnas de facturación, dimensiones del CRM con maduración). Si entrenas el sistema con esas columnas y luego intentas inferir sobre "ayer", verás % de nulos altísimo en columnas que están bien — solo aún no se han llenado. Eso es FP garantizado.

**Recomendado:** ejecuta `runner.audit_maturity("<tabla>")` ANTES de bootstrap (paso 7). Devuelve para cada columna el lag de maduración estimado y una recomendación KEEP / REVIEW / EXCLUDE. Las REVIEW y EXCLUDE deben copiarse a `bq.exclude_cols`.

Si la tabla tiene un **dual SLA** (algunas columnas frescas, otras maduras con latencia conocida), puedes crear dos cfgs:
- Uno "fresh" excluye las inmaduras → para T1+T2+T3 en producción diaria.
- Uno "mature" con todas las columnas → para T4 entrenado offline con datos consolidados.

Ver `CFG_PORTABILIDADES_FRESH` en `config.py` como ejemplo. La tabla `discounts` también funciona con dual SLA: `bq.exclude_cols` y `row_level.exclude_cols` son distintos.

---

## 2. Decidir el muestreo

Sobre tablas de cientos de millones de filas, entrenar al 100% no aporta señal proporcional al coste. El sistema muestrea por hash sobre una columna estable:

```python
"sample_hash_col": "phone_nm",   # columna por la que hashear
"sample_pct":      10,           # porcentaje a retener (0..100)
```

La consulta resultante usa `MOD(ABS(FARM_FINGERPRINT(CAST(<col> AS STRING))), 100) < <pct>`. Esto garantiza reproducibilidad (el mismo subconjunto siempre) y representatividad (uniforme sobre el dominio de la columna).

**Elegir `sample_hash_col`:**

- Para tablas **evento**: la PK natural o un identificador de entidad estable (`phone_nm` en portabilidades). No uses la fecha — concentrarías el muestreo.
- Para tablas **snapshot**: la PK del snapshot (`clave_pk` en discounts). Cuidado si la PK es compuesta o cambia entre snapshots.

**Elegir `sample_pct`:**

- 10% es el default validado. Da ~580M filas para portabilidades, ~38M para discounts — suficiente para entrenar.
- T4 (row-level) usa además `row_level.sample_pct_train` (default 10) para training y `sample_pct_infer` (default 100) para inferencia diaria.

---

## 3. Auditar madurez (opcional pero muy recomendado)

```python
from dev_tools import runner
report = runner.audit_maturity(
    table_type="<tu_tabla>",
    end_date="2026-04-01",
    lookback_days=180,
)
print(report.recommended_excludes())
```

Devuelve columnas con latencia detectada > umbral. Cópialas a `bq.exclude_cols` en el cfg del paso 4.

---

## 4. Escribir `CFG_<TABLA>` en `config.py`

El cfg es la **única pieza específica de la tabla**. Tiene esta estructura general:

```python
CFG_MI_TABLA = {
    "bq": {
        # ─── Identificación BQ ───────────────────────────────────────
        "project_id":       "mi-proyecto-gcp",
        "dataset":          "MI_DATASET",
        "table":            "mi_tabla",

        # ─── Columnas de fecha (DEPENDE del tipo de tabla) ───────────
        # Evento: una sola columna
        "date_col":         "request_date",
        # Snapshot: dejar date_col=None y declarar init/end
        # "date_col":       None,
        # "init_date_col":  "init_date",
        # "end_date_col":   "max_end_date",
        # "snapshot_col":   "snapshot_date",   # opcional, para versionado

        # ─── Esquema declarado (opcional, defaults seguros) ──────────
        "category_cols":    [],     # si quieres restringir T2 a una lista
        "numeric_cols":     [],     # ídem

        # ─── Muestreo ────────────────────────────────────────────────
        "sample_hash_col":  "phone_nm",
        "sample_pct":       10,

        # ─── Ventana temporal de entrenamiento ───────────────────────
        "start_date":       "2023-01-01",
        "end_date":         "2025-12-31",
        "train_end":        "2025-10-31",
        "test_start":       "2025-11-01",

        # ─── EXCLUSIÓN DE COLUMNAS ───────────────────────────────────
        # Aquí van TODAS las columnas que no quieres ver:
        #   - PII / RGPD (obligatorio)
        #   - Columnas inmaduras (audit_maturity → REVIEW/EXCLUDE)
        #   - Columnas funcionalmente descartadas
        "exclude_cols": [
            # PII — RGPD art. 4(1) / AI Act art. 10
            "customer_id", "customer_gid", "account_id", "service_id",
            "zip_code", "dealer", "subdealer",
            # Funcionales (auditoría operativa)
            "billing_type_receiver", "brand_receiver",
            # Inmaduras (audit_maturity)
            "tariff_fee_lag30d", "ARPU_lag90d",
        ],
    },

    # ─── Paths automáticos ───────────────────────────────────────────
    "paths": _paths("mi_tabla"),   # crea models/mi_tabla/, data/mi_tabla/, plots/mi_tabla/

    # ─── Hiperparámetros por tier (usar defaults salvo casos extremos) ─
    "tranad":         dict(_DEFAULT_TRANAD),
    "scoring":        dict(_DEFAULT_SCORING),
    "stress_test":    dict(_DEFAULT_STRESS),
    "explainability": dict(_DEFAULT_EXPLAINABILITY),
    "tier1":          dict(_DEFAULT_TIER1),
    "tier2":          dict(_DEFAULT_TIER2),

    # ─── Tier 4 — row-level (si la tabla lo soporta) ─────────────────
    "row_level": {
        **_DEFAULT_ROW_LEVEL,
        # Sobreescribir solo lo que aplique:
        "train_start":  "2024-01-01",
        "train_end":    "2025-06-30",
        "val_start":    "2025-06-30",
        "val_end":      "2025-10-01",
        "eval_start":   "2025-10-01",
        # CRÍTICO: lista propia de exclusiones para el row-level.
        # Típicamente es bq.exclude_cols + cualquier columna que el AE
        # NO debe ver (PII residual, IDs, fechas absolutas, etc.).
        "exclude_cols": [
            "customer_id", "customer_gid", "account_id", "service_id",
            "zip_code", "dealer", "subdealer",
            "billing_type_receiver", "brand_receiver",
            # IDs y fechas absolutas que solo añaden ruido al AE:
            "clave_pk", "discount_id", "init_date", "max_end_date",
        ],
    },

    # ─── Flags ───────────────────────────────────────────────────────
    "force_download": False,
    "force_features": False,
    "force_retrain":  False,
    "verbose":        True,
}
```

### 4d. Si tienes columnas ARRAY<STRUCT>

Añade una sección `array_struct_features` al cfg, **una entrada por columna anidada**:

```python
CFG_MI_TABLA = {
    # ... resto del cfg arriba ...

    "array_struct_features": {
        # NOMBRE de la columna BQ con tipo ARRAY<STRUCT<...>>
        "funcional": {

            # ─ Tokens categóricos: cada uno se convierte en multi-hot ─
            "categorical_tokens": {
                "service_type":   {"path": ["service_type"]},
                "technology_ds":  {"path": ["technology_ds"]},
                "line_type_ds":   {"path": ["line_type_ds"]},

                # null_as_token: True si quieres que NULL sea un token explícito
                "bundle_type":    {"path": ["bundle_type"], "null_as_token": True},

                # path puede tener varios niveles (struct anidado)
                "segment_ds":     {"path": ["crm", "segment_ds"]},
                "sub_segment_ds": {"path": ["crm", "sub_segment_ds"]},

                # Vocabulario top-N para cardinalidad alta:
                # "vocab": "top_n", "top_n": 50, "include_other": True
            },

            # ─ Agregados numéricos: mean/max/sum/std sobre el array ──
            "numeric_aggs": {
                "tariff_fee":  {"path": ["tariff_fee"],
                                "aggs": ["mean", "max"]},
                "ARPU_REAL":   {"path": ["ARPU_REAL"],
                                "aggs": ["mean", "max"]},
                # Path anidado: struct dentro de struct dentro de array
                "imp_without_tax": {
                    "path": ["crm", "billed", "imp_without_tax"],
                    "aggs": ["mean", "sum"],
                },
            },

            # ─ Lengths estructurales: forma del array ────────────────
            "length_features": [
                {"name": "len_outer",
                 "path": [], "mode": "count"},
                {"name": "total_len_crm",
                 "path": ["crm"], "mode": "count"},
                {"name": "total_len_billed",
                 "path": ["crm", "billed"], "mode": "count"},
                {"name": "pct_crm_with_billed",
                 "path": ["crm"], "mode": "pct_nonempty_child", "child": "billed"},
            ],
        },

        # Si tienes más columnas array, añade otra entrada:
        # "otra_columna_array": { ... },
    },
}
```

**Modos de `length_features`:**
- `"count"` — número total de elementos en la subestructura del path.
- `"pct_nonempty_child"` — porcentaje de elementos del padre cuyo hijo `child` está poblado.

**El aggregator es genérico y no-op por defecto.** Si tu tabla no tiene ARRAY<STRUCT>, no declares la clave y no se ejecuta nada extra.

### 4e. Registrar en `TABLES`

Al final de `config.py`:

```python
TABLES: dict[str, dict] = {
    "portabilidades":       CFG_PORTABILIDADES,
    "portabilidades_fresh": CFG_PORTABILIDADES_FRESH,
    "discounts":            CFG_DISCOUNTS,
    "mi_tabla":             CFG_MI_TABLA,   # ← añadir esta línea
}
```

A partir de ahora `get_cfg("mi_tabla")` devuelve tu cfg.

---

## 5. Crear el pipeline (solo si es snapshot)

### Caso evento

Crea `src/pipelines/mi_tabla.py` con 3 líneas (réplica de `portabilidades.py`):

```python
from pipelines.base import run_feature_pipeline
from config import CFG_MI_TABLA

def run_pipeline(cfg: dict = None, **overrides):
    c = dict(cfg or CFG_MI_TABLA)
    c.update(overrides)
    return run_feature_pipeline(c)
```

Y en `main.py`, dentro de `_run_table_pipeline`, añade un caso si quieres llamarlo por nombre — aunque para tablas evento el default ya funciona sin tocar nada.

### Caso snapshot

Copia `pipelines/discounts.py` como `pipelines/mi_tabla.py` y adapta:

- La query de descarga (`_build_download_query`): ajusta filtros y rango de fechas.
- El bucle `build_all_from_snapshots`: usa tus `init_date_col` / `end_date_col`.
- Las columnas categóricas vs numéricas: la auto-detección de `base.py` suele bastar, pero puedes especificarlas en `cfg["bq"]["category_cols"]` / `numeric_cols`.

En `main.py`, dentro de `_run_table_pipeline`, añade:

```python
if table == "mi_tabla":
    from pipelines.mi_tabla import run_pipeline
    return run_pipeline(cfg)
```

---

## 6. Bootstrap

```python
from dev_tools import runner

# Entrenar todos los tiers
runner.bootstrap("mi_tabla")

# Sin T4 si la tabla es snapshot sin date_col adecuada
runner.bootstrap("mi_tabla", with_row_level=False)

# Forzar redescarga y reentrenamiento
runner.bootstrap("mi_tabla", force=True)
```

Tarda según el tamaño de la tabla y la GPU disponible. Para una tabla de ~50M filas al 10%: 15-30 min descarga + features, 5-15 min entrenando T3, 10-20 min T4.

Comprueba al final que en `models/mi_tabla/` están todos los `.pkl`, `.ckpt`, `.pt`, `.json` esperados.

---

## 7. Verificar

```python
runner.verify("mi_tabla")
```

Esto:
1. Corre `monitor()` sobre los últimos 30 días sanos y reporta FP rate por tier. Objetivo `<5%`.
2. Hace lo mismo a nivel de fila para T4.
3. Lanza un stress test sintético (Tier 3) y verifica detección.

Si FP > 5%:
- Revisa qué columnas disparan T1 → es muy probable que sean columnas inmaduras que se filtraron. Vuelve al `audit_maturity` y añade más a `exclude_cols`.
- Si dispara T3 sin razón aparente, revisa `tranad.k_sigma` en el cfg (default 5.0; puedes subirlo a 6 o 7 si el ruido base es alto).

Si la detección sintética falla:
- Verifica que el entrenamiento convergió: revisa los logs de TranAD en el último bootstrap.
- Si las inyecciones son sobre columnas con cardinalidad >500, T1/T2 las ignoran por construcción (umbral `_CAT_MAX_CARDINALITY`).

---

## 8. Uso normal a partir de aquí

```python
# Monitorización diaria
runner.monitor(table_type="mi_tabla", eval_days=20)

# Detalle de un día concreto
runner.inspect("2025-11-15")

# Dashboard
from dashboard import display_dashboard
display_dashboard(runner.results_multi,
                  results_row_level=runner.results_row_level)
```

Para una validación operativa contundente del sistema sobre la tabla nueva, prueba `runner.inject(...)` con varios tipos de corrupción a nivel raw y verifica que cada tier caza lo que debe (la tabla de consenso esperada está en `fake_error._EXPECTED_TIERS`).

---

## Checklist mínimo antes de subir a producción

- [ ] **PII** — todas las columnas identificadoras directas y cuasi-identificadores están en `bq.exclude_cols` Y en `row_level.exclude_cols`. Revisado con privacidad.
- [ ] **Madurez** — `runner.audit_maturity` ejecutado, recomendaciones aplicadas.
- [ ] **ARRAY<STRUCT>** — si existen, declaradas en `array_struct_features` o aceptado conscientemente que se ignoran.
- [ ] **Muestreo** — `sample_hash_col` es estable y representativo. `sample_pct` justificado (10% es el default validado).
- [ ] **Bootstrap completo** — los siete artefactos esperados están en `models/<tabla>/`.
- [ ] **Verify** — `fp_multi <= 5%`, `fp_row_level <= 5%`, stress test detecta lo que debe.
- [ ] **Inject end-to-end** — al menos una corrupción sintética por tipo (`null_spike`, `category_drop`, `correlation_break`, `volume_drop`) probada y cazada por el tier esperado.

---

## FAQ

**P: ¿Puedo usar el sistema sobre una tabla pequeña (~1M filas)?**
R: Sí, pero baja `sample_pct` a 100 y considera saltar T4 (`with_row_level=False`) — los embeddings de entidad necesitan volumen.

**P: ¿Y si mi tabla no tiene día de la semana (frecuencia mensual, p.ej.)?**
R: T1/T2 asumen ciclo semanal por construcción. Para una tabla mensual, tendrías que adaptar el agrupamiento DOW por mes en `statistical_detector.py` y `pca_detector.py`. No es trivial; abre una issue.

**P: ¿Puedo usar otra base que no sea BigQuery?**
R: Sí, pero hay que reescribir las queries en `pipelines/base.py`, `pipelines/discounts.py` (o tu pipeline equivalente) y la descarga de `core/row_level_pipeline.py`. La infraestructura del resto del sistema es agnóstica.

**P: Mi tabla tiene 200 columnas. ¿Va a funcionar?**
R: Sí. T1/T2 escalan linealmente con columnas; T3 trabaja sobre los 72 features universales (independiente del nº de columnas raw); T4 con embeddings funciona bien hasta varios cientos de columnas categóricas siempre que la cardinalidad agregada sea razonable.

**P: ¿Cómo desactivo permanentemente un tier?**
R: Pasa `tiers=[1, 2]` (sin el que quieras quitar) a `runner.monitor()` y `runner.bootstrap()`. No hace falta tocar el cfg.
