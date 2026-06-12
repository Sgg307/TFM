"""
test_suites.py — Suites de stress test por tabla
=================================================
Centraliza TODOS los escenarios de test en un único sitio.
Cada suite es una lista de (tipo_anomalía, intensidad | None).

Uso desde dev_tools.py (no llamar directamente en notebooks):
    from test_suites import resolve_suite
    tests, date = resolve_suite("sensitivity", "portabilidades")

Para añadir una tabla nueva:
    1. Crea una sección con su TEST_DATE por defecto.
    2. Define sus suites en SUITES["<nombre_tabla>"].
    3. Si tiene tipos de anomalía propios, añádelos a _VALID_TYPES.
"""

from __future__ import annotations

# ── Tipos de anomalía soportados ─────────────────────────────────────────────
_VALID_TYPES = {
    "quality_spike",
    "quality_global",
    "cat_disappear",
    "cat_appear",
    "volume_drop",
    "schema_change",
    "distribution_shift",
}

# ── Fechas de referencia por tabla ────────────────────────────────────────────
# Usar martes-jueves de meses sin festivos para mayor estabilidad.
_DEFAULT_DATES: dict[str, str] = {
    "portabilidades": "2025-01-30",
    "discounts":      "2025-02-18",   # ajustar cuando esté disponible
}

# ── Suites ────────────────────────────────────────────────────────────────────
#
# Estructura:
#   SUITES[tabla][nombre_suite] = [(tipo, intensidad), …]
#
# intensidad: float 0-1 ó None (el módulo stress_test usa su propio default)

SUITES: dict[str, dict[str, list]] = {

    # ── portabilidades ────────────────────────────────────────────────────────
    "portabilidades": {

        # Cobertura completa, intensidades canónicas del paper
        "standard": [
            ("quality_spike",      0.80),
            ("quality_global",     0.60),
            ("cat_disappear",      None),
            ("cat_appear",         None),
            ("volume_drop",        0.70),
            ("volume_drop",        0.15),
            ("volume_drop",        0.10),
            ("schema_change",      None),
            ("distribution_shift", 0.80),
        ],

        # Límites inferiores — busca el umbral de detección real
        "sensitivity": [
            ("quality_spike",      0.30),
            ("quality_spike",      0.20),
            ("quality_spike",      0.15),   # límite esperado ~25 %
            ("quality_global",     0.25),
            ("cat_disappear",      0.40),   # desaparición parcial
            ("cat_appear",         0.30),
            ("volume_drop",        0.15),
            ("volume_drop",        0.10),
            ("volume_drop",        0.08),   # caída mínima detectable
            ("schema_change",      0.40),
            ("distribution_shift", 0.30),
        ],

        # Escenarios rápidos para smoke-test tras un cambio de código
        "smoke": [
            ("quality_spike",      0.80),
            ("volume_drop",        0.70),
            ("distribution_shift", 0.80),
        ],

        # Suite completa = standard + sensitivity (sin duplicados)
        "full": [
            ("quality_spike",      0.80),
            ("quality_spike",      0.30),
            ("quality_spike",      0.20),
            ("quality_spike",      0.15),
            ("quality_global",     0.60),
            ("quality_global",     0.25),
            ("cat_disappear",      None),
            ("cat_disappear",      0.40),
            ("cat_appear",         None),
            ("cat_appear",         0.30),
            ("volume_drop",        0.70),
            ("volume_drop",        0.15),
            ("volume_drop",        0.10),
            ("volume_drop",        0.08),
            ("schema_change",      None),
            ("schema_change",      0.40),
            ("distribution_shift", 0.80),
            ("distribution_shift", 0.30),
        ],
    },

    # ── discounts (semantic_discounts) ────────────────────────────────────────
    "discounts": {

        "standard": [
            ("quality_spike",      0.80),
            ("quality_global",     0.60),
            ("volume_drop",        0.70),
            ("distribution_shift", 0.80),
        ],

        "smoke": [
            ("quality_spike",      0.80),
            ("volume_drop",        0.70),
        ],

        # Ampliar cuando el modelo de descuentos esté entrenado
        "full": [
            ("quality_spike",      0.80),
            ("quality_spike",      0.30),
            ("quality_global",     0.60),
            ("quality_global",     0.25),
            ("volume_drop",        0.70),
            ("volume_drop",        0.15),
            ("volume_drop",        0.10),
            ("distribution_shift", 0.80),
            ("distribution_shift", 0.30),
        ],
    },
}


# ── API pública ───────────────────────────────────────────────────────────────

def resolve_suite(
    suite_name: str,
    table: str = "portabilidades",
) -> tuple[list[tuple], str]:
    """
    Devuelve (lista_de_tests, fecha_por_defecto) para una suite y tabla.

    Parameters
    ----------
    suite_name : 'standard' | 'sensitivity' | 'smoke' | 'full'
    table      : clave de tabla registrada en SUITES

    Returns
    -------
    tests : lista de (tipo, intensidad)
    date  : fecha de referencia por defecto para esa tabla
    """
    if table not in SUITES:
        available = ", ".join(sorted(SUITES))
        raise KeyError(
            f"Tabla '{table}' no está registrada. "
            f"Tablas disponibles: {available}"
        )

    table_suites = SUITES[table]
    if suite_name not in table_suites:
        available = ", ".join(sorted(table_suites))
        raise KeyError(
            f"Suite '{suite_name}' no existe para la tabla '{table}'. "
            f"Suites disponibles: {available}"
        )

    date = _DEFAULT_DATES.get(table, "2025-01-30")
    return table_suites[suite_name], date


def list_suites(table: str | None = None) -> None:
    """Imprime las suites disponibles, opcionalmente filtradas por tabla."""
    tables = [table] if table else sorted(SUITES)
    for tbl in tables:
        print(f"\n📋 {tbl}  (fecha por defecto: {_DEFAULT_DATES.get(tbl, '—')})")
        for suite_name, tests in SUITES[tbl].items():
            print(f"   {suite_name:15s}  {len(tests)} escenarios")
