"""
dev_tools.py — Utilidades de desarrollo para Jupyter / Vertex AI
================================================================

Uso típico:

    from dev_tools import runner

    # Entrenar desde cero (todos los tiers o un subset)
    runner.bootstrap("portabilidades")
    runner.bootstrap("portabilidades", with_tier3=False)   # solo T1+T2+T4

    # Inferencia sobre una ventana reciente
    runner.monitor(bq_table="..._error_mvt", eval_days=20)
    runner.monitor(eval_days=20, tiers=[1, 2])             # sin TranAD ni T4
    runner.monitor(eval_days=20, tiers=[1, 2, 3, 4])       # con row-level también

    # Inyección de incidente sintético + reacción multi-tier
    runner.inject(injections=[("null_spike", {"col": "brand_donor", "pct": 0.6})])

    # Healthcheck (FP rates + stress test)
    runner.verify("portabilidades")

    # Stress test independiente
    runner.stress(suite="standard")

    # Dashboard
    from dashboard import display_dashboard
    display_dashboard(runner.results_multi, results_row_level=runner.results_row_level)
"""

from __future__ import annotations
import sys
import copy
import importlib
import logging
from pathlib import Path
from typing import List, Optional, Union

_PROJECT_MODULES = [
    "config", "pipelines.base", "pipelines.discounts",
    "core.model", "core.scoring", "core.explainability", "core.column_attribution",
    "pipelines.column_level", "core.statistical_detector", "core.pca_detector",
    "core.alert_fusion", "stress_test", "main", "report", "dashboard", "fake_error",
    "core.schema_encoder", "core.row_level_model", "core.row_level_scoring",
    "core.row_level_pipeline", "core.labels", "test_suites", "column_maturity_audit",
]

_NOISY_LOGGERS = [
    None, "pytorch_lightning", "lightning.pytorch", "lightning",
    "lightning.pytorch.utilities.rank_zero", "lightning.fabric.utilities.rank_zero",
    "pipelines.base", "core.model", "core.scoring", "core.column_attribution",
    "pipelines.column_level", "core.statistical_detector", "core.pca_detector",
    "core.alert_fusion", "core.explainability", "main",
    "core.schema_encoder", "core.row_level_model", "core.row_level_scoring",
    "core.row_level_pipeline",
]


class TelcoDevRunner:
    def __init__(self):
        self.results_multi: dict = None
        self.results_row_level: dict = None
        self.last_table: str = None

    def reload_all(self, quiet: bool = False):
        reloaded = []
        for name in _PROJECT_MODULES:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
                reloaded.append(name)
        if not quiet:
            print(f"Reloaded: {', '.join(reloaded)}")

    # ═════════════════════════════════════════════════════════════════════════
    # MONITOR — inferencia multi-tier sobre una ventana
    # ═════════════════════════════════════════════════════════════════════════

    def monitor(
        self,
        table_type:       str  = "portabilidades",
        bq_table:         Optional[str] = None,
        bq_dataset:       Optional[str] = None,
        eval_days:        int  = 20,
        end_date:         Union[None, str, int] = None,
        tiers:            Optional[List[int]] = None,
        verbose:          bool = False,
        plot:             bool = False,
        show_table:       bool = False,
        force_retrain:    bool = False,
        force_refeatures: bool = False,
        reload:           bool = True,
    ) -> dict:
        """Punto de entrada unificado para inferencia multi-tier.

        Args:
            table_type: pipeline a usar ("portabilidades", "discounts").
            bq_table:   nombre BQ alternativo. None = la del config.
            tiers:      subconjunto de [1, 2, 3, 4] a evaluar (default todos).
                        T1/T2/T3 se calculan en run_multi_tier_mode; T4 se
                        ejecuta como pipeline complementario via row_level().
            end_date:   None = MAX(date) en BQ. str = "YYYY-MM-DD". int = offset.
        """
        import pandas as pd

        if tiers is None:
            tiers = [1, 2, 3, 4]
        tiers = [int(t) for t in tiers]
        run_multi = any(t in tiers for t in (1, 2, 3))
        run_t4    = 4 in tiers

        saved_levels = []
        if not verbose:
            for name in _NOISY_LOGGERS:
                lg = logging.getLogger(name)
                saved_levels.append((lg, lg.level))
                lg.setLevel(logging.ERROR)
        if reload:
            self.reload_all(quiet=not verbose)

        from config import get_cfg, DATA_DIR, PLOTS_DIR
        from main import run_multi_tier_mode
        from report import format_developer, format_period_table

        cfg = copy.deepcopy(get_cfg(table_type))

        # ── Override de BQ + paths efímeros por tabla externa ────────────────
        if bq_table or bq_dataset:
            if bq_table:    cfg["bq"]["table"]   = bq_table
            if bq_dataset:  cfg["bq"]["dataset"] = bq_dataset
            safe = (bq_table or cfg["bq"]["table"]) \
                .replace(".", "_").replace("`", "").replace(" ", "_")
            d = DATA_DIR / safe; d.mkdir(parents=True, exist_ok=True)
            p = PLOTS_DIR / safe; p.mkdir(parents=True, exist_ok=True)
            cfg["paths"]["raw_data"]          = d / "raw_features.parquet"
            cfg["paths"]["scaled_data"]        = d / "scaled_features.parquet"
            cfg["paths"]["column_level_data"]  = d / "column_level_scaled.parquet"
            cfg["paths"]["plots_dir"]          = p
            for f in d.glob("*.parquet"):
                f.unlink()

        # ── Rango de fechas y train_split efectivo ───────────────────────────
        actual_end = self._resolve_end_date(end_date)
        if bq_table or bq_dataset or end_date is not None:
            if actual_end is None:
                actual_end = self._query_max_date(cfg)
            seq_len = cfg.get("tranad", {}).get("seq_len", 14)
            context = seq_len + 7 + 3
            start_dt = pd.Timestamp(actual_end) - pd.Timedelta(days=eval_days + context)
            cfg["bq"]["start_date"] = start_dt.strftime("%Y-%m-%d")
            cfg["bq"]["end_date"]   = actual_end
            total_est = eval_days + context
            cfg.setdefault("tranad", {})["train_split"] = (
                max(total_est - eval_days - 2, 7) / total_est
            )

        cfg["force_download"] = False
        cfg["force_retrain"]  = force_retrain
        cfg["force_features"] = force_refeatures
        cfg["_eval_days"]     = eval_days

        display_name = bq_table or cfg["bq"]["table"]

        # ── T1+T2+T3 ─────────────────────────────────────────────────────────
        if run_multi:
            try:
                self.results_multi = run_multi_tier_mode(
                    table=table_type,
                    tiers=[t for t in tiers if t in (1, 2, 3)],
                    force_retrain=force_retrain,
                    force_refeatures=force_refeatures,
                    explain_top_k=999,
                    cfg_override=cfg,
                    inference_mode=True,
                )
            finally:
                pass  # restauramos loggers más abajo

            # Filtrar ventana
            period_df = self.results_multi["period_df"].copy()
            if actual_end is not None:
                end_ts   = pd.Timestamp(actual_end)
                start_ts = end_ts - pd.Timedelta(days=eval_days - 1)
                period_df = period_df[
                    (period_df["date"] >= start_ts) & (period_df["date"] <= end_ts)
                ].reset_index(drop=True)
            elif len(period_df) > eval_days:
                period_df = period_df.tail(eval_days).reset_index(drop=True)
            self.results_multi["period_df"] = period_df

            combined = self.results_multi.get("reports", [])
            eval_set = {pd.Timestamp(d) for d in period_df["date"]}
            combined = [r for r in combined if pd.Timestamp(r["date"]) in eval_set]
            self.results_multi["combined_reports"] = combined
            self.results_multi["_display_name"]    = display_name
            self.results_multi["_tiers_run"]       = tiers
        else:
            # T4-only: aún así devolvemos un esqueleto vacío
            self.results_multi = {
                "period_df": pd.DataFrame(),
                "combined_reports": [],
                "_display_name": display_name,
                "_tiers_run": tiers,
            }

        # ── T4 (opcional, se cuelga del results_multi) ───────────────────────
        if run_t4 and self._resolve_date_col(cfg) is not None:
            try:
                rl = self.row_level(
                    table_type=table_type, bq_table=bq_table, bq_dataset=bq_dataset,
                    eval_days=eval_days, end_date=end_date,
                    verbose=verbose, reload=False,
                )
                self.results_multi["row_level"] = rl
            except Exception as exc:
                print(f"  ⚠️ Row-level no disponible: {exc}")
                self.results_multi["row_level"] = None
        elif run_t4:
            print("  ⚠️ Tier 4 omitido: la tabla no tiene date_col (snapshot).")


        # ── Merge T4 en period_df para que entre en la fusión del dashboard ──
        if run_multi and self.results_multi.get("row_level") is not None:
            from core.alert_fusion import merge_tier4_into_period
            rl = self.results_multi["row_level"]
            self.results_multi["period_df"] = merge_tier4_into_period(
                self.results_multi["period_df"],
                rl.get("summary_df"),
                t4_threshold=None,   # calibración dinámica day-level (median + K×MAD)
                cfg=cfg,
            )

        # Restaurar loggers
        for lg, lvl in saved_levels:
            lg.setLevel(lvl)

        self.last_table = table_type

        # ── Output ───────────────────────────────────────────────────────────
        if run_multi:
            pdf = self.results_multi["period_df"]
            if verbose:
                print(format_developer(pdf, self.results_multi.get("combined_reports", []),
                                       display_name))
            if show_table:
                print(format_period_table(pdf))
            if plot:
                self._show_timeline(pdf, table_type)

            n_anom = int(pdf["anomaly"].sum())
            print(f"  {n_anom}/{len(pdf)} anomalías detectadas. "
                  f"Usa display_dashboard(runner.results_multi, "
                  f"results_row_level=runner.results_row_level) para ver.")
        return self.results_multi

    # ═════════════════════════════════════════════════════════════════════════
    # ROW-LEVEL — inferencia rápida (Tier 4)
    # ═════════════════════════════════════════════════════════════════════════

    def row_level(
        self,
        table_type: str  = "portabilidades",
        bq_table:   Optional[str] = None,
        bq_dataset: Optional[str] = None,
        eval_days:  int  = 5,
        end_date:   Union[None, str, int] = None,
        verbose:    bool = True,
        reload:     bool = True,
    ) -> dict:
        """Inferencia row-level con umbral DINÁMICO sobre la ventana.

        Necesita modelo entrenado por `runner.bootstrap(...)` previamente.
        Aplica median + K×MAD sobre los días de contexto (toda la ventana
        excepto el último día) para fijar el umbral por fila. Esto elimina
        la dependencia del percentil estático (que sobre-/sub-estimaba).
        """
        import pandas as pd

        saved_levels = []
        if not verbose:
            for name in _NOISY_LOGGERS:
                lg = logging.getLogger(name)
                saved_levels.append((lg, lg.level))
                lg.setLevel(logging.ERROR)
        if reload:
            self.reload_all(quiet=True)

        try:
            from config import get_cfg
            from core.row_level_pipeline import RowLevelPipeline

            cfg = copy.deepcopy(get_cfg(table_type))
            if bq_table:   cfg["bq"]["table"]   = bq_table
            if bq_dataset: cfg["bq"]["dataset"] = bq_dataset

            actual_end = self._resolve_end_date(end_date)
            if actual_end is None:
                actual_end = self._query_max_date(cfg)

            end_date = actual_end
            end_ts   = pd.Timestamp(actual_end)
            start_ts = end_ts - pd.Timedelta(days=eval_days - 1)

            cfg["bq"]["start_date"] = start_ts.strftime("%Y-%m-%d")
            cfg["bq"]["end_date"]   = actual_end
            # Forzar 100% sampling para inferencia (1 partición × N días)
            cfg["row_level"]["sample_pct_train"] = cfg["row_level"].get(
                "sample_pct_infer", 100,
            )

            pipe = RowLevelPipeline(cfg, model_type="ae")
            pipe.load()
            print(f"  Modelo row-level cargado ({pipe.model_path.name})")

            print(f"  Descargando {start_ts.strftime('%Y-%m-%d')} → {actual_end}...")
            from google.cloud import bigquery as bqlib
            client = bqlib.Client(project=cfg["bq"]["project_id"])
            bq = cfg["bq"]
            fqn = f"`{bq['project_id']}.{bq['dataset']}.{bq['table']}`"

            date_col_event = bq.get("date_col")
            init_col       = bq.get("init_date_col")
            end_col        = bq.get("end_date_col")

            if date_col_event:
                # Tabla event-based (p.ej. portabilidades)
                date_col = date_col_event
                query = (
                    f"SELECT * FROM {fqn} "
                    f"WHERE {date_col} BETWEEN '{start_ts.strftime('%Y-%m-%d')}' "
                    f"AND '{actual_end}'"
                )
                df = client.query(query).to_dataframe(progress_bar_type="tqdm")
                print(f"  {len(df):,} filas descargadas")

            elif init_col and end_col:
                # Tabla snapshot. Procesamos día por día SIN concatenar — para tablas
                # grandes (millones de activos vitalicios) el concat explota la RAM.

                sample_pct_infer = int(cfg.get("row_level", {}).get("sample_pct_infer", 100))
                sample_hash_col  = bq.get("sample_hash_col") or bq.get("pk_col") or "clave_pk"
                sample_clause = ""
                if 0 < sample_pct_infer < 100:
                    sample_clause = (
                        f"AND MOD(ABS(FARM_FINGERPRINT(CAST({sample_hash_col} AS STRING))), 100) "
                        f"< {sample_pct_infer} "
                    )
                    print(f"  Sample SQL: {sample_pct_infer}% por {sample_hash_col}")

                query = (
                    f"SELECT * FROM {fqn} "
                    f"WHERE {init_col} <= '{actual_end}' "
                    f"AND ({end_col} >= '{start_ts.strftime('%Y-%m-%d')}' "
                    f"OR {end_col} IS NULL "
                    f"OR EXTRACT(YEAR FROM {end_col}) >= 9999) "
                    f"{sample_clause}"
                )
                df_raw = client.query(query).to_dataframe(progress_bar_type="tqdm")
                print(f"  {len(df_raw):,} descuentos activos descargados (snapshot)")

                # Fix robusto de fechas
                far_future = pd.Timestamp("2099-12-31")
                for col in (init_col, end_col):
                    s = df_raw[col].astype(str)
                    s = s.where(~s.str.startswith("9999"), other=None)
                    df_raw[col] = pd.to_datetime(s, errors="coerce")
                df_raw[end_col]  = df_raw[end_col].fillna(far_future)
                df_raw[init_col] = df_raw[init_col].fillna(pd.Timestamp("1970-01-01"))

                # Procesamiento por día: scoring inline, sin construir df mega.
                import gc
                days = list(pd.date_range(start_ts, end_ts))
                print(f"  Procesando {len(days)} días sin concatenar (eficiente en RAM)...")
            
                thr_static = float(pipe.scorer.row_threshold)
                reports    = []
                for i, day in enumerate(days, 1):
                    mask = (df_raw[init_col] <= day) & (df_raw[end_col] >= day)
                    day_df = df_raw.loc[mask].copy()
                    day_df["_snapshot_date"] = day

                    report = pipe.score_single_day(
                        day_df, str(day.date()),
                        threshold=thr_static,
                    )
                    reports.append(report)
                    print(f"    día {i}/{len(days)} ({day.strftime('%Y-%m-%d')}): "
                          f"{report.n_total:>7,} filas | {report.n_anomalous:>5,} anómalas "
                          f"({report.pct_anomalous:.2%})")
                    del day_df
                    gc.collect()

                del df_raw
                gc.collect()

                # Saltamos el flujo standard de scoring (que esperaba un df grande)
                thr_used = thr_static
                df = None        # señal para saltar el bloque siguiente
                date_col = "_snapshot_date"

            else:
                raise RuntimeError(
                    "cfg.bq necesita date_col (event-based) o "
                    "init_date_col + end_date_col (snapshot)"
                )

            # ── Score con dynamic threshold sobre la ventana ────────────────
            if df is not None:
                if eval_days >= 3:
                    out = pipe.score_range_dynamic(df, date_col, n_eval=1)
                    reports = out["reports"]
                    thr_used = out["threshold"]
                    if len(reports) < len(out["scores_by_day"]):
                        reports = []
                        df_c = df.copy()
                        df_c["_d"] = pd.to_datetime(df_c[date_col]).dt.normalize()
                        for d in sorted(out["scores_by_day"].keys()):
                            day_df = df_c[df_c["_d"] == d].drop(columns=["_d"])
                            reports.append(pipe.score_single_day(day_df, str(d.date()),
                                                     threshold=thr_used))
                else:
                    df["_d"] = pd.to_datetime(df[date_col]).dt.normalize()
                    dates = sorted(df["_d"].unique())
                    reports = [
                        pipe.score_single_day(
                            df[df["_d"] == d].drop(columns=["_d"]),
                            str(pd.Timestamp(d).date()),
                        )
                        for d in dates
                    ]
                    thr_used = pipe.scorer.row_threshold
            # Si df is None, reports y thr_used ya vienen del bloque snapshot

            summary_df = pd.DataFrame([
                {"date":          r.date,
                 "n_total":       r.n_total,
                 "n_anomalous":   r.n_anomalous,
                 "pct_anomalous": r.pct_anomalous,
                 "mean_score":    r.mean_score,
                 "top_col":       r.top_columns[0]["col"] if r.top_columns else "",
                 "threshold":     r.threshold_used}
                for r in reports
            ])

            # ── Imprimir ─────────────────────────────────────────────────────
            display_name = bq_table or cfg["bq"]["table"]
            _DOW = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
            print(f"\n{'═'*65}")
            print(f"  ROW-LEVEL — {display_name}  (umbral {thr_used:.4f})")
            print(f"  Periodo: {start_ts.strftime('%Y-%m-%d')} → {actual_end}  "
                  f"({len(reports)} días)")
            print(f"{'═'*65}")
            for r in reports:
                icon = "🟡" if r.pct_anomalous > 0.005 else "✅"
                print(f"  {icon} {r.date.strftime('%Y-%m-%d')} ({_DOW.get(r.dow, '?')})  "
                      f"{r.n_total:>7,} filas  "
                      f"{r.n_anomalous:>5,} anómalas ({r.pct_anomalous:.2%})")
                if verbose and r.top_columns:
                    for tc in r.top_columns[:3]:
                        ratio = tc.get("ratio_vs_normal", 0)
                        marker = "🔴" if ratio > 5 else "🟡" if ratio > 2 else "  "
                        print(f"       {marker} {tc['col']:<30} "
                              f"{ratio:.1f}x vs normal")
            total_anom = summary_df["n_anomalous"].sum()
            total_rows = summary_df["n_total"].sum()
            print(f"\n  Total: {total_anom:,}/{total_rows:,} filas anómalas "
                  f"({total_anom / max(total_rows, 1):.3%})")
            print(f"{'═'*65}")

            self.results_row_level = {
                "reports":       reports,
                "summary_df":    summary_df,
                "pipeline":      pipe,
                "_threshold":    thr_used,
                "_display_name": display_name,
            }
            return self.results_row_level

        finally:
            for lg, lvl in saved_levels:
                lg.setLevel(lvl)

    # ═════════════════════════════════════════════════════════════════════════
    # BOOTSTRAP — entrena los tiers seleccionados de una tabla, desde cero
    # ═════════════════════════════════════════════════════════════════════════

    def bootstrap(
        self,
        table_type:     str  = "portabilidades",
        force:          bool = False,
        with_tier1:     bool = True,
        with_tier2:     bool = True,
        with_tier3:     bool = True,
        with_row_level: bool = True,
        model_type:     str  = "ae",
        verbose:        bool = True,
        reload:         bool = True,
    ) -> dict:
        """Entrena y persiste artefactos para los tiers solicitados.

        Cada `with_tierN` flag activa/desactiva ese tier independientemente.
        La tabla `discounts` (sin date_col) salta T4 automáticamente.
        """
        from pathlib import Path

        if reload:
            self.reload_all(quiet=not verbose)

        saved_levels = []
        if not verbose:
            for name in _NOISY_LOGGERS:
                lg = logging.getLogger(name)
                saved_levels.append((lg, lg.level))
                lg.setLevel(logging.ERROR)

        from config import get_cfg
        from main import run_multi_tier_mode

        cfg_tpl = get_cfg(table_type)
        date_col = self._resolve_date_col(cfg_tpl)

        tiers_to_run = []
        if with_tier1: tiers_to_run.append(1)
        if with_tier2: tiers_to_run.append(2)
        if with_tier3: tiers_to_run.append(3)

        print(f"\n{'═'*65}")
        print(f"  BOOTSTRAP — {table_type}  (force={force}, tiers={tiers_to_run}"
              f"{', +T4' if with_row_level else ''})")
        print(f"{'═'*65}")

        results_mt = None
        rl_summary = None
        try:
            # ── 1. Multi-tier (T1 + T2 + T3) ─────────────────────────────────
            if tiers_to_run:
                print(f"\n  [1/2] Multi-tier (tiers={tiers_to_run})...")
                # deepcopy: no contaminamos el cfg global.
                cfg_run = copy.deepcopy(cfg_tpl)
                results_mt = run_multi_tier_mode(
                    table=table_type,
                    tiers=tiers_to_run,
                    force_retrain=force,
                    force_refeatures=force,
                    explain_top_k=0,
                    cfg_override=cfg_run,
                    inference_mode=False,
                )
            else:
                print("\n  [1/2] Multi-tier: omitido (sin tiers seleccionados).")

            # ── 2. Row-level (T4) ────────────────────────────────────────────
            if with_row_level and date_col is None:
                print(f"\n  [2/2] Row-level: SALTADO — '{table_type}' no tiene "
                      f"date_col (tabla snapshot).")
            elif with_row_level:
                print(f"\n  [2/2] Row-level (T4 autoencoder, model_type={model_type})...")
                from core.row_level_pipeline import RowLevelPipeline
                pipe = RowLevelPipeline(copy.deepcopy(cfg_tpl), model_type=model_type)
                rl_summary = pipe.train(force_download=force)
                pipe.save()
                print(f"        Row-level entrenado: "
                      f"{rl_summary.get('n_params', 0):,} params, "
                      f"{rl_summary.get('train_time_s', 0):.0f}s")
            else:
                print("\n  [2/2] Row-level: omitido (with_row_level=False).")
        finally:
            for lg, lvl in saved_levels:
                lg.setLevel(lvl)

        # ── Verificación de artefactos ───────────────────────────────────────
        paths = cfg_tpl["paths"]
        print(f"\n  {'─'*61}")
        print(f"  Artefactos en models/{table_type}/:")
        checks = []
        if with_tier1: checks.append(("tier1",        paths["tier1_baselines"]))
        if with_tier2: checks.append(("tier2",        paths["tier2_pca"]))
        if with_tier2: checks.append(("col_scaler",   paths["column_level_scaler"]))
        if with_tier2: checks.append(("col_contract", paths["column_level_features"]))
        if with_tier3: checks.append(("tranad",       paths["tranad_model"]))
        if with_tier3: checks.append(("scaler_q",     paths["scaler_quality"]))
        if with_tier3: checks.append(("scaler_s",     paths["scaler_structural"]))
        for name, p in checks:
            ok = "✅" if Path(p).exists() else "❌"
            print(f"    {ok} {name:<14} {Path(p).name}")
        if rl_summary is not None:
            print(f"    ✅ {'row_encoder':<14} {Path(paths['row_level_encoder']).name}")
            print(f"    ✅ {'row_model':<14} {Path(paths['row_level_model']).name}")
            print(f"    ✅ {'row_thresh':<14} {Path(paths['row_level_thresholds']).name}")
        print(f"  {'─'*61}")
        print(f"  Listo. Ejecuta: runner.verify('{table_type}')")
        print(f"{'═'*65}\n")

        self.last_table = table_type
        return {
            "results_multi":     results_mt,
            "row_level_summary": rl_summary,
            "tiers_trained":     tiers_to_run + ([4] if rl_summary else []),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # VERIFY — healthcheck: FP en datos sanos + detección sintética
    # ═════════════════════════════════════════════════════════════════════════

    def verify(
        self,
        table_type:     str  = "portabilidades",
        eval_days:      int  = 30,
        with_row_level: bool = True,
        with_stress:    bool = True,
        fp_target:      float = 0.05,
        verbose:        bool = False,
        end_date:       Optional[str] = None,
    ) -> dict:
        """Healthcheck del sistema para una tabla.

        Bloques:
          1. FP MULTI-TIER  — inferencia sobre datos sanos.
          2. FP ROW-LEVEL   — ídem a nivel de fila (si la tabla lo soporta).
          3. DETECCIÓN T3   — stress test sobre el modelo cargado en (1).
                              Reutiliza scaled_data/dates del monitor para
                              evitar un segundo pase de feature pipeline.

        Requiere runner.bootstrap(table_type) previo.
        """
        import pandas as pd
        from pathlib import Path
        from config import get_cfg

        cfg = get_cfg(table_type)
        date_col = self._resolve_date_col(cfg)

        # ── Comprobar artefactos ─────────────────────────────────────────────
        missing = [n for n, p in [
            ("tranad", cfg["paths"]["tranad_model"]),
            ("tier1",  cfg["paths"]["tier1_baselines"]),
            ("tier2",  cfg["paths"]["tier2_pca"]),
        ] if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"Faltan artefactos {missing}. "
                               f"Ejecuta runner.bootstrap('{table_type}') primero.")

        print(f"\n{'█'*65}\n  VERIFY — {table_type}\n{'█'*65}")
        out = {}

        # ── 1. FP MULTI-TIER ─────────────────────────────────────────────────
        print(f"\n  ── 1. Falsos positivos MULTI-TIER (datos sanos, {eval_days}d) ──")
        res_mt = self.monitor(
            table_type=table_type, eval_days=eval_days,
            end_date=end_date, tiers=[1, 2, 3],
            verbose=verbose, reload=True,
        )
        pdf = res_mt["period_df"]
        n_days = len(pdf)
        n_anom = int(pdf["anomaly"].sum())
        fp_multi = n_anom / max(n_days, 1)
        n_t3 = int(pdf.get("t3_anomaly", pd.Series(dtype=bool)).sum())
        n_t2 = int(pdf.get("t2_anomaly", pd.Series(dtype=bool)).sum())
        verdict_mt = "✅" if fp_multi <= fp_target else "⚠️"
        print(f"     {verdict_mt} FP global: {n_anom}/{n_days} = {fp_multi:.1%} "
              f"(objetivo <{fp_target:.0%})")
        print(f"        Desglose: T3(TranAD)={n_t3}  T2(PCA)={n_t2}")
        out["period_df"] = pdf
        out["fp_multi"]  = fp_multi

        # ── 2. FP ROW-LEVEL ──────────────────────────────────────────────────
        out["fp_row_level"] = None
        if with_row_level and date_col is None:
            print(f"\n  ── 2. Row-level: N/A ('{table_type}' es snapshot) ──")
        elif with_row_level:
            rl_days = min(eval_days, 7)
            print(f"\n  ── 2. Falsos positivos ROW-LEVEL (datos sanos, {rl_days}d) ──")
            try:
                res_rl = self.row_level(
                    table_type=table_type, eval_days=rl_days,
                    end_date=end_date, verbose=verbose, reload=False,
                )
                sdf        = res_rl["summary_df"]
                total_anom = int(sdf["n_anomalous"].sum())
                total_rows = int(sdf["n_total"].sum())
                fp_row     = total_anom / max(total_rows, 1)
                mean_pct   = sdf["pct_anomalous"].mean()
                max_pct    = sdf["pct_anomalous"].max()
                verdict_rl = "✅" if mean_pct <= fp_target else "⚠️"
                print(f"     {verdict_rl} FP filas: {fp_row:.3%} global  |  "
                      f"media diaria {mean_pct:.2%}  |  pico {max_pct:.2%}")
                out["summary_df"]   = sdf
                out["fp_row_level"] = mean_pct
            except FileNotFoundError:
                print("     ⚠️ Modelo row-level no encontrado. "
                      "Corre bootstrap con with_row_level=True.")
        else:
            print("\n  ── 2. Row-level: omitido (with_row_level=False) ──")

        # ── 3. STRESS TEST sobre el modelo ya cargado ────────────────────────
        out["stress_df"] = None
        if with_stress:
            print(f"\n  ── 3. Detección sintética TIER 3 (inyección en espacio escalado) ──")
            t3r = res_mt.get("tier3_results")
            if t3r is None or t3r.get("scaled_data") is None:
                print("     ⚠️ No hay artefactos T3 disponibles en results_multi.")
            else:
                from stress_test import run_stress_test
                from core.scoring import generate_alerts

                scaled_data  = t3r["scaled_data"]
                dates_univ   = t3r["dates"]
                feature_cols = t3r["feature_cols"]
                features_df  = pd.DataFrame(scaled_data, columns=feature_cols)
                features_df.insert(0, "date", dates_univ)

                stress_df = run_stress_test(
                    scaled_data=scaled_data,
                    features_df=features_df,
                    feature_cols=feature_cols,
                    model=t3r["model"],
                    alerts_fn=generate_alerts,
                    thresholds=t3r["thresholds"],
                    cfg=get_cfg(table_type),
                )
                out["stress_df"] = stress_df
                valid = stress_df[stress_df["detected"].notna()]
                n_det = int(valid["detected"].sum()) if len(valid) else 0
                out["stress_detected"] = (n_det, len(valid))

        # ── VEREDICTO ────────────────────────────────────────────────────────
        print(f"\n{'█'*65}\n  VEREDICTO — {table_type}\n{'█'*65}")
        print(f"    Multi-tier FP : {out['fp_multi']:.1%} "
              f"{'✅' if out['fp_multi'] <= fp_target else '⚠️ revisar'}")
        if out["fp_row_level"] is not None:
            print(f"    Row-level FP  : {out['fp_row_level']:.2%} "
                  f"{'✅' if out['fp_row_level'] <= fp_target else '⚠️ revisar'}")
        if out["stress_df"] is not None:
            nd, nt = out["stress_detected"]
            print(f"    Detección T3  : {nd}/{nt} anomalías sintéticas "
                  f"{'✅' if nd == nt and nt > 0 else '⚠️'}")
        print(f"{'█'*65}\n")

        self.last_table = table_type
        return out

    # ═════════════════════════════════════════════════════════════════════════
    # STRESS — ejecuta una suite del catálogo test_suites
    # ═════════════════════════════════════════════════════════════════════════

    def stress(
        self,
        suite:        str = "standard",
        table_type:   str = "portabilidades",
        test_date:    Optional[str] = None,
        eval_days:    int = 30,
        verbose:      bool = False,
        reload:       bool = True,
    ) -> "pd.DataFrame":
        """Ejecuta una suite de stress tests (catalogo de `test_suites.py`).

        Carga modelo + features via monitor(tiers=[3]) y aplica las inyecciones
        sintéticas en el espacio de features escaladas. Reporta detección por
        cada test.

        Args:
            suite:      nombre de suite en test_suites.SUITES.
            table_type: tabla a usar.
            test_date:  fecha objetivo a corromper. None = la default de la
                        suite o (último día - 7).

        Returns: DataFrame con resultados por test (mismo formato que
                 stress_test.run_stress_tests).
        """
        import pandas as pd
        from config import get_cfg
        from test_suites import resolve_suite
        from stress_test import run_stress_tests
        from core.scoring import generate_alerts

        if reload:
            self.reload_all(quiet=not verbose)

        # 1. Resolver la suite
        tests, default_date = resolve_suite(suite, table_type)
        target = test_date or default_date

        # 2. Convertir formato suite → [(date, kind, intensity), ...]
        # test_suites entrega normalmente tuplas (kind, intensity) o
        # (date_offset, kind, intensity). Las normalizamos:
        norm_tests = []
        for t in tests:
            if len(t) == 2:
                kind, intensity = t
                norm_tests.append((target or -7, kind, intensity))
            elif len(t) == 3:
                a, b, c = t
                if isinstance(a, str) and (target is None or test_date is None):
                    norm_tests.append((a, b, c))
                elif test_date is not None:
                    norm_tests.append((test_date, b, c))
                else:
                    norm_tests.append((a, b, c))
            else:
                raise ValueError(f"Formato de test desconocido: {t!r}")

        # 3. Cargar modelo y datos via monitor (solo T3)
        res_mt = self.monitor(
            table_type=table_type, eval_days=eval_days,
            tiers=[3], verbose=verbose, reload=False,
        )
        t3r = res_mt.get("tier3_results")
        if t3r is None or t3r.get("scaled_data") is None:
            raise RuntimeError("No se pudo cargar el modelo TranAD. "
                               "Comprueba que bootstrap se ejecutó antes.")

        scaled_data  = t3r["scaled_data"]
        dates_univ   = t3r["dates"]
        feature_cols = t3r["feature_cols"]
        features_df  = pd.DataFrame(scaled_data, columns=feature_cols)
        features_df.insert(0, "date", dates_univ)

        return run_stress_tests(
            tests=norm_tests,
            scaled_data=scaled_data,
            features_df=features_df,
            feature_cols=feature_cols,
            model=t3r["model"],
            alerts_fn=generate_alerts,
            thresholds=t3r["thresholds"],
            cfg=get_cfg(table_type),
        )

    # ═════════════════════════════════════════════════════════════════════════
    # INJECT — incidente sintético a nivel RAW + reacción multi-tier
    # ═════════════════════════════════════════════════════════════════════════

    def inject(
        self,
        table_type:     str  = "portabilidades",
        target_date:    Optional[str] = None,
        injections:     Optional[list] = None,
        window_days:    int  = 50,
        eval_days:      int  = 15,
        dest_table:     Optional[str] = None,
        dest_suffix:    str  = "fake_error",
        upload:         bool = True,
        run_tiers:      bool = True,
        with_row_level: bool = True,
        verbose:        bool = False,
        reload:         bool = True,
    ) -> dict:
        """Crea una tabla `_fake_error` con un día corrompido y la pasa por
        todos los tiers (incluido T4 si la tabla lo soporta)."""
        import pandas as pd
        from config import get_cfg
        from fake_error import make_fake_error_table

        if reload:
            self.reload_all(quiet=not verbose)

        cfg = copy.deepcopy(get_cfg(table_type))

        info = make_fake_error_table(
            cfg, target_date=target_date, injections=injections,
            window_days=window_days, dest_table=dest_table,
            dest_suffix=dest_suffix, upload=upload,
        )

        print(f"\n{'═'*65}\n  INYECCIÓN — {info.dest_table}\n"
              f"  Día corrompido: {info.target_date}  "
              f"(ventana sana {info.window_start} → {info.window_end})\n{'═'*65}")
        for m in info.manifest:
            print(f"  • {m['effect']}")
        print(f"  Tiers que DEBERÍAN cazarlo: {', '.join(info.expected_tiers)}\n{'═'*65}")

        if not upload or not run_tiers:
            print("\n  (sin ejecución de tiers: upload=False o run_tiers=False)")
            return {"info": info}

        target = info.target_date

        # ── Multi-tier sobre la fake (sample 100%) ───────────────────────────
        tiers_to_eval = [1, 2, 3] + ([4] if with_row_level else [])
        # Truco: sample_pct=100 al monitor cuando se especifica bq_table → se
        # ajusta dentro del propio config copiado. Lo seteamos via cfg directo
        # tras el deepcopy del monitor.
        prev_pct = cfg["bq"].get("sample_pct", 100)
        cfg["bq"]["sample_pct"] = 100
        try:
            res_mt = self.monitor(
                table_type=table_type, bq_table=info.dest_table,
                eval_days=eval_days, end_date=target,
                tiers=tiers_to_eval, verbose=verbose, reload=False,
            )
        finally:
            cfg["bq"]["sample_pct"] = prev_pct

        rl_row = None
        res_rl = res_mt.get("row_level")
        if res_rl and isinstance(res_rl, dict) and "summary_df" in res_rl:
            sdf = res_rl["summary_df"]
            m = sdf[pd.to_datetime(sdf["date"]).dt.strftime("%Y-%m-%d") == target]
            if not m.empty:
                rl_row = m.iloc[0]

        # ── Consenso en el día objetivo ──────────────────────────────────────
        pdf = res_mt["period_df"]
        tgt = pdf[pd.to_datetime(pdf["date"]).dt.strftime("%Y-%m-%d") == target]

        print(f"\n{'█'*65}\n  CONSENSO MULTI-TIER — {target}\n{'█'*65}")
        if tgt.empty:
            print(f"  ⚠️ {target} no está en la ventana evaluada (sube eval_days).")
        else:
            r = tgt.iloc[0]
            yn = lambda b: "🔴 SÍ" if b else "⚪ no"
            print(f"  T1 Estadístico : {yn(r.get('t1_n_flagged', 0) > 0)}  "
                  f"({int(r.get('t1_n_flagged', 0))} cols, "
                  f"z_max={r.get('t1_max_z', 0):.1f})")
            print(f"  T2 PCA         : {yn(bool(r.get('t2_anomaly', False)))}  "
                  f"(z={r.get('t2_z_score', 0):.2f})")
            print(f"  T3 TranAD      : {yn(bool(r.get('t3_anomaly', False)))}  "
                  f"(canal: {r.get('t3_channel', '—')})")
            if rl_row is not None:
                print(f"  T4 Row-level   : {yn(rl_row['pct_anomalous'] > 0.005)}  "
                      f"({rl_row['pct_anomalous']:.2%} filas, top: {rl_row['top_col']})")
            elif with_row_level:
                print(f"  T4 Row-level   : (no evaluado)")
            print(f"  {'─'*61}")
            print(f"  VEREDICTO: {'🔴 ANOMALÍA' if r.get('anomaly') else '⚪ normal'}  "
                  f"| confianza {r.get('confidence', 0):.0%} "
                  f"| detectores: {r.get('tiers_firing', '—')}")
            print(f"  Esperado: {', '.join(info.expected_tiers)}")
        print(f"{'█'*65}\n")
        print(f"  Detalle: runner.inspect('{target}')  | "
              f"dashboard: display_dashboard(runner.results_multi, "
              f"results_row_level=runner.results_row_level)")

        self.last_table = table_type
        return {
            "info":          info,
            "period_df":     pdf,
            "target_row":    None if tgt.empty else tgt.iloc[0].to_dict(),
            "row_level":     res_rl,
            "results_multi": res_mt,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # AUDIT_MATURITY — identifica columnas inmaduras a excluir del entrenamiento
    # ═════════════════════════════════════════════════════════════════════════

    def audit_maturity(
        self,
        table_type:    str = "portabilidades",
        end_date:      Optional[str] = None,
        lookback_days: int = 180,
        sample_pct:    int = 10,
        threshold_pp:  float = 5.0,
        extra_exclude: Optional[List[str]] = None,
        verbose:       bool = True,
    ):
        """Audita cuánto tarda cada columna en estabilizarse y devuelve
        recomendación KEEP/REVIEW/EXCLUDE.

        Útil ANTES de re-bootstrappear: identifica las columnas inmaduras
        que conviene excluir del entrenamiento para que la inferencia
        sobre ayer/hoy no genere FP por columnas que aún no están llenas.

        Returns: MaturityReport (mira .recommended_excludes()).
        """
        from column_maturity_audit import audit_maturity
        return audit_maturity(
            table_type=table_type,
            end_date=end_date,
            lookback_days=lookback_days,
            sample_pct=sample_pct,
            threshold_pp=threshold_pp,
            extra_exclude=extra_exclude,
            verbose=verbose,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # INSPECT — detalle de un día específico
    # ═════════════════════════════════════════════════════════════════════════

    def inspect(self, date: str):
        if self.results_multi is None:
            raise RuntimeError("Ejecuta runner.monitor() primero.")
        import pandas as pd
        from report import format_day_detail
        ts = pd.Timestamp(date)
        for cr in self.results_multi.get("combined_reports", []):
            if pd.Timestamp(cr["date"]) == ts:
                print(format_day_detail(cr))
                return cr
        pdf = self.results_multi["period_df"]
        if (pdf["date"] == ts).any():
            print(f"  {date} — día normal.")
        else:
            print(f"  {date} — fuera de la ventana evaluada.")
        return None

    def inspect_row_level(self, date: str):
        if self.results_row_level is None:
            raise RuntimeError("Ejecuta runner.row_level() primero.")
        import pandas as pd
        ts = pd.Timestamp(date)
        for r in self.results_row_level["reports"]:
            if r.date == ts:
                print(r.summary)
                return r
        print(f"  {date} — no encontrado en la evaluación row-level.")
        return None

    # ═════════════════════════════════════════════════════════════════════════
    # HELPERS PRIVADOS
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_end_date(end_date):
        if end_date is None: return None
        if isinstance(end_date, str): return end_date
        if isinstance(end_date, int):
            import pandas as pd
            return (pd.Timestamp.now().normalize()
                    + pd.Timedelta(days=end_date)).strftime("%Y-%m-%d")
        raise ValueError(f"end_date inválido: {end_date!r}")

    @staticmethod
    def _resolve_date_col(cfg) -> str | None:
        """Devuelve la columna de fecha activa: date_col, init_date_col o snapshot_col."""
        bq = (cfg or {}).get("bq", {}) or {}
        return bq.get("date_col") or bq.get("init_date_col") or bq.get("snapshot_col")

    
    @staticmethod
    def _query_max_date(cfg):
        from google.cloud import bigquery
        bq = cfg["bq"]
        date_col = TelcoDevRunner._resolve_date_col(cfg)
        if not date_col:
            raise ValueError("cfg.bq necesita date_col, init_date_col o snapshot_col")
        fqn = f"`{bq['project_id']}.{bq['dataset']}.{bq['table']}`"
        client = bigquery.Client(project=bq["project_id"])
        hash_col = bq.get("sample_hash_col")
        pct      = bq.get("sample_pct", 100)
        sample_clause = ""
        if hash_col and pct < 100:
            sample_clause = (
                f" AND MOD(ABS(FARM_FINGERPRINT(CAST({hash_col} AS STRING))), 100) < {pct}"
            )
        row = client.query(
            f"SELECT DATE(MAX({date_col})) AS d FROM {fqn} "
            f"WHERE {date_col} <= CURRENT_DATE(){sample_clause}"
        ).to_dataframe()
        return row["d"].iloc[0].strftime("%Y-%m-%d")

    def _show_timeline(self, period_df, table):
        try:
            from core.explainability import plot_anomaly_timeline
            if self.results_multi and self.results_multi.get("tier3_results"):
                alerts_df = self.results_multi["tier3_results"]["alerts_df"]
                from config import PLOTS_DIR
                fig = plot_anomaly_timeline(
                    alerts_df, {"table_name": table},
                    save_path=PLOTS_DIR / table / "anomaly_timeline.html",
                )
                fig.show()
        except Exception:
            pass

    def __repr__(self):
        status = ""
        if self.results_multi:
            pdf = self.results_multi.get("period_df")
            if pdf is not None and len(pdf):
                n = int(pdf["anomaly"].sum())
                status = f" | {n}/{len(pdf)} anomalías"
        return f"<TelcoDevRunner tabla={self.last_table or '-'}{status}>"


runner = TelcoDevRunner()
