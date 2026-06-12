"""pca_detector.py — Tier 2: PCA per-DOW cross-column anomaly detection"""

from __future__ import annotations
import logging, pickle
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


class CrossColumnPCADetector:
    def __init__(self, explained_variance=0.95, z_threshold=4.0):
        self.explained_variance, self.z_threshold = explained_variance, z_threshold
        self.pca_per_dow: Dict[int, PCA] = {}
        self.error_stats: Dict[int, dict] = {}
        self.feature_cols: List[str] = []

    def fit(self, wide_scaled: np.ndarray, dates: pd.DatetimeIndex, train_mask: np.ndarray, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        train_data, train_dates = wide_scaled[train_mask], dates[train_mask]

        for dow in range(7):
            mask = np.array([d.dayofweek == dow for d in train_dates])
            if mask.sum() < 15:
                log.warning(f"[TIER2] DOW {dow}: {mask.sum()} días — insuficiente"); continue

            data = train_data[mask]
            pca = PCA(n_components=self.explained_variance, svd_solver="full")
            pca.fit(data)
            errors = np.mean((data - pca.inverse_transform(pca.transform(data))) ** 2, axis=1)
            self.pca_per_dow[dow] = pca
            self.error_stats[dow] = {"mu": float(errors.mean()), "sigma": float(errors.std()+1e-10), "p99": float(np.percentile(errors, 99))}
            log.info(f"[TIER2] DOW {dow}: {pca.n_components_} PCs ({pca.explained_variance_ratio_.sum():.1%})")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path,"wb") as f:
            pickle.dump({"pca_per_dow": self.pca_per_dow, "error_stats": self.error_stats,
                         "feature_cols": self.feature_cols, "explained_variance": self.explained_variance,
                         "z_threshold": self.z_threshold}, f)

    def load(self, path: Path):
        with open(path,"rb") as f: d = pickle.load(f)
        self.pca_per_dow, self.error_stats = d["pca_per_dow"], d["error_stats"]
        self.feature_cols, self.explained_variance = d["feature_cols"], d["explained_variance"]
        self.z_threshold = d["z_threshold"]
        log.info(f"[TIER2] Cargado desde {path}")

    def score_day(self, day_features_scaled: np.ndarray, day_date) -> Dict:
        dow = pd.Timestamp(day_date).dayofweek
        pca = self.pca_per_dow.get(dow)
        if pca is None: return _empty(day_date)

        x = day_features_scaled.reshape(1, -1)
        recon = pca.inverse_transform(pca.transform(x))
        fe = (day_features_scaled - recon.flatten()) ** 2
        te = float(fe.mean())
        st = self.error_stats[dow]
        z = (te - st["mu"]) / st["sigma"]

        # Top features
        top_k = min(15, len(self.feature_cols))
        top_idx = np.argsort(fe)[::-1][:top_k]
        total_fe = fe.sum() + 1e-12
        tf = pd.DataFrame({"feature": [self.feature_cols[i] for i in top_idx],
                           "recon_error": fe[top_idx], "pct_of_total": 100*fe[top_idx]/total_fe})
        tf["column"] = tf["feature"].str.rsplit("__",n=1).str[0]
        tf["stat"] = tf["feature"].str.rsplit("__",n=1).str[1]

        # Top components
        latent = pca.transform(x).flatten()
        cs = latent ** 2
        top_ci = np.argsort(cs)[::-1][:5]
        comp_rows = []
        for ci in top_ci:
            ld = pca.components_[ci]
            tli = np.argsort(np.abs(ld))[::-1][:5]
            comp_rows.append({"component": ci, "score": float(cs[ci]),
                              "var_explained": float(pca.explained_variance_ratio_[ci]),
                              "top_loadings": ", ".join(f"{self.feature_cols[j]} ({ld[j]:+.3f})" for j in tli)})

        # Aggregate by column
        col_err = {}
        for i, f in enumerate(self.feature_cols):
            c = f.rsplit("__",1)[0]
            col_err[c] = col_err.get(c, 0.) + fe[i]
        tc = pd.DataFrame([{"column":c,"recon_error":e} for c,e in col_err.items()]).sort_values("recon_error",ascending=False).head(10).reset_index(drop=True)

        return {"date": day_date, "anomaly": abs(z) > self.z_threshold,
                "total_error": te, "z_score": float(z),
                "n_components": pca.n_components_, "top_features": tf,
                "top_components": pd.DataFrame(comp_rows), "top_columns": tc}


def _empty(date):
    return {"date":date,"anomaly":False,"total_error":0.,"z_score":0.,"n_components":0,
            "top_features":pd.DataFrame(),"top_components":pd.DataFrame(),"top_columns":pd.DataFrame()}

def format_tier2_report(result: Dict) -> str:
    if not result["anomaly"]:
        return f"  Tier 2 (PCA): ✅ OK (z={result['z_score']:.2f}, {result['n_components']} PCs)"
    lines = [f"  Tier 2 (PCA): 🚨 z={result['z_score']:.2f} ({result['n_components']} PCs)"]
    tc = result.get("top_components")
    if tc is not None and not tc.empty:
        for _, r in tc.head(3).iterrows():
            lines.append(f"    • PC{r['component']}: score={r['score']:.4f} ({r['var_explained']:.1%})")
            lines.append(f"      {r['top_loadings']}")
    tcol = result.get("top_columns")
    if tcol is not None and not tcol.empty:
        lines.append(f"    Columnas: {', '.join(tcol.head(5)['column'])}")
    return "\n".join(lines)
