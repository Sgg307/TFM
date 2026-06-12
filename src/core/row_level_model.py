"""
row_level_model.py — Autoencoder tabular para detección de anomalías row-level.

  - TabularAE:  autoencoder determinista (MSE numéricas + CE categóricas)
  - TabularVAE: variante variacional (+ β·KL)

Entity embeddings para categóricas, RobustScaler para numéricas (vía SchemaEncoder).
Expone error de reconstrucción por feature para explicabilidad directa.

    encoder = SchemaEncoder.load("models/portabilidades/row_level_encoder.pkl")
    model   = TabularAE.from_encoder(encoder, bottleneck_dim=64)
    trainer = RowLevelTrainer(model, encoder, cfg)
    trainer.fit(train_df, val_df)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.schema_encoder import SchemaEncoder, EncodedBatch

log = logging.getLogger(__name__)


# ── Autoencoder base ──────────────────────────────────────────────────────────

class TabularAE(nn.Module):
    """AE tabular con entity embeddings. Encoder → bottleneck → decoder → heads (CE por cat, linear para num)."""

    def __init__(
        self,
        cat_specs: List[Tuple[str, int, int]],   # (name, n_cats, emb_dim)
        n_num_features: int,
        encoder_layers: List[int] = (256, 128),
        bottleneck_dim: int = 64,
        decoder_layers: List[int] = (128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cat_specs = cat_specs
        self.n_num_features = n_num_features
        self.bottleneck_dim = bottleneck_dim

        # Entity embeddings (padding_idx=1 = PAD_TOKEN del SchemaEncoder)
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for name, n_cats, emb_dim in cat_specs:
            self.embeddings[name] = nn.Embedding(n_cats, emb_dim, padding_idx=1)
            total_emb_dim += emb_dim

        self.input_dim = total_emb_dim + n_num_features

        enc_layers, prev = [], self.input_dim
        for dim in encoder_layers:
            enc_layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers, prev = [], bottleneck_dim
        for dim in decoder_layers:
            dec_layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        self.decoder_trunk = nn.Sequential(*dec_layers)
        self._decoder_out_dim = prev

        # Una head por columna categórica (logits) + una head para todas las numéricas
        self.cat_heads = nn.ModuleDict({name: nn.Linear(prev, n_cats) for name, n_cats, _ in cat_specs})
        self.num_head = nn.Linear(prev, n_num_features) if n_num_features > 0 else None

    @classmethod
    def from_encoder(cls, schema_encoder: SchemaEncoder, **kwargs) -> "TabularAE":
        """Crea el modelo con dimensiones derivadas del SchemaEncoder."""
        return cls(cat_specs=schema_encoder.cat_specs,
                   n_num_features=schema_encoder.n_num_features, **kwargs)

    def forward(
        self, cat_tensors: Dict[str, torch.LongTensor], num_tensor: torch.FloatTensor
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """→ (cat_logits[col→[B,n_cats]], num_recon[B,n_num] o None, z[B,bottleneck])."""
        emb_parts = [self.embeddings[name](cat_tensors[name]) for name, _, _ in self.cat_specs]
        x = torch.cat(emb_parts + ([num_tensor] if num_tensor.shape[1] > 0 else []), dim=1)
        z = self.encoder(x)
        h = self.decoder_trunk(z)
        cat_logits = {name: self.cat_heads[name](h) for name, _, _ in self.cat_specs}
        num_recon = self.num_head(h) if self.num_head is not None else None
        return cat_logits, num_recon, z

    def reconstruction_error(
        self, cat_tensors: Dict[str, torch.LongTensor], num_tensor: torch.FloatTensor
    ) -> Dict[str, torch.Tensor]:
        """Error de reconstrucción POR FEATURE: per_cat_col (CE/fila), per_num_col (MSE/feature), total (media/fila)."""
        cat_logits, num_recon, _ = self.forward(cat_tensors, num_tensor)
        errors, all_errors = {}, []

        per_cat = {}
        for name, _, _ in self.cat_specs:
            ce = F.cross_entropy(cat_logits[name], cat_tensors[name], reduction="none")
            per_cat[name] = ce
            all_errors.append(ce)
        errors["per_cat_col"] = per_cat

        if num_recon is not None and num_tensor.shape[1] > 0:
            mse_per_feature = (num_tensor - num_recon) ** 2
            errors["per_num_col"] = mse_per_feature
            all_errors.append(mse_per_feature.mean(dim=1))
        else:
            errors["per_num_col"] = torch.zeros(num_tensor.shape[0], 0)

        errors["total"] = torch.stack(all_errors, dim=1).mean(dim=1) if all_errors else torch.zeros(1)
        return errors


# ── VAE ───────────────────────────────────────────────────────────────────────

class TabularVAE(TabularAE):
    """Bottleneck estocástico (reparameterization trick). Score = recon_error + β·KL."""

    def __init__(self, *args, beta_kl: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_kl = beta_kl

        # El encoder de TabularAE termina en Linear(prev, bottleneck); lo partimos
        # en dos proyecciones (mu, logvar) sobre el penúltimo tamaño.
        enc_modules = list(self.encoder.children())
        last_linear = enc_modules[-1]
        assert isinstance(last_linear, nn.Linear)
        pre_bottleneck_dim = last_linear.in_features

        self.encoder = nn.Sequential(*enc_modules[:-1])
        self.fc_mu = nn.Linear(pre_bottleneck_dim, self.bottleneck_dim)
        self.fc_logvar = nn.Linear(pre_bottleneck_dim, self.bottleneck_dim)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu                                          # inferencia: usar mu directamente

    def forward(
        self, cat_tensors: Dict[str, torch.LongTensor], num_tensor: torch.FloatTensor
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        emb_parts = [self.embeddings[name](cat_tensors[name]) for name, _, _ in self.cat_specs]
        x = torch.cat(emb_parts + ([num_tensor] if num_tensor.shape[1] > 0 else []), dim=1)
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        h = self.decoder_trunk(z)
        cat_logits = {name: self.cat_heads[name](h) for name, _, _ in self.cat_specs}
        num_recon = self.num_head(h) if self.num_head is not None else None
        self._last_mu, self._last_logvar = mu, logvar      # para kl_divergence()
        return cat_logits, num_recon, z

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(z|x) || N(0,1)). Llamar tras forward()."""
        return -0.5 * torch.mean(1 + self._last_logvar - self._last_mu.pow(2) - self._last_logvar.exp())



# ── DAE ───────────────────────────────────────────────────────────────────────

class TabularDAE(TabularAE):
    """Denoising AE: corrupción del input durante entrenamiento, targets limpios.

    Arquitectura idéntica a TabularAE. La regularización viene de forzar al
    modelo a reconstruir el input ORIGINAL partiendo de una versión corrupta.

    Corrupción aplicada solo si self.training=True:
      - Categóricas: con prob `mask_prob`, índice → PAD_TOKEN (=1).
      - Numéricas:   ruido gaussiano N(0, noise_std) aditivo.

    En eval (scoring/inferencia) el forward es idéntico a TabularAE.
    Vincent et al. (2008), "Extracting and composing robust features with DAEs".
    """

    PAD_INDEX = 1   # debe coincidir con padding_idx del SchemaEncoder

    def __init__(self, *args, mask_prob: float = 0.15, noise_std: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = float(mask_prob)
        self.noise_std = float(noise_std)

    def _corrupt(
        self,
        cat_tensors: Dict[str, torch.LongTensor],
        num_tensor: torch.FloatTensor,
    ) -> Tuple[Dict[str, torch.LongTensor], torch.FloatTensor]:
        """Aplica masking categórico + jitter numérico. NO modifica los tensores in-place."""
        corrupted_cat = {}
        for name, _, _ in self.cat_specs:
            t = cat_tensors[name]
            if self.mask_prob > 0:
                mask = torch.rand_like(t, dtype=torch.float) < self.mask_prob
                t = torch.where(mask, torch.full_like(t, self.PAD_INDEX), t)
            corrupted_cat[name] = t

        if num_tensor.shape[1] > 0 and self.noise_std > 0:
            corrupted_num = num_tensor + torch.randn_like(num_tensor) * self.noise_std
        else:
            corrupted_num = num_tensor
        return corrupted_cat, corrupted_num

    def forward(
        self, cat_tensors: Dict[str, torch.LongTensor], num_tensor: torch.FloatTensor
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if self.training:
            cat_tensors, num_tensor = self._corrupt(cat_tensors, num_tensor)
        return super().forward(cat_tensors, num_tensor)


# ── Loss combinada ────────────────────────────────────────────────────────────

def combined_loss(
    cat_logits: Dict[str, torch.Tensor],
    cat_targets: Dict[str, torch.LongTensor],
    num_recon: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """α·MSE_num + (1-α)·mean(CE_cat). Devuelve (loss, breakdown)."""
    breakdown = {}

    ce_losses = []
    for col, logits in cat_logits.items():
        ce = F.cross_entropy(logits, cat_targets[col])
        ce_losses.append(ce)
        breakdown[f"ce_{col}"] = ce.item()
    mean_ce = torch.stack(ce_losses).mean() if ce_losses else torch.tensor(0.0)
    breakdown["mean_ce"] = mean_ce.item()

    if num_recon is not None and num_targets is not None and num_targets.shape[1] > 0:
        mse = F.mse_loss(num_recon, num_targets)
    else:
        mse = torch.tensor(0.0)
    breakdown["mse"] = mse.item()

    total = alpha * mse + (1 - alpha) * mean_ce
    breakdown["total"] = total.item()
    return total, breakdown


# ── Trainer ───────────────────────────────────────────────────────────────────

class RowLevelTrainer:
    """Entrena un TabularAE/TabularVAE con early stopping y ReduceLROnPlateau."""

    def __init__(self, model: TabularAE, schema_encoder: SchemaEncoder, cfg: dict, device: str = "auto"):
        self.model = model
        self.schema_encoder = schema_encoder
        self.cfg = cfg
        self.rl = cfg.get("row_level", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)
        self.model.to(self.device)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Dict[str, List[float]]:
        """Entrena sobre días normales. val_df habilita early stopping. → history {train_loss, val_loss}."""
        lr = self.rl.get("lr", 1e-3)
        batch_size = self.rl.get("batch_size", 4096)
        max_epochs = self.rl.get("max_epochs", 30)
        patience = self.rl.get("patience", 5)
        alpha = self.rl.get("alpha_loss", 0.5)
        is_vae = isinstance(self.model, TabularVAE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        train_loader = self._make_loader(train_df, batch_size, shuffle=True)
        val_loader = self._make_loader(val_df, batch_size, shuffle=False) if val_df is not None else None

        history = {"train_loss": [], "val_loss": []}
        best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

        log.info(f"[TRAIN] {type(self.model).__name__} | {len(train_df):,} train rows | "
                 f"device={self.device} | epochs={max_epochs} | batch={batch_size}")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, optimizer, alpha, is_vae, train=True)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, None, alpha, is_vae, train=False)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                val_loss = None

            val_str = f"val={val_loss:.6f}" if val_loss is not None else "no val"
            log.info(f"  Epoch {epoch:>3}/{max_epochs} | train={train_loss:.6f} | {val_str} | {time.time()-t0:.1f}s")

            if epochs_no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def save(self, model_path: Path, encoder_path: Optional[Path] = None):
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_class": type(self.model).__name__,
            "cat_specs": self.model.cat_specs,
            "n_num_features": self.model.n_num_features,
            "bottleneck_dim": self.model.bottleneck_dim,
        }, model_path)
        log.info(f"[TRAIN] Modelo → {model_path}")
        if encoder_path is not None:
            self.schema_encoder.save(encoder_path)

    # ── Internos ───────────────────────────────────────────────────────────────

    def _make_loader(self, df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        encoded = self.schema_encoder.transform(df)
        # num_tensor + categóricas apiladas en [batch, n_cat_cols]
        if encoded.cat_col_names:
            cat_stacked = torch.stack([encoded.cat_tensors[c] for c in encoded.cat_col_names], dim=1)
        else:
            cat_stacked = torch.zeros(encoded.n_rows, 0, dtype=torch.long)
        ds = TensorDataset(encoded.num_tensor, cat_stacked)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=self.device.type == "cuda")

    def _unpack_batch(self, batch: Tuple[torch.Tensor, ...]
                      ) -> Tuple[Dict[str, torch.LongTensor], torch.FloatTensor]:
        num_tensor, cat_stacked = batch[0].to(self.device), batch[1].to(self.device)
        cat_tensors = {col: cat_stacked[:, i] for i, col in enumerate(self.schema_encoder.cat_cols)}
        return cat_tensors, num_tensor

    def _run_epoch(self, loader: DataLoader, optimizer, alpha: float, is_vae: bool, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total_loss, n_batches = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                cat_tensors, num_tensor = self._unpack_batch(batch)
                cat_logits, num_recon, _ = self.model(cat_tensors, num_tensor)
                loss, _ = combined_loss(cat_logits, cat_tensors, num_recon, num_tensor, alpha)
                if is_vae:
                    loss = loss + self.model.beta_kl * self.model.kl_divergence()
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)
