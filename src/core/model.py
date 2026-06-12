"""
model.py — TranAD con PyTorch Lightning
=========================================
Implementación de "TranAD: Deep Transformer Networks for Anomaly
Detection in Multivariate Time Series Data" (Tuli et al., 2022).

Arquitectura:
  Encoder compartido (Transformer, Pre-LN)
    ├─ Decoder W1 (Fase 1): reconstrucción directa
    └─ Decoder W2 (Fase 2): condicionado al error de W1

Loss de dos fases con α = 1/(phase2_epoch + 1):
  Fase 1: L = MaxMeanLoss(W1, x)
  Fase 2: L = α·L(W1) + (1-α)·L(W2)

Anomaly score (Ec. 6):
  score(t) = mean_features( |x-W1|² + |x-W2|² )
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Ventanas deslizantes: (seq_len, n_features) → (n_features,)
    El modelo predice el día siguiente dado seq_len días de contexto.
    """

    def __init__(self, data: np.ndarray, seq_len: int = 14):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx: idx + self.seq_len], self.data[idx + self.seq_len]


# ─────────────────────────────────────────────────────────────────────────────
# 2. BLOQUES TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(1)])


class SharedEncoder(nn.Module):
    """Encoder compartido (Pre-LN) entre W1 y W2."""

    def __init__(self, n_features, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(self.pos_enc(self.input_proj(x)))


class TranADDecoder(nn.Module):
    """
    W1: query_in_dim = n_features
    W2: query_in_dim = n_features * 2 (target || error_W1)
    """

    def __init__(self, query_in_dim, n_features, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.query_proj = nn.Linear(query_in_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder     = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, query, memory):
        q = self.pos_enc(self.query_proj(query))
        return self.output_proj(self.decoder(q, memory)).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELO TRANAD
# ─────────────────────────────────────────────────────────────────────────────

class TranAD(pl.LightningModule):

    def __init__(
        self,
        n_features:    int,
        d_model:       int   = 128,
        n_heads:       int   = 4,
        n_layers:      int   = 3,
        dropout:       float = 0.2,
        lr:            float = 3e-4,
        phase1_epochs: int   = 40,
        phase2_epochs: int   = 40,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_features    = n_features
        self.lr            = lr
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs

        self.encoder    = SharedEncoder(n_features, d_model, n_heads, n_layers, dropout)
        self.decoder_w1 = TranADDecoder(n_features,     n_features, d_model, n_heads, n_layers, dropout)
        self.decoder_w2 = TranADDecoder(n_features * 2, n_features, d_model, n_heads, n_layers, dropout)

    def forward(self, window, target):
        memory   = self.encoder(window)
        recon_w1 = self.decoder_w1(target.unsqueeze(1), memory)
        error_w1 = torch.abs(target - recon_w1).detach()
        query_w2 = torch.cat([target, error_w1], dim=-1).unsqueeze(1)
        recon_w2 = self.decoder_w2(query_w2, memory)
        return recon_w1, recon_w2

    def _max_mean_loss(self, reconstruction, target):
        """70% error medio + 30% error del peor feature."""
        err = (target - reconstruction) ** 2
        return 0.7 * err.mean() + 0.3 * err.max(dim=0).values.mean()

    def _compute_loss(self, recon_w1, recon_w2, target, epoch):
        loss_w1 = self._max_mean_loss(recon_w1, target)
        loss_w2 = self._max_mean_loss(recon_w2, target)
        if epoch < self.phase1_epochs:
            return loss_w1
        alpha = 1.0 / (epoch - self.phase1_epochs + 2)
        return alpha * loss_w1 + (1.0 - alpha) * loss_w2

    def training_step(self, batch, batch_idx):
        window, target = batch
        r1, r2 = self(window, target)
        loss = self._compute_loss(r1, r2, target, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        window, target = batch
        r1, r2 = self(window, target)
        loss = 0.5 * self._max_mean_loss(r1, target) + 0.5 * self._max_mean_loss(r2, target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer    = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        total_epochs = self.phase1_epochs + self.phase2_epochs
        warmup       = 5

        def lr_lambda(epoch):
            if epoch < warmup:
                return 0.1 + 0.9 * (epoch / warmup)
            progress = (epoch - warmup) / max(total_epochs - warmup, 1)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def compute_errors(self, window, target):
        """Returns (errors_w1, errors_w2) as numpy arrays of shape (batch, n_features)."""
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            r1, r2 = self(window.to(device), target.to(device))
            return ((target.to(device) - r1) ** 2).cpu().numpy(), \
                   ((target.to(device) - r2) ** 2).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CÁLCULO DE ERRORES
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_errors(model, scaled_data, cfg):
    """Returns (errors_w1, errors_w2) de shape (n_valid_days, n_features)."""
    seq_len = cfg["tranad"]["seq_len"]
    dataset = SlidingWindowDataset(scaled_data, seq_len=seq_len)
    loader  = DataLoader(dataset, batch_size=cfg["tranad"]["batch_size"], shuffle=False, num_workers=0)

    all_w1, all_w2 = [], []
    model.eval()
    with torch.no_grad():
        for window, target in loader:
            ew1, ew2 = model.compute_errors(window, target)
            all_w1.append(ew1)
            all_w2.append(ew2)
    return np.vstack(all_w1), np.vstack(all_w2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_tranad(scaled_data, feature_cols, cfg, train_mask):
    """Entrena TranAD con caché de checkpoint."""
    model_path = cfg["paths"]["tranad_model"]
    hp         = cfg["tranad"]
    n_features = len(feature_cols)

    if model_path.exists() and not cfg.get("force_retrain", False):
        log.info(f"[MODELO] Cargando desde {model_path}")
        model = TranAD.load_from_checkpoint(str(model_path), weights_only=False)
        model.eval()
        return model

    log.info(f"[MODELO] Entrenando — {n_features} features, seq_len={hp['seq_len']}")

    train_data = scaled_data[train_mask]
    val_data   = scaled_data[~train_mask]

    if len(val_data) <= hp["seq_len"]:
        split      = int(len(train_data) * 0.8)
        val_data   = train_data[split:]
        train_data = train_data[:split]

    train_ds = SlidingWindowDataset(train_data, seq_len=hp["seq_len"])
    val_ds   = SlidingWindowDataset(val_data,   seq_len=hp["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, num_workers=0)

    model = TranAD(
        n_features=n_features, d_model=hp["d_model"], n_heads=hp["n_heads"],
        n_layers=hp["n_layers"], dropout=hp["dropout"], lr=hp["lr"],
        phase1_epochs=hp["phase1_epochs"], phase2_epochs=hp["phase2_epochs"],
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(model_path.parent), filename="tranad_best",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=hp.get("patience", 15), mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=hp["phase1_epochs"] + hp["phase2_epochs"],
        callbacks=[checkpoint_cb, early_stop_cb, pl.callbacks.LearningRateMonitor("epoch")],
        enable_progress_bar=cfg.get("verbose", True),
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)
    best = TranAD.load_from_checkpoint(checkpoint_cb.best_model_path, weights_only=False)
    best.eval()
    log.info(f"[MODELO] Completado. Ckpt: {checkpoint_cb.best_model_path}")
    return best
