# digital_block/emotion/emotion_policy_trainer.py

#!/usr/bin/env python3
"""
Train EmotionPolicyNet:
  (affect, block_labels, U_e/U_a, traits) -> E (16D), block_pred, future_loss_pred
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from digital_block.block_style.block_style_mapper import AFFECT_KEYS, BLOCK_KEYS
from digital_block_profile import get_default_trait_vector  # use real profile


class EmotionPolicyDataset(Dataset):
    def __init__(self, jsonl_path: Path) -> None:
        self.samples: List[Dict[str, Any]] = []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                affect = obj.get("affect_generic")
                labels = obj.get("block_labels")
                signals = obj.get("model_signals") or {}

                if not affect or not labels:
                    continue

                future_loss = signals.get("future_loss", None)
                if future_loss is None:
                    continue

                x_aff = [float(affect.get(k, 0.0)) for k in AFFECT_KEYS]
                x_blk = [float(labels.get(k, 0.5)) for k in BLOCK_KEYS]
                u_e = float(signals.get("U_e", 0.0))
                u_a = float(signals.get("U_a", 0.0))
                x_ctx = [u_e, u_a]

                traits_vec = get_default_trait_vector().to_tensor().tolist()

                x_vec = x_aff + x_blk + x_ctx + traits_vec

                self.samples.append(
                    {
                        "x": torch.tensor(x_vec, dtype=torch.float32),
                        "y_block": torch.tensor(
                            [float(labels.get(k, 0.5)) for k in BLOCK_KEYS],
                            dtype=torch.float32,
                        ),
                        "y_future": torch.tensor(float(future_loss), dtype=torch.float32),
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No valid samples in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class EmotionPolicyNet(nn.Module):
    def __init__(self, input_dim: int, emotion_dim: int = 16) -> None:
        super().__init__()
        hidden = 64
        self.emotion_dim = emotion_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, emotion_dim),
            nn.Tanh(),
        )

        self.block_head = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(BLOCK_KEYS)),
            nn.Sigmoid(),
        )

        self.future_head = nn.Sequential(
            nn.Linear(emotion_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        E = self.emotion_head(h)
        block_pred = self.block_head(E)
        future_pred = self.future_head(E).squeeze(-1)
        return E, block_pred, future_pred


def train(
    data_path: Path,
    out_path: Path,
    batch_size: int,
    lr: float,
    epochs: int,
    lambda_block: float,
    lambda_future: float,
    device: str,
) -> None:
    dataset = EmotionPolicyDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample = dataset[0]
    input_dim = sample["x"].shape[0]

    model = EmotionPolicyNet(input_dim=input_dim, emotion_dim=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    print(f"Training EmotionPolicyNet on {len(dataset)} samples, input_dim={input_dim}, device={device}")
    for epoch in range(1, epochs + 1):
        total, total_b, total_f = 0.0, 0.0, 0.0
        for batch in loader:
            x = batch["x"].to(device)
            y_block = batch["y_block"].to(device)
            y_future = batch["y_future"].to(device)

            E, block_pred, future_pred = model(x)

            L_block = mse(block_pred, y_block)
            L_future = mse(future_pred, y_future)
            loss = lambda_block * L_block + lambda_future * L_future

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += loss.item() * bs
            total_b += L_block.item() * bs
            total_f += L_future.item() * bs

        n = len(dataset)
        print(
            f"[Epoch {epoch:03d}] loss={total/n:.6f} "
            f"L_block={total_b/n:.6f} L_future={total_f/n:.6f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved EmotionPolicyNet weights to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lambda_block", type=float, default=1.0)
    p.add_argument("--lambda_future", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    train(
        data_path=Path(args.data),
        out_path=Path(args.out),
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        lambda_block=args.lambda_block,
        lambda_future=args.lambda_future,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
