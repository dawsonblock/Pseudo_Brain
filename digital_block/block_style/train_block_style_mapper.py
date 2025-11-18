# digital_block/block_style/train_block_style_mapper.py

#!/usr/bin/env python3
"""
Train BlockStyleMapper on JSONL with:
  - affect_generic
  - block_labels
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .block_style_mapper import BlockStyleMapper, AFFECT_KEYS, BLOCK_KEYS


class BlockStyleDataset(Dataset):
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
                if not affect or not labels:
                    continue

                x = [float(affect.get(k, 0.0)) for k in AFFECT_KEYS]
                y = [float(labels.get(k, 0.5)) for k in BLOCK_KEYS]

                self.samples.append(
                    {
                        "x": torch.tensor(x, dtype=torch.float32),
                        "y": torch.tensor(y, dtype=torch.float32),
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No valid samples in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def train(
    data_path: Path,
    out_path: Path,
    batch_size: int,
    lr: float,
    epochs: int,
    device: str,
) -> None:
    dataset = BlockStyleDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BlockStyleMapper().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    print(f"Training BlockStyleMapper on {len(dataset)} samples, device={device}")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            loss = mse(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        print(f"[Epoch {epoch:03d}] loss={total_loss / len(dataset):.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved BlockStyleMapper weights to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    train(
        data_path=Path(args.data),
        out_path=Path(args.out),
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
