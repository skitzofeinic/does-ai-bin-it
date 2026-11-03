"""Train een model op de train-split en kies de beste epoch op validatie.

De testset wordt hier niet aangeraakt; dat doet evaluate.py, één keer per model.

Gebruik (vanuit does-ai-bin-it/):
    python src/train.py --model cnn
    python src/train.py --model mobilenet

Schrijft `model/<naam>.pt` (niet in git) en `results/history_<naam>.json`.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from models import IMAGE_SIZE, build
from preprocess import (CLASSES, ROOT, SEED, WasteDataset, class_counts,
                        eval_transform, load_split, train_transform)

DEFAULTS = {
    # Val bal. acc: 40 epochs 0,727 (onderfit), 100 epochs 0,784, 200 epochs 0,796 (vlakt af)
    "cnn": {"epochs": 200, "lr": 2e-3},
    # 15 epochs 0,907; 30 epochs 0,903, dus langer helpt niet
    "mobilenet": {"epochs": 15, "lr": 5e-4},
    # DABI 1.2: een blok extra en RandomErasing, vanaf nul met dezelfde instellingen als cnn.
    # Verder trainen vanaf cnn (30 epochs, lr 3e-4 / 1e-4 / 5e-5) bleef rond 0,79-0,81 hangen.
    "cnn-1.2": {"epochs": 200, "lr": 2e-3, "erase": True},
}

PLASTIC, GLASS = CLASSES.index("plastic"), CLASSES.index("glas")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(files: list[str]) -> torch.Tensor:
    """Omgekeerd evenredig met het aantal trainingsfoto's, gemiddeld 1."""
    counts = torch.tensor(class_counts(files), dtype=torch.float)
    weights = counts.sum() / (len(counts) * counts)
    return weights / weights.mean()


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Eén keer door de loader. Met optimizer wordt er getraind."""
    model.train(optimizer is not None)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": total_loss / len(labels),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        # De meest voorkomende verwarring van DABI 1.1
        "plastic_as_glass": int(((labels == PLASTIC) & (preds == GLASS)).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=DEFAULTS, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--erase", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    epochs = args.epochs or DEFAULTS[args.model]["epochs"]
    lr = args.lr or DEFAULTS[args.model]["lr"]
    erase = DEFAULTS[args.model].get("erase", False) if args.erase is None else args.erase

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = IMAGE_SIZE[args.model]
    split = load_split()

    loaders = {
        "train": DataLoader(WasteDataset(split["train"], train_transform(size, erase)),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0),
        "val": DataLoader(WasteDataset(split["val"], eval_transform(size)),
                          batch_size=64, num_workers=args.workers,
                          persistent_workers=args.workers > 0),
    }
    weights = class_weights(split["train"])
    model = build(args.model)
    init = DEFAULTS[args.model].get("init")
    if init:
        model.load_state_dict(torch.load(ROOT / "model" / f"{init}.pt", map_location="cpu"))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(split['train'])} train / {len(split['val'])} val, {device}"
          + (f", start vanaf {init}" if init else ""))
    history, best = [], None
    if init:
        # Epoch 0: het startpunt, zodat een verbetering ten opzichte van de vorige versie zichtbaar is
        start_val = run_epoch(model, loaders["val"], loss_fn, device)
        best = {"epoch": 0, "val": start_val}
        print(f"  0  start                              val loss {start_val['loss']:.3f} "
              f"acc {start_val['accuracy']:.3f} bal {start_val['balanced_accuracy']:.3f} "
              f"plastic->glas {start_val['plastic_as_glass']}")
    start = time.time()
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val = run_epoch(model, loaders["val"], loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train": train, "val": val})
        # Beste epoch op gebalanceerde accuracy: gewone accuracy beloont het negeren van restafval
        if best is None or val["balanced_accuracy"] > best["val"]["balanced_accuracy"]:
            best = history[-1]
            (ROOT / "model").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "model" / f"{args.model}.pt")
        print(f"{epoch:3d}  train loss {train['loss']:.3f} acc {train['accuracy']:.3f}  "
              f"val loss {val['loss']:.3f} acc {val['accuracy']:.3f} "
              f"bal {val['balanced_accuracy']:.3f} plastic->glas {val['plastic_as_glass']}", flush=True)

    out = {
        "model": args.model,
        "init": init,
        "classes": CLASSES,
        "image_size": size,
        "epochs": epochs,
        "lr": lr,
        "label_smoothing": args.label_smoothing,
        "random_erasing": erase,
        "batch_size": args.batch_size,
        "class_weights": [round(w, 4) for w in weights.tolist()],
        "seed": SEED,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seconds": round(time.time() - start, 1),
        "best_epoch": best["epoch"],
        "history": history,
    }
    (ROOT / "results" / f"history_{args.model}.json").write_text(json.dumps(out, indent=1))
    print(f"beste epoch {best['epoch']}: val bal. acc {best['val']['balanced_accuracy']:.3f}, "
          f"acc {best['val']['accuracy']:.3f} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
