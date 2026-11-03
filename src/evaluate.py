"""Evalueer een getraind model op de testset. Eén keer per model, na de training.

Gebruik (vanuit does-ai-bin-it/):
    python src/evaluate.py --model cnn
    python src/evaluate.py --model mobilenet
    python src/evaluate.py --duplicates      # schatting bijna-duplicaten train/test

Schrijft `results/evaluation_<naam>.json`: accuracy, precision/recall/F1 per klasse,
confusion matrix, zekerheid bij goed en fout, en alle foute voorspellingen.
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader

from models import IMAGE_SIZE, build
from preprocess import (CLASSES, ROOT, WasteDataset, data_dir, eval_transform,
                        load_split)

CONFIDENT = 0.8  # "zeker" voor de analyse van overtuigende fouten


def load_model(name: str, device: str = "cpu") -> torch.nn.Module:
    model = build(name)
    model.load_state_dict(torch.load(ROOT / "model" / f"{name}.pt", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, files: list[str], size: int, device: str) -> np.ndarray:
    """Kansen per klasse (N, 5) voor een lijst foto's."""
    loader = DataLoader(WasteDataset(files, eval_transform(size)), batch_size=64)
    return torch.cat([
        torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader
    ]).numpy()


def evaluate(name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = load_split()["test"]
    labels = np.array(WasteDataset(files, None).labels)
    probs = predict(load_model(name, device), files, IMAGE_SIZE[name], device)
    preds, conf = probs.argmax(1), probs.max(1)
    correct = preds == labels

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(CLASSES)), zero_division=0)
    mistakes = sorted(
        ({"file": files[i], "true": CLASSES[labels[i]], "predicted": CLASSES[preds[i]],
          "confidence": round(float(conf[i]), 4)} for i in np.where(~correct)[0]),
        key=lambda m: -m["confidence"])
    return {
        "model": name,
        "classes": CLASSES,
        "n_test": len(files),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "per_class": [
            {"class": c, "precision": round(float(p), 4), "recall": round(float(r), 4),
             "f1": round(float(f), 4), "support": int(s)}
            for c, p, r, f, s in zip(CLASSES, precision, recall, f1, support)
        ],
        # rij = echte klasse, kolom = voorspelde klasse
        "confusion_matrix": confusion_matrix(labels, preds, labels=range(len(CLASSES))).tolist(),
        "mean_confidence_correct": round(float(conf[correct].mean()), 4),
        "mean_confidence_wrong": round(float(conf[~correct].mean()), 4) if (~correct).any() else None,
        "confident_mistakes": int(((~correct) & (conf >= CONFIDENT)).sum()),
        "confident_threshold": CONFIDENT,
        "mistakes": mistakes,
    }


def dhash(path, size: int = 8) -> int:
    """Verschil-hash: grijs, verkleind tot (size+1) x size, bit per horizontale buurvergelijking."""
    pixels = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(max_distance: int = 4) -> dict:
    """Ruwe schatting: testfoto's met een trainfoto uit dezelfde map die visueel bijna gelijk is.

    Een dHash-afstand van maximaal 4 (op 64 bits) duidt op bijna dezelfde opname. Op witte
    achtergrond geeft dit ook valse treffers; het is een indicatie, geen bewijs.
    """
    root, split = data_dir(), load_split()
    train = {f: dhash(root / f) for f in split["train"]}
    pairs = []
    for f in split["test"]:
        h, folder = dhash(root / f), f.split("/")[0]
        best = min(((bin(h ^ th).count("1"), t) for t, th in train.items()
                    if t.startswith(folder + "/")))
        if best[0] <= max_distance:
            pairs.append({"test": f, "train": best[1], "distance": best[0]})
    return {"method": "dHash 8x8, zelfde map", "max_distance": max_distance,
            "n_test": len(split["test"]), "n_flagged": len(pairs), "pairs": pairs}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=IMAGE_SIZE)
    group.add_argument("--duplicates", action="store_true")
    args = parser.parse_args()

    if args.duplicates:
        out = near_duplicates()
        (ROOT / "results" / "near_duplicates.json").write_text(json.dumps(out, indent=1))
        print(f"{out['n_flagged']} van {out['n_test']} testfoto's lijken sterk op een trainfoto")
        return

    out = evaluate(args.model)
    (ROOT / "results" / f"evaluation_{args.model}.json").write_text(json.dumps(out, indent=1))
    print(f"{args.model}: accuracy {out['accuracy']:.3f}, gebalanceerd {out['balanced_accuracy']:.3f}")
    for row in out["per_class"]:
        print(f"  {row['class']:17s} P {row['precision']:.2f}  R {row['recall']:.2f}  "
              f"F1 {row['f1']:.2f}  (n={row['support']})")
    print("  confusion (rij = echt, kolom = voorspeld):")
    for c, row in zip(CLASSES, out["confusion_matrix"]):
        print(f"    {c:17s} {row}")
    print(f"  zekerheid goed {out['mean_confidence_correct']:.2f}, fout {out['mean_confidence_wrong']}, "
          f"{out['confident_mistakes']} fouten met >= {CONFIDENT:.0%} zekerheid")


if __name__ == "__main__":
    main()
