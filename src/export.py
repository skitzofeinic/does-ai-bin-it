"""Exporteer de getrainde modellen naar ONNX en controleer dat ONNX hetzelfde voorspelt.

Gebruik (vanuit does-ai-bin-it/):  python src/export.py [--model cnn-1.2]

- `model/<naam>.onnx`: invoer `image` (N, 3, S, S, genormaliseerd), uitvoer `probabilities` (N, 5);
- `results/parity_onnx.json`: verschil PyTorch vs ONNX Runtime (Python) op de hele testset.

De vergelijking met de browser (ONNX Runtime Web) volgt in de explainer.
"""

import argparse
import json

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate import load_model
from models import IMAGE_SIZE
from preprocess import ROOT, WasteDataset, eval_transform, load_split


def export(name: str) -> dict:
    model = nn.Sequential(load_model(name), nn.Softmax(dim=1)).eval()
    size = IMAGE_SIZE[name]
    path = ROOT / "model" / f"{name}.onnx"
    torch.onnx.export(
        model, (torch.zeros(1, 3, size, size),), path,
        input_names=["image"], output_names=["probabilities"],
        dynamic_shapes={"input": {0: torch.export.Dim("batch")}},
        external_data=False,
        verbose=False,  # de voortgangsmeldingen bevatten emoji die een Windows-console (cp1252) niet aankan
    )

    files = load_split()["test"]
    images = torch.cat([x for x, _ in DataLoader(WasteDataset(files, eval_transform(size)), batch_size=64)])
    with torch.no_grad():
        expected = model(images).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": images.numpy()})[0]
    return {
        "model": name,
        "onnx_bytes": path.stat().st_size,
        "n_test": len(files),
        "max_abs_diff": float(np.abs(actual - expected).max()),
        "same_prediction": int((actual.argmax(1) == expected.argmax(1)).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    # Eén model exporteren; de andere regels in parity_onnx.json blijven staan
    parser.add_argument("--model", choices=IMAGE_SIZE)
    args = parser.parse_args()
    parity = ROOT / "results" / "parity_onnx.json"
    if args.model:
        out = [r for r in json.loads(parity.read_text()) if r["model"] != args.model] + [export(args.model)]
        out.sort(key=lambda r: list(IMAGE_SIZE).index(r["model"]))
    else:
        out = [export(name) for name in IMAGE_SIZE]
    parity.write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['model']}: {r['onnx_bytes'] / 1e6:.1f} MB, max verschil {r['max_abs_diff']:.2e}, "
              f"zelfde voorspelling {r['same_prediction']}/{r['n_test']}")


if __name__ == "__main__":
    main()
