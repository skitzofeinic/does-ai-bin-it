"""Scores per Nederlandse bak (papier, glas, PMD, rest) in plaats van per materiaal.

De modellen kennen vijf materialen; plastic en metaal gaan samen in PMD. De kans op een bak is
de som van de kansen op de materialen erin, net als in de browser. Gerekend met de ONNX-modellen
(model/<naam>.onnx), dezelfde bestanden als de site, zodat er geen PyTorch-checkpoints nodig zijn.

Gebruik (vanuit does-ai-bin-it/):  python src/evaluate_bins.py

Schrijft per model `results/evaluation_bins_<naam>.json` (zelfde opbouw als evaluate.py)
en `results/variations_bins.json` (zelfde opbouw als variations.py).
"""

import json

import numpy as np
import onnxruntime as ort
from PIL import Image

from evaluate import summarize
from models import IMAGE_SIZE
from preprocess import BINS, CLASS_TO_BIN, ROOT, data_dir, eval_transform, label_of, load_split
from variations import EXPERIMENTS

TO_BINS = np.zeros((len(CLASS_TO_BIN), len(BINS)), dtype=np.float32)
TO_BINS[range(len(CLASS_TO_BIN)), CLASS_TO_BIN] = 1  # (5, 4): materiaalkansen @ TO_BINS = bakkansen


def bin_probs(session: ort.InferenceSession, images: list[Image.Image], size: int) -> np.ndarray:
    transform = eval_transform(size)
    out = []
    for i in range(0, len(images), 64):
        x = np.stack([transform(im).numpy() for im in images[i:i + 64]])
        out.append(session.run(None, {"image": x})[0])
    return np.concatenate(out) @ TO_BINS


def main():
    files = load_split()["test"]
    root = data_dir()
    originals = [Image.open(root / f).convert("RGB") for f in files]
    labels = np.array([CLASS_TO_BIN[label_of(f)] for f in files])
    sessions = {name: ort.InferenceSession(str(ROOT / "model" / f"{name}.onnx"), providers=["CPUExecutionProvider"])
                for name in IMAGE_SIZE}

    for name, session in sessions.items():
        out = summarize(name, files, labels, bin_probs(session, originals, IMAGE_SIZE[name]), BINS)
        (ROOT / "results" / f"evaluation_bins_{name}.json").write_text(json.dumps(out, indent=1))
        print(f"{name}: juiste bak {out['accuracy']:.3f}, gebalanceerd {out['balanced_accuracy']:.3f}, "
              f"{out['confident_mistakes']} fouten met >= 80% zekerheid")

    variations = {"n_test": len(files), "classes": BINS, "experiments": []}
    for exp_name, exp in EXPERIMENTS.items():
        rows = []
        for level in exp["levels"]:
            changed = [exp["fn"](im, level) for im in originals]
            acc = {name: round(float((bin_probs(s, changed, IMAGE_SIZE[name]).argmax(1) == labels).mean()), 4)
                   for name, s in sessions.items()}
            rows.append({"level": level, "accuracy": acc})
            print(f"{exp_name:18s} {level:>6}  " + "  ".join(f"{m} {a:.3f}" for m, a in acc.items()))
        variations["experiments"].append({"name": exp_name, "unit": exp["unit"], "neutral": exp["neutral"], "results": rows})
    (ROOT / "results" / "variations_bins.json").write_text(json.dumps(variations, indent=1))


if __name__ == "__main__":
    main()
