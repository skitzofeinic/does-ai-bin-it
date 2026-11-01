"""Experiment: hoe verandert de test-accuracy als je de foto's aanpast?

Eén aanpassing tegelijk, op alle 380 testfoto's, voor beide modellen. Geen training of
modelkeuze: de modellen liggen vast, dit meet alleen gevoeligheid.

Gebruik (vanuit docs/Anh/waste-classifier/):  python src/variations.py
Schrijft `results/variations.json`. De explainer doet dezelfde aanpassingen live in de browser.
"""

import json

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from evaluate import load_model
from models import IMAGE_SIZE
from preprocess import CLASSES, ROOT, data_dir, eval_transform, label_of, load_split

DARK = (40, 40, 40)
REFERENCE_WIDTH = 512  # vervaagstraal in pixels bij deze breedte; schaalt mee met de foto


def brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def rotate(image: Image.Image, degrees: float) -> Image.Image:
    """Tegen de klok in; hoeken vullen met wit, zoals de posterboard-achtergrond."""
    return image.rotate(degrees, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.BoxBlur(radius * image.width / REFERENCE_WIDTH))


def shrink_on_dark(image: Image.Image, scale: float) -> Image.Image:
    """Foto kleiner in het midden van een donkere ondergrond: ander decor, voorwerp kleiner."""
    if scale >= 1:
        return image
    canvas = Image.new("RGB", image.size, DARK)
    small = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    canvas.paste(small, ((image.width - small.width) // 2, (image.height - small.height) // 2))
    return canvas


EXPERIMENTS = {
    "licht": {"fn": brightness, "levels": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "neutral": 1.0,
              "unit": "x helderheid"},
    "draaien": {"fn": rotate, "levels": [0, 15, 30, 45, 90, 180], "neutral": 0, "unit": "graden"},
    "vervagen": {"fn": blur, "levels": [0, 2, 4, 8, 16], "neutral": 0, "unit": "px straal"},
    "kleiner op donker": {"fn": shrink_on_dark, "levels": [1.0, 0.75, 0.5, 0.35], "neutral": 1.0,
                          "unit": "x formaat"},
}


@torch.no_grad()
def accuracy(model, images: list[Image.Image], labels: np.ndarray, size: int, device: str) -> float:
    transform = eval_transform(size)
    preds = []
    for i in range(0, len(images), 64):
        x = torch.stack([transform(im) for im in images[i:i + 64]]).to(device)
        preds.append(model(x).argmax(1).cpu())
    return float((torch.cat(preds).numpy() == labels).mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = load_split()["test"]
    root = data_dir()
    originals = [Image.open(root / f).convert("RGB") for f in files]
    labels = np.array([label_of(f) for f in files])
    models = {name: load_model(name, device) for name in IMAGE_SIZE}

    out = {"n_test": len(files), "classes": CLASSES, "experiments": []}
    for name, exp in EXPERIMENTS.items():
        rows = []
        for level in exp["levels"]:
            changed = [exp["fn"](im, level) for im in originals]
            acc = {m: round(accuracy(models[m], changed, labels, IMAGE_SIZE[m], device), 4)
                   for m in models}
            rows.append({"level": level, "accuracy": acc})
            print(f"{name:18s} {level:>6}  " + "  ".join(f"{m} {a:.3f}" for m, a in acc.items()))
        out["experiments"].append({"name": name, "unit": exp["unit"], "neutral": exp["neutral"],
                                   "results": rows})
    (ROOT / "results" / "variations.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
