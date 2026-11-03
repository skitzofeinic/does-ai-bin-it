"""Zet de gegevens voor de explainer klaar (na train.py en evaluate.py).

Gebruik (vanuit does-ai-bin-it/):  python src/export_web.py

- `assets/examples/*.jpg`: testfoto's op 320 x 240 (bron: TrashNet), een paar per klasse
  plus de meest leerzame fouten;
- `explainer/data/web.json`: klassen, aantallen per split, de voorbeelden met de kansen
  die PyTorch er zelf voor geeft (referentie voor de browsercontrole), dezelfde kansen
  voor een synthetisch testbeeld zonder JPEG, en de filters van de eerste CNN-laag.
"""

import json

import numpy as np
import torch
from PIL import Image

from evaluate import load_model
from models import IMAGE_SIZE
from preprocess import (CLASSES, ROOT, SEED, class_counts, data_dir,
                        eval_transform, label_of, load_split, list_images)

EXAMPLES = ROOT / "assets" / "examples"
WEB = ROOT / "explainer" / "data" / "web.json"
EXAMPLE_SIZE = (320, 240)  # korte zijde 240 >= 224, dus nooit vergroten voor MobileNet
PER_CLASS = 2

# Fouten uit results/README.md die in de explainer terugkomen, met wat er te zien is
MISTAKES = {
    "paper/paper93.jpg": "Tijdschriftpagina vol kleurige plaatjes",
    "glass/glass21.jpg": "Doorzichtige glazen pot met reflecties",
    "plastic/plastic36.jpg": "Doorzichtige plastic fles",
    "metal/metal402.jpg": "Blik met papieren etiket",
    "plastic/plastic455.jpg": "Gewatteerde envelop: papier met plastic noppen",
    "trash/trash89.jpg": "Tissue, gelabeld als restafval",
}


@torch.no_grad()
def probabilities(models, image: Image.Image) -> dict[str, list[float]]:
    return {
        name: [round(p, 6) for p in torch.softmax(
            model(eval_transform(IMAGE_SIZE[name])(image).unsqueeze(0)), 1)[0].tolist()]
        for name, model in models.items()
    }


def pick_examples(models, test: list[str]) -> list[str]:
    """Per klasse PER_CLASS testfoto's die beide modellen goed hebben (vaste seed)."""
    rng = np.random.default_rng(SEED)
    root, picked = data_dir(), []
    for c in range(len(CLASSES)):
        candidates = [f for f in test if label_of(f) == c and f not in MISTAKES]
        for f in rng.permutation(candidates):
            probs = probabilities(models, Image.open(root / f).convert("RGB"))
            if all(int(np.argmax(p)) == c for p in probs.values()):
                picked.append(str(f))
            if len(picked) == (c + 1) * PER_CLASS:
                break
    return picked


def synthetic(w: int = 320, h: int = 240) -> Image.Image:
    """Testbeeld uit een formule; explainer/app.js bouwt precies hetzelfde beeld.

    Zo vergelijkt de browsercontrole het model zonder JPEG-decodering ertussen
    (browsers openen JPEG net iets anders dan Pillow).
    """
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([(x * 255) // (w - 1), (y * 255) // (h - 1), (x * 7 + y * 13) % 256], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


def conv1(model) -> dict:
    """Eerste convolutie met BatchNorm erin gevouwen: relu(w * x + b) is de echte activatie."""
    conv, bn = model.features[0][0], model.features[0][1]
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.view(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * scale
    return {"shape": list(weight.shape),
            "weight": [round(w, 6) for w in weight.flatten().tolist()],
            "bias": [round(b, 6) for b in bias.tolist()]}


def main():
    models = {name: load_model(name) for name in IMAGE_SIZE}
    split = load_split()
    root = data_dir()
    EXAMPLES.mkdir(parents=True, exist_ok=True)

    examples = []
    for f in pick_examples(models, split["test"]) + list(MISTAKES):
        assert f in split["test"], f
        name = f.split("/")[1]
        image = Image.open(root / f).convert("RGB").resize(EXAMPLE_SIZE, Image.LANCZOS)
        image.save(EXAMPLES / name, quality=90)
        # Referentie op het opgeslagen bestand: precies wat de browser ook inleest
        saved = Image.open(EXAMPLES / name).convert("RGB")
        examples.append({"file": name, "label": label_of(f), "source": f,
                         "note": MISTAKES.get(f), "probabilities": probabilities(models, saved)})

    folders = {}
    for f in list_images(root):
        folders[f.split("/")[0]] = folders.get(f.split("/")[0], 0) + 1
    data = {
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "trashnet_folders": folders,
        "split_counts": {k: class_counts(v) for k, v in split.items()},
        "examples": examples,
        "synthetic": {"w": 320, "h": 240, "probabilities": probabilities(models, synthetic())},
        "conv1": conv1(models["cnn"]),
    }
    WEB.parent.mkdir(parents=True, exist_ok=True)
    WEB.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(examples)} voorbeelden in {EXAMPLES}, gegevens in {WEB} "
          f"({WEB.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
