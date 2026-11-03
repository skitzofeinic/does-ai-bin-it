"""Dataset, klassen, split en voorbewerking voor de Waste Recycle Classifier.

Gedeeld door training, evaluatie en export, zodat overal precies dezelfde
klassen, split en voorbewerking gebruikt worden.

De dataset (TrashNet, `dataset-resized`) staat buiten de repository:
standaard `~/.cache/waste-classifier/dataset-resized`, aan te passen met
de omgevingsvariabele `WASTE_DATA`.

Gebruik (vanuit does-ai-bin-it/):  python src/preprocess.py
maakt `results/split.json` aan als die nog niet bestaat.
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "results" / "split.json"
SEED = 42

# Vijf klassen; papier en karton gaan in Nederland in dezelfde bak
CLASSES = ["papier en karton", "glas", "plastic", "metaal", "restafval"]

# De Nederlandse bakken: plastic en metaal (blik) gaan samen in PMD. De modellen leren de vijf
# materialen; de kans op een bak is de som van de kansen op de materialen erin.
BINS = ["papier en karton", "glas", "PMD", "restafval"]
CLASS_TO_BIN = [0, 1, 2, 2, 3]
TRASHNET_TO_CLASS = {
    "paper": 0,
    "cardboard": 0,
    "glass": 1,
    "plastic": 2,
    "metal": 3,
    "trash": 4,
}

# ImageNet-gemiddelden voor beide modellen, zodat de browser maar één codepad heeft
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def data_dir() -> Path:
    """Map met de zes TrashNet-submappen (cardboard, glass, ...)."""
    default = Path.home() / ".cache" / "waste-classifier" / "dataset-resized"
    path = Path(os.environ.get("WASTE_DATA", default))
    if not (path / "glass").is_dir():
        raise SystemExit(
            f"TrashNet niet gevonden in {path}. Pak dataset-resized.zip daar uit "
            "of zet WASTE_DATA op de juiste map."
        )
    return path


def list_images(root: Path) -> list[str]:
    """Alle foto's als paden relatief aan de datamap, bv. 'glass/glass1.jpg'."""
    return sorted(
        f"{folder}/{f.name}"
        for folder in TRASHNET_TO_CLASS
        for f in (root / folder).glob("*.jpg")
    )


def label_of(relpath: str) -> int:
    return TRASHNET_TO_CLASS[relpath.split("/")[0]]


def make_split(files: list[str], seed: int = SEED) -> dict[str, list[str]]:
    """70 / 15 / 15, gestratificeerd op de zes TrashNet-mappen.

    Op de mappen in plaats van de vijf klassen, zodat papier en karton elk
    evenredig over train, validatie en test verdeeld worden.
    """
    folders = [f.split("/")[0] for f in files]
    train, rest, _, rest_folders = train_test_split(
        files, folders, test_size=0.30, stratify=folders, random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest_folders, random_state=seed
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def load_split() -> dict[str, list[str]]:
    """Leest `results/split.json`; maakt hem eerst aan als hij nog niet bestaat."""
    if not SPLIT_FILE.exists():
        split = make_split(list_images(data_dir()))
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps({"seed": SEED, **split}, indent=1))
    data = json.loads(SPLIT_FILE.read_text())
    return {k: data[k] for k in ("train", "val", "test")}


def object_box(image: Image.Image, threshold: int = 40) -> tuple[int, int, int, int] | None:
    """Vierkant rond het voorwerp, met hele getallen zodat de browser precies hetzelfde vak vindt.

    TrashNet heeft overal een egale achtergrond: de achtergrondkleur is het gemiddelde van
    de randpixels, en elke pixel die daar in een kanaal meer dan `threshold` van afwijkt
    hoort bij het voorwerp. Het vak krijgt 20% marge, is minstens 30% van de korte zijde
    en hooguit de korte zijde. None als er niets gevonden wordt.
    """
    a = np.asarray(image, dtype=np.int32)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    background = border.sum(0) // len(border)
    foreground = np.abs(a - background).max(2) > threshold
    # Een rij of kolom telt pas mee vanaf 1% voorgrond, zodat losse ruispixels het vak niet oprekken
    rows = np.flatnonzero(foreground.sum(1) >= max(1, w // 100))
    cols = np.flatnonzero(foreground.sum(0) >= max(1, h // 100))
    if len(rows) == 0 or len(cols) == 0:
        return None
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    short = min(w, h)
    side = max(bottom - top, right - left)
    side = min(max(side + side // 5, short * 3 // 10), short)
    x0 = min(max((left + right) // 2 - side // 2, 0), w - side)
    y0 = min(max((top + bottom) // 2 - side // 2, 0), h - side)
    return int(x0), int(y0), int(x0 + side), int(y0 + side)


def object_crop(image: Image.Image) -> Image.Image:
    """Uitsnijden rond het voorwerp, zodat kleine voorwerpen niet in een paar pixels verdwijnen."""
    box = object_box(image)
    return image.crop(box) if box else image


def eval_transform(size: int, crop: bool = False) -> transforms.Compose:
    """Wat de browser ook doet: korte zijde naar `size`, midden uitsnijden, normaliseren.
    Met crop eerst uitsnijden rond het voorwerp (object_crop)."""
    return transforms.Compose([
        *([transforms.Lambda(object_crop)] if crop else []),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(size: int, erase: bool = False, crop: bool = False,
                    min_scale: float = 0.6) -> transforms.Compose:
    """Augmentatie: TrashNet is overal witte achtergrond en vergelijkbaar licht.
    Met erase worden willekeurige stukken weggevlakt, zodat het model niet op één
    glimplek of etiket leunt (plastic vs. glas). Met crop eerst uitsnijden rond het
    voorwerp, net als eval_transform. Een lagere min_scale zoomt verder in."""
    return transforms.Compose([
        *([transforms.Lambda(object_crop)] if crop else []),
        transforms.RandomResizedCrop(size, scale=(min_scale, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        *([transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))] if erase else []),
    ])


class WasteDataset(Dataset):
    """Foto's uit één split, met label 0-4."""

    def __init__(self, files: list[str], transform, root: Path | None = None):
        self.root = root or data_dir()
        self.files = files
        self.labels = [label_of(f) for f in files]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        image = Image.open(self.root / self.files[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def class_counts(files: list[str]) -> list[int]:
    counts = Counter(label_of(f) for f in files)
    return [counts[c] for c in range(len(CLASSES))]


if __name__ == "__main__":
    split = load_split()
    print(f"Split in {SPLIT_FILE}")
    for name, files in split.items():
        counts = ", ".join(f"{c} {n}" for c, n in zip(CLASSES, class_counts(files)))
        print(f"  {name:5s} {len(files):4d}  ({counts})")
