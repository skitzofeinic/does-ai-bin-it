"""Tests voor klassen, split en voorbewerking. Draaien vanuit does-ai-bin-it/:

    python -m pytest tests
"""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (CLASSES, MEAN, STD, TRASHNET_TO_CLASS, eval_transform,  # noqa: E402
                        label_of, make_split, train_transform)

FOLDER_SIZES = {"cardboard": 40, "glass": 50, "metal": 41, "paper": 59, "plastic": 48, "trash": 14}
FILES = [f"{folder}/{folder}{i}.jpg" for folder, n in FOLDER_SIZES.items() for i in range(1, n + 1)]


def test_paper_and_cardboard_merge_into_one_class():
    assert label_of("paper/paper1.jpg") == label_of("cardboard/cardboard1.jpg") == 0
    assert sorted(set(TRASHNET_TO_CLASS.values())) == list(range(len(CLASSES)))


def test_split_is_disjoint_complete_and_deterministic():
    split = make_split(FILES)
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert sum(len(s) for s in sets) == len(FILES)
    assert set().union(*sets) == set(FILES)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert make_split(FILES) == split


def test_split_is_stratified_per_folder():
    split = make_split(FILES)
    for folder, n in FOLDER_SIZES.items():
        in_test = sum(f.startswith(folder + "/") for f in split["test"])
        assert abs(in_test - 0.15 * n) <= 1.5, folder


def test_eval_transform_crops_centre_square_and_normalises():
    # 4:3 zoals TrashNet: links en rechts wit, midden zwart. Het vierkant beslaat x = 64-448;
    # zwart loopt 8 px verder, zodat het schalen de rand niet in het uitsnijdsel mengt
    image = Image.new("RGB", (512, 384), "white")
    image.paste((0, 0, 0), (56, 0, 456, 384))
    x = eval_transform(128)(image)
    assert x.shape == (3, 128, 128)
    black = torch.tensor([-m / s for m, s in zip(MEAN, STD)]).view(3, 1, 1)
    assert torch.allclose(x, black.expand_as(x), atol=1e-5)  # het witte deel valt weg


def test_train_transform_keeps_shape():
    x = train_transform(128)(Image.new("RGB", (512, 384), "gray"))
    assert x.shape == (3, 128, 128)
