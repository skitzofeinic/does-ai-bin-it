"""De browservoorbewerking (explainer/preprocess.js) moet gelijk zijn aan eval_transform.

Draait het JavaScript via Node; overgeslagen als Node niet geïnstalleerd is.
"""

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import eval_transform  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "js_preprocess.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node niet gevonden")


def run_js(pixels: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = pixels.shape
    payload = {"rgb": base64.b64encode(pixels.tobytes()).decode(), "w": w, "h": h, "size": size}
    result = subprocess.run(["node", str(SCRIPT)], input=json.dumps(payload),
                            capture_output=True, text=True, check=True)
    out = json.loads(result.stdout)
    crop = np.frombuffer(base64.b64decode(out["crop"]), dtype=np.uint8).reshape(size, size, 3)
    return crop, np.array(out["tensor"], dtype=np.float32).reshape(3, size, size)


# 4:3 zoals TrashNet, staand, vierkant, en een breedte waarbij het uitsnijden op .5 valt
@pytest.mark.parametrize("w,h,size", [(320, 240, 128), (320, 240, 224), (240, 320, 128),
                                      (200, 200, 128), (99, 61, 32), (512, 384, 128)])
def test_js_matches_pillow_and_torchvision(w, h, size):
    rng = np.random.default_rng(w * h + size)
    # Vloeiend beeld met ruis: lijkt meer op een foto dan pure ruis
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 / w, yy * 255 / h, (xx + yy) * 127 / (w + h)], axis=-1)
    pixels = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)

    crop, tensor = run_js(pixels, size)
    expected = eval_transform(size)(image).numpy()
    # Het vierkante beeld moet bit voor bit gelijk zijn; de tensor op float32-afronding na
    expected_crop = np.asarray(eval_transform(size).transforms[1](
        eval_transform(size).transforms[0](image)))
    assert np.array_equal(crop, expected_crop)
    assert np.allclose(tensor, expected, atol=1e-5)
    assert torch.from_numpy(tensor).shape == (3, size, size)
