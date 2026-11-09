"""Zet `model/cnn.onnx` terug om naar een PyTorch-checkpoint, als startpunt voor verder trainen.

Gebruik (vanuit does-ai-bin-it/):  python src/onnx_to_checkpoint.py

De ONNX-export heeft elke BatchNorm in de convolutie ervoor gevouwen: per kanaal
W' = W * s en b = beta - mean * s. Voor voorspellen is dat gelijk, maar de losse
BatchNorm-waarden zijn weg. Die bouwen we zo terug dat de uitvoer exact gelijk blijft:

- conv krijgt W', en BatchNorm met mean m en var v van W'x op de trainingsfoto's
  (met augmentatie, zoals bij het oorspronkelijke trainen), gamma = sqrt(v + eps)
  en beta = m + b. In eval-modus geeft dat precies W'x + b;
- in train-modus normaliseert BatchNorm met de batch, en omdat m en v de echte
  statistieken zijn, blijft de uitvoer dicht bij die van het getrainde model.

Schrijft `model/cnn.pt` (niet in git) en controleert dat het op de testset
hetzelfde voorspelt als de ONNX-versie. Alleen nodig als het originele
checkpoint ontbreekt; een bestaand `model/cnn.pt` wordt niet overschreven.
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from torch.utils.data import DataLoader

from models import IMAGE_SIZE, WasteCNN
from preprocess import ROOT, SEED, WasteDataset, eval_transform, load_split, train_transform


def onnx_weights(path) -> dict[str, torch.Tensor]:
    graph = onnx.load(path).graph
    return {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in graph.initializer}


@torch.no_grad()
def from_onnx(path) -> WasteCNN:
    w = onnx_weights(path)
    model = WasteCNN().eval()
    convs = [model.features[i][0] for i in range(4)]
    norms = [model.features[i][1] for i in range(4)]

    # Stap 1: BatchNorm als doorgeefluik (mean 0, var 1 - eps, gamma 1), beta = gevouwen bias
    for i, (conv, bn) in enumerate(zip(convs, norms)):
        conv.weight.copy_(w[f"0.features.{i}.0.weight"])
        bn.running_mean.zero_()
        bn.running_var.fill_(1 - bn.eps)
        bn.weight.fill_(1)
        bn.bias.copy_(w[f"0.features.{i}.0.weight_bias"])
    model.head[3].weight.copy_(w["0.head.3.weight"])
    model.head[3].bias.copy_(w["0.head.3.bias"])

    # Stap 2: mean en var van W'x per kanaal meten op de trainingsfoto's
    sums = [torch.zeros(c.out_channels, dtype=torch.float64) for c in convs]
    squares = [torch.zeros(c.out_channels, dtype=torch.float64) for c in convs]
    counts = [0] * 4

    def hook(i):
        def collect(_, __, out):
            x = out.double().transpose(0, 1).flatten(1)
            sums[i] += x.sum(1)
            squares[i] += (x * x).sum(1)
            counts[i] += x.shape[1]
        return collect

    handles = [conv.register_forward_hook(hook(i)) for i, conv in enumerate(convs)]
    torch.manual_seed(SEED)
    files = load_split()["train"]
    for x, _ in DataLoader(WasteDataset(files, train_transform(IMAGE_SIZE["cnn"])), batch_size=64):
        model(x)
    for h in handles:
        h.remove()

    # Stap 3: BatchNorm zo invullen dat eval-uitvoer gelijk blijft
    for i, bn in enumerate(norms):
        mean = sums[i] / counts[i]
        var = squares[i] / counts[i] - mean * mean
        bias = bn.bias.clone()
        bn.running_mean.copy_(mean.float())
        bn.running_var.copy_(var.float())
        bn.weight.copy_(torch.sqrt(var + bn.eps).float())
        bn.bias.copy_(mean.float() + bias)
    return model


def main():
    if (ROOT / "model" / "cnn.pt").exists():
        raise SystemExit("model/cnn.pt bestaat al; het origineel is beter dan een teruggerekende versie")
    path = ROOT / "model" / "cnn.onnx"
    model = from_onnx(path)

    files = load_split()["test"]
    images = torch.cat([x for x, _ in DataLoader(WasteDataset(files, eval_transform(IMAGE_SIZE["cnn"])),
                                                  batch_size=64)])
    with torch.no_grad():
        ours = torch.softmax(model(images), 1).numpy()
    onnx_out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"image": images.numpy()})[0]
    diff = float(np.abs(ours - onnx_out).max())
    same = int((ours.argmax(1) == onnx_out.argmax(1)).sum())
    print(f"max verschil met ONNX {diff:.2e}, zelfde voorspelling {same}/{len(files)}")
    if same != len(files) or diff > 1e-4:
        raise SystemExit("checkpoint wijkt af van de ONNX-versie, niet opgeslagen")

    torch.save(model.state_dict(), ROOT / "model" / "cnn.pt")
    print("geschreven: model/cnn.pt")


if __name__ == "__main__":
    main()
