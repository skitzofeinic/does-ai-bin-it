# Does AI bin it?

An interactive explainer that shows how an image classifier sorts waste, and where it goes wrong. You sort the photos yourself first, then see what two real models chose: a compact CNN trained from scratch and a pretrained MobileNetV3-Small for comparison. Both run entirely in the browser.

## What's on the page

1. **Sort along**: eight test photos. Pick a bin, then see what both models said, with a running score. You can also upload your own photo.
2. **How it sees**: from photo to 49,152 numbers, through the 32 real filters of the CNN's first layer, to five percentages. Runs live on the chosen photo.
3. **Everything it saw**: all 2,527 training photos per bin, the split and the learning curves.
4. **Results**: all 380 test photos per class (right, wrong or confidently wrong), per-class accuracy for both models and the most common mix-ups.
5. **Break it**: change light, angle, blur and background and watch the prediction move, plus the same experiment measured on the full test set.
6. **Confidently in the wrong bin**: the most confident mistakes and the hard cases.
7. **Responsible AI**: data quality and bias, reliability, transparency, privacy, human control and sustainability, each backed by a measured number.
8. **Learn more**: residual waste, local bin rules and the browser-versus-Python check.
9. **Why it was made**: why, and how it was built in four steps.

The page is available in Dutch and English (NL | EN toggle, or open with `?lang=en`) and has a dark mode.

## Results

On 380 unseen test photos:

| | Compact CNN (from scratch) | MobileNetV3-Small (pretrained) |
|---|---|---|
| Accuracy | **79.5%** | **90.8%** |
| Balanced accuracy | 77.8% | 89.3% |
| Macro-F1 | 0.758 | 0.885 |
| Errors | 78 of 380 | 35 of 380 |
| Mean confidence, right / wrong | 88% / 59% | 98% / 74% |
| Errors with ≥ 80% confidence | 13 (17% of errors) | 15 (43% of errors) |
| Parameters / ONNX size | 390k / 1.6 MB | 1.5M / 6.4 MB |

What stands out:

- **Prior knowledge counts.** The pretrained model makes fewer than half the errors, with the same 1,768 training photos.
- **Plastic often becomes glass** for the CNN (12 of 73): transparent, shiny, similar shapes.
- **Residual waste is the weakest class** for both models. With only 20 test photos, one photo shifts recall by 5 points.
- **Confident is not the same as correct.** The better model is more often confidently wrong.
- **The CNN learned the white background.** Halving the light drops it from 79.5% to 56.3%. MobileNet barely notices (91.1%).

Full numbers, confusion matrices, error analysis and the robustness experiment are in [results/README.md](results/README.md) (in Dutch).

## Data

[TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, 2017): 2,527 photos, resized to 512 × 384 (`dataset-resized.zip`). The six TrashNet classes are merged into five bins, because paper and cardboard go in the same bin in the Netherlands:

| Bin | TrashNet classes | Photos |
|---|---|---|
| paper and cardboard | `paper`, `cardboard` | 997 |
| glass | `glass` | 501 |
| plastic | `plastic` | 482 |
| metal | `metal` | 410 |
| residual waste | `trash` | 137 |

- **Not in this repository.** The GitHub README gives no data license and asks for attribution, so only 16 sample photos are included (`assets/examples/`). The scripts look for the dataset in `~/.cache/waste-classifier/dataset-resized`. Set `WASTE_DATA` to use another folder.
- **Stratified 70 / 15 / 15 split, seed 42** (1,768 / 379 / 380). The filenames are stored in `results/split.json`, so both models are judged on exactly the same test photos. All choices were made on validation; each model used the test set once, at the end.
- **Imbalance handled explicitly.** Residual waste has over 7× fewer photos than paper and cardboard. Training uses class weights in the loss, and evaluation reports per-class recall, not just accuracy.

## Models

- **Compact CNN, trained from scratch**: 128 × 128 RGB input, four convolution blocks, no pretrained weights. 200 epochs (balanced validation accuracy: 72.7% at 40, 78.4% at 100, 79.6% at 200, where it levels off).
- **MobileNetV3-Small**, pretrained on ImageNet and fine-tuned on TrashNet. Shown throughout as the *comparison*, not as the self-trained model. 15 epochs (30 gave no gain).
- **Augmentation**: random crops, flips, small rotations and colour/brightness jitter. TrashNet has a white background and similar lighting everywhere; without augmentation the model mostly learns that setup.
- **Same preprocessing in Python and the browser**: scale the short side, centre-crop to a square, pixels to 0–1, normalise with the ImageNet means (for both models, so there is only one code path).

## In the browser

- **ONNX Runtime Web** (`onnxruntime-web@1.20.1` from jsDelivr): no backend, works on any static host.
- **Everything stays local.** An uploaded photo never leaves the browser.
- **Fallback**: precomputed predictions for the sample photos, for when ONNX Runtime doesn't load. Uploads don't work then, and the page says so.
- **Verified against Python**:
  - PyTorch and ONNX give the same prediction on all 380 test photos. The largest probability difference is 8.3·10⁻⁷ (CNN) and 7.6·10⁻⁶ (MobileNet).
  - Resizing and cropping in JavaScript match Pillow/torchvision bit for bit.
  - On real JPEGs, the browser and Python agree on all 32 predictions, but confidence differs by 0.1 to 2.6 points because browsers decode JPEGs slightly differently than Pillow.
  - The page also runs this check live in the visitor's browser, under "Learn more".
- **One model call at a time**: the game, lab and check share a queue, because concurrent calls gave inconsistent outputs.

## Project structure

```
index.html            the explainer (Dutch text; English lives in explainer/i18n.js)
explainer/            app.js, preprocess.js, i18n.js, styles.css, data/web.json
assets/examples/      16 TrashNet sample photos
model/                cnn.onnx, mobilenet.onnx (PyTorch checkpoints are not committed)
src/                  preprocess, models, train, evaluate, export, export_web, variations
results/              split, training histories, evaluations, parity and robustness results
tests/                Python and JavaScript preprocessing tests
```

## Run it

The page loads models and data with `fetch`, so it needs a local web server:

```sh
python -m http.server 8000
```

Then open <http://localhost:8000/>.

## Reproduce

Python 3.12. To get PyTorch with CUDA 12.8, add the PyTorch index (without a GPU, the plain PyPI version also works, just slower):

```sh
python -m venv .venv
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Unzip TrashNet's `dataset-resized.zip` into `~/.cache/waste-classifier/` (or set `WASTE_DATA`), then run from the project root:

```sh
python src/preprocess.py                  # creates results/split.json if missing
python src/train.py --model cnn
python src/train.py --model mobilenet
python src/evaluate.py --model cnn
python src/evaluate.py --model mobilenet
python src/evaluate.py --duplicates       # near-duplicates between train and test
python src/export.py                      # ONNX export + parity check
python src/export_web.py                  # data for the explainer
python src/variations.py                  # light / angle / blur / background experiment
python -m pytest tests                    # JS test is skipped without Node
```

Retraining gives similar but not identical numbers, because cuDNN on the GPU is not fully deterministic even with a fixed seed. The reported numbers belong to the histories in `results/`.

## Limitations

- **One setting.** Every photo has a white background, was taken indoors with an iPhone, and shows one object. These numbers say nothing about waste in a real kitchen or container.
- **Near-duplicates.** TrashNet likely contains several photos of the same object. A hash comparison found 2 real duplicates between train and test. That is a lower bound, so test scores may be slightly optimistic.
- **Small classes.** Each class has 20 to 149 test photos, so differences of a few percent are not meaningful.

## Credits

Dataset: Gary Thung and Mindy Yang, *Classification of Trash for Recyclability Status*, 2017 ([TrashNet](https://github.com/garythung/trashnet)). Bin colours follow the official Dutch waste pictograms (afvalcirculair.nl).
