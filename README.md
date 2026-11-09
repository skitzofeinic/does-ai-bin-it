# Does AI bin it?

A personal project: a game and interactive explainer about how image classifiers sort waste, and where they go wrong. You sort photos into the Dutch bins yourself, then see what three models chose: **DABI 1.1** and **DABI 1.2** (Does AI Bin It, a compact CNN trained from scratch) and a pretrained **MobileNetV3-Small** for comparison. Everything runs in the browser.

## What's on the site

**The game (`index.html`, shown first).** Eight test photos and the four Dutch bins: paper and cardboard, glass, PMD and residual waste. Pick a bin, then see what each model chose. The end screen has a podium and an overview per photo. You can also upload your own photo.

**The explainer (`explainer.html`).** It shows what the models really detect, per material:

1. **README**: what the page covers, with a table of contents.
2. **How it sees**: from photo to 49,152 numbers, through the 32 real filters of DABI 1.2's first layer, to five percentages. Runs live on the chosen photo.
3. **Everything it saw**: all 2,527 photos per material, the split and the learning curves.
4. **Results**: all 380 test photos per material (right, wrong or confidently wrong) for each model, the most common mix-ups, and DABI 1.1 vs DABI 1.2.
5. **Break it**: change light, angle, blur and background and watch the prediction move, plus the same experiment measured on the full test set.
6. **AI really binned this one**: the most confident mistakes and the hard cases.
7. **Responsible AI**: data quality and bias, reliability, transparency, privacy, human control and sustainability, each backed by a measured number.
8. **Learn more**: the basics, why it goes wrong, and how the Dutch bins work.
9. **Why it was made**: how it was built in four steps.

The site is available in Dutch and English (NL | EN toggle, or open with `?lang=en`) and has a dark mode.

## Results

On 380 unseen test photos:

| | DABI 1.1 | DABI 1.2 | MobileNetV3-Small (pretrained) |
|---|---|---|---|
| Accuracy (right material) | **79.5%** | **84.0%** | **90.8%** |
| Balanced accuracy | 77.8% | 83.5% | 89.3% |
| Macro-F1 | 0.758 | 0.812 | 0.885 |
| Errors | 78 of 380 | 61 of 380 | 35 of 380 |
| Mean confidence, right / wrong | 88% / 59% | 89% / 64% | 98% / 74% |
| Errors with ≥ 80% confidence | 13 (17% of errors) | 15 (25% of errors) | 15 (43% of errors) |
| Right bin, as in the game | 80.8% | 84.7% | 91.6% |
| Parameters / ONNX size | 390k / 1.6 MB | 980k / 4.0 MB | 1.5M / 6.4 MB |

What stands out:

- **Prior knowledge counts.** The pretrained model makes the fewest errors (35, against 78 and 61), with the same 1,768 training photos.
- **DABI 1.2 closes part of the gap.** It adds a fifth convolution block and random erasing. On validation, each change contributes about half of the gain.
- **Plastic often becomes glass**: 12 of 73 test photos for DABI 1.1, 8 for DABI 1.2 and 4 for MobileNet. Transparent, shiny, similar shapes.
- **Residual waste is the weakest class** for all three models. With only 20 test photos, one photo shifts recall by 5 points.
- **Confident is not the same as correct.** The better the model, the larger the share of its errors that are confident.
- **Both DABI versions learned the white background.** Halving the light drops DABI 1.1 from 79.5% to 56.3% and DABI 1.2 from 84.0% to 60.8%. MobileNet barely notices (91.1%).

Full numbers, confusion matrices, error analysis, the variants that did not work and the robustness experiment are in [results/README.md](results/README.md) (in Dutch).

## Data

[TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, 2017): 2,527 photos, resized to 512 × 384 (`dataset-resized.zip`). The models learn five materials. The game maps those onto the Dutch bins:

| Material | TrashNet classes | Photos | Bin in the game |
|---|---|---|---|
| paper and cardboard | `paper`, `cardboard` | 997 | paper and cardboard |
| glass | `glass` | 501 | glass |
| plastic | `plastic` | 482 | PMD |
| metal | `metal` | 410 | PMD |
| residual waste | `trash` | 137 | residual waste |

- **Not in this repository.** TrashNet is MIT-licensed (the dataset zip lives in its GitHub repository), but the full set is 42 MB and easy to download, so only 16 sample photos are included (`assets/examples/`, with the MIT notice in `assets/examples/LICENSE`). The scripts look for the dataset in `~/.cache/waste-classifier/dataset-resized`. Set `WASTE_DATA` to use another folder.
- **Stratified 70 / 15 / 15 split, seed 42** (1,768 / 379 / 380). The filenames are stored in `results/split.json`, so all models are judged on exactly the same test photos. All choices were made on validation; each model used the test set once, at the end.
- **Imbalance handled explicitly.** Residual waste has over 7× fewer photos than paper and cardboard. Training uses class weights in the loss, and evaluation reports per-class recall, not just accuracy.

## Materials and bins

In most Dutch municipalities plastic and metal (cans) go together in the PMD bin, so the game and the explainer count differently:

- **The game counts the bin.** A model's chance of a bin is the sum of its chances for the materials in that bin, so mistaking plastic for metal is still the right bin.
- **The explainer counts the material**, so those mistakes stay visible. Where the game would count one as right, the page says so.
- **Colours.** The bins use the official Rijkswaterstaat colours. Metal has no official colour of its own because cans belong in PMD, so it is pink in the explainer only.

## Models

- **DABI 1.1**, trained from scratch: 128 × 128 RGB input, four convolution blocks, no pretrained weights. 200 epochs (balanced validation accuracy: 72.7% at 40, 78.4% at 100, 79.6% at 200, where it levels off).
- **DABI 1.2**, trained from scratch with the same settings: a fifth convolution block and random erasing during training. Fine-tuning DABI 1.1, stronger zoom and cropping to the object did not beat it on validation.
- **MobileNetV3-Small**, pretrained on ImageNet and fine-tuned on TrashNet. Shown throughout as the *comparison*: it starts with a large head start that DABI does not have. 15 epochs (30 gave no gain).
- **Augmentation**: random crops, flips, small rotations and colour/brightness jitter. TrashNet has a white background and similar lighting everywhere; without augmentation the model mostly learns that setup.
- **Same preprocessing in Python and the browser**: scale the short side, centre-crop to a square, pixels to 0–1, normalise with the ImageNet means (for all models, so there is only one code path).

## In the browser

- **ONNX Runtime Web** (`onnxruntime-web@1.20.1` from jsDelivr): no backend, works on any static host.
- **Everything stays local.** An uploaded photo never leaves the browser.
- **Fallback**: precomputed predictions for the sample photos, for when ONNX Runtime doesn't load. Uploads don't work then, and the page says so.
- **Verified against Python**:
  - PyTorch and ONNX give the same prediction on all 380 test photos. The largest probability difference is 8.3·10⁻⁷ (DABI 1.1), 1.0·10⁻⁶ (DABI 1.2) and 7.6·10⁻⁶ (MobileNet).
  - Resizing and cropping in JavaScript match Pillow/torchvision bit for bit.
  - On the 16 sample JPEGs, the browser and Python agreed on all 48 predictions in headless Brave. Confidence can differ slightly (up to 2.6 points in earlier measurements) because browsers decode JPEGs slightly differently than Pillow.
  - The page also runs this check live in the visitor's browser, under "Learn more".
- **One model call at a time**: the game, lab and check share a queue, because concurrent calls gave inconsistent outputs.

## Project structure

```
index.html            the game, shown first (Dutch text; English lives in explainer/i18n.js)
explainer.html        the explainer: how DABI sees, data, results, lab, Responsible AI
explainer/            app.js, preprocess.js, i18n.js, styles.css, data/web.json
assets/examples/      16 TrashNet sample photos
model/                cnn.onnx (DABI 1.1), cnn-1.2.onnx (DABI 1.2), mobilenet.onnx (PyTorch checkpoints are not committed)
src/                  preprocess, models, train, evaluate, evaluate_bins, export, export_web, variations
results/              split, training histories, evaluations, parity and robustness results
tests/                Python and JavaScript preprocessing tests
```

## Run it

The site loads models and data with `fetch`, so it needs a local web server:

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

Unzip TrashNet's `dataset-resized.zip` into `~/.cache/waste-classifier/` (or set `WASTE_DATA`), then run from the project root. In the scripts, `cnn` is DABI 1.1 and `cnn-1.2` is DABI 1.2:

```sh
python src/preprocess.py                  # creates results/split.json if missing
python src/train.py --model cnn
python src/train.py --model cnn-1.2
python src/train.py --model mobilenet
python src/evaluate.py --model cnn
python src/evaluate.py --model cnn-1.2
python src/evaluate.py --model mobilenet
python src/evaluate.py --duplicates       # near-duplicates between train and test
python src/export.py                      # ONNX export + parity check
python src/export_web.py                  # data for the explainer
python src/variations.py                  # light / angle / blur / background experiment
python src/evaluate_bins.py               # scores and robustness per Dutch bin (PMD = plastic + metal), from the ONNX models
python -m pytest tests                    # JS test is skipped without Node
```

Retraining gives similar but not identical numbers, because cuDNN on the GPU is not fully deterministic even with a fixed seed. The reported numbers belong to the histories in `results/`.

## Limitations

- **One setting.** Every photo has a white background, was taken indoors with an iPhone, and shows one object. These numbers say nothing about waste in a real kitchen or container.
- **No food waste.** TrashNet has no food or garden waste, so the models always pick one of the five materials.
- **Near-duplicates.** TrashNet likely contains several photos of the same object. A hash comparison found 2 real duplicates between train and test. That is a lower bound, so test scores may be slightly optimistic.
- **Small classes.** Each class has 20 to 149 test photos, so differences of a few percent are not meaningful.

## Credits

Dataset: Gary Thung and Mindy Yang, *Classification of Trash for Recyclability Status*, 2017 ([TrashNet](https://github.com/garythung/trashnet), MIT License).

Papers: Howard et al., [*Searching for MobileNetV3*](https://arxiv.org/abs/1905.02244), ICCV 2019. Russakovsky et al., [*ImageNet Large Scale Visual Recognition Challenge*](https://arxiv.org/abs/1409.0575), IJCV 2015. Zhong et al., [*Random Erasing Data Augmentation*](https://arxiv.org/abs/1708.04896), AAAI 2020.

Software: PyTorch (BSD), torchvision (BSD-3-Clause), ONNX Runtime (MIT), JetBrains Mono (SIL Open Font License 1.1).

Bin colours follow the Dutch waste-separation set by Rijkswaterstaat, designed by Mijksenaar ([guidelines](https://www.afvalcirculair.nl/publish/pages/124417/rws_richtlijnen_pictogrammenset_def-toegankelijk.pdf)). The usage right is granted without limit for promoting waste separation; the pictograms themselves may not be modified, and this project only uses the colours.

The MobileNet weights come from torchvision and were pretrained on ImageNet. ImageNet's [terms of access](https://image-net.org/download) limit the database to non-commercial research and education; whether that also binds weights trained on it is not settled. This project is non-commercial; check those terms before any commercial use.
