# Resultaten

Alle cijfers hieronder zijn gemeten met de scripts in `src/`, op de vaste split in `split.json` (seed 42). De testset (380 foto's) is per model één keer gebruikt, nadat alle keuzes op de validatieset gemaakt waren.

| Bestand | Inhoud | Gemaakt door |
|---|---|---|
| `split.json` | bestandsnamen per split (1.768 / 379 / 380) | `src/preprocess.py` |
| `history_<model>.json` | loss en accuracy per epoch, instellingen, beste epoch | `src/train.py` |
| `evaluation_<model>.json` | testmetrics, confusion matrix, alle foute voorspellingen | `src/evaluate.py --model` |
| `near_duplicates.json` | kandidaat-duplicaten tussen train en test | `src/evaluate.py --duplicates` |
| `parity_onnx.json` | PyTorch tegenover ONNX Runtime op de testset | `src/export.py` |

## Testset

| | Compact CNN (vanaf nul) | MobileNetV3-Small (voorgetraind, vergelijking) |
|---|---|---|
| Accuracy | **79,5%** | **90,8%** |
| Gebalanceerde accuracy | 77,8% | 89,3% |
| Macro-F1 | 0,758 | 0,885 |
| Fouten | 78 van 380 | 35 van 380 |
| Gemiddelde zekerheid bij goed / fout | 88% / 59% | 98% / 74% |
| Fouten met ≥ 80% zekerheid | 13 (17% van de fouten) | 15 (43% van de fouten) |
| Gekozen epoch | 168 van 200 | 14 van 15 |
| Parameters / ONNX-bestand | 390k / 1,6 MB | 1,5M / 6,4 MB |

Per klasse (F1): 

| Klasse | n | CNN | MobileNet |
|---|---|---|---|
| papier en karton | 149 | 0,88 | 0,95 |
| glas | 76 | 0,74 | 0,88 |
| plastic | 73 | 0,76 | 0,88 |
| metaal | 62 | 0,77 | 0,90 |
| restafval | 20 | 0,64 | 0,81 |

Confusion matrix CNN (rij = echte klasse, kolom = voorspelling; volgorde zoals hierboven):

```
papier en karton  125   6   4   8   6
glas                4  62   5   5   0
plastic             4  12  52   1   4
metaal              3   7   2  48   2
restafval           0   4   0   1  15
```

## Wat opvalt

- **Voorkennis telt**: het voorgetrainde model maakt ruim de helft minder fouten (35 tegenover 78), met maar 1.768 eigen trainingsfoto's voor allebei.
- **Plastic wordt vaak glas** bij het CNN (12 van 73): doorzichtig, glanzend, vergelijkbare vorm. Transparantie is een eigenschap, geen materiaal.
- **Restafval is het zwakst** bij beide modellen. Met 20 testfoto's verschuift één foto de recall al 5 procentpunt; dit cijfer is dus onzeker.
- **Zeker is niet hetzelfde als juist**: het CNN was 97% zeker dat een tijdschriftpagina restafval was; MobileNet 94% zeker dat een blik glas was. Het betere model is vaker overtuigd fout: 43% van zijn fouten heeft ≥ 80% zekerheid, bij het CNN 17%.

## Foutanalyse (de meest zekere fouten, bekeken)

- **Twijfelgevallen in de data**: een gewatteerde envelop (papier met plastic noppen) als plastic gelabeld, een tissue als restafval gelabeld (het model zegt papier), een wattenschijfje als papier. Hier twijfelt een mens ook; het label is een keuze.
- **Close-ups zonder vorm**: foto's van alleen een oppervlak of rand (metal331, plastic264, cardboard99) geven het model weinig houvast.
- **Materiaal dat misleidt**: een doorzichtige glazen pot wordt metaal (reflecties), bedrukt tijdschriftpapier wordt restafval.

## Bijna-duplicaten tussen train en test

Een ruwe hash-vergelijking (dHash 8×8, binnen dezelfde map, afstand ≤ 4) markeerde 18 van de 380 testfoto's. Bij handmatige controle waren er **2 echt hetzelfde voorwerp**: `paper345` ≈ `paper451` (hetzelfde tijdschrift) en `plastic9` ≈ `plastic11` (hetzelfde flesje). De andere 16 waren verschillende voorwerpen op een vergelijkbare achtergrond.

Dit is een **ondergrens**: de methode vindt alleen bijna identieke opnames, niet hetzelfde voorwerp vanuit een andere hoek. De testscores kunnen daardoor iets te rooskleurig zijn. Het CNN had `plastic9` fout, MobileNet had beide goed.

## PyTorch tegenover ONNX

Op alle 380 testfoto's geven de ONNX-modellen (ONNX Runtime, Python) **dezelfde voorspelling** als PyTorch. Het grootste verschil in een kans is 8,3·10⁻⁷ (CNN) en 7,6·10⁻⁶ (MobileNet). De vergelijking met de browser (ONNX Runtime Web) volgt in de explainer.

## Beperkingen

- Alle foto's: witte achtergrond, binnen, iPhone, één voorwerp in beeld. Deze cijfers zeggen niets over afval in een echte keuken of container.
- Eén training per model; door niet-deterministische GPU-berekeningen geeft opnieuw trainen vergelijkbare maar niet identieke cijfers.
- Klein aantal testfoto's per klasse (20–149); verschillen van een paar procent zijn niet betekenisvol.
