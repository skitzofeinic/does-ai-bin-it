# Resultaten

Alle cijfers hieronder zijn gemeten met de scripts in `src/`, op de vaste split in `split.json` (seed 42). De testset (380 foto's) is per model één keer gebruikt, nadat alle keuzes op de validatieset gemaakt waren.

Drie modellen: **DABI 1.1** en **DABI 1.2** (Does AI Bin It, het eigen CNN, vanaf nul getraind) en **MobileNetV3-Small** (voorgetraind op ImageNet, ter vergelijking).

| Bestand | Inhoud | Gemaakt door |
|---|---|---|
| `split.json` | bestandsnamen per split (1.768 / 379 / 380) | `src/preprocess.py` |
| `history_<model>.json` | loss en accuracy per epoch, instellingen, beste epoch | `src/train.py` |
| `evaluation_<model>.json` | testmetrics per materiaal, confusion matrix, alle foute voorspellingen | `src/evaluate.py --model` |
| `evaluation_bins_<model>.json` | dezelfde metrics per Nederlandse bak (PMD = plastic + metaal) | `src/evaluate_bins.py` |
| `near_duplicates.json` | kandidaat-duplicaten tussen train en test | `src/evaluate.py --duplicates` |
| `parity_onnx.json` | PyTorch tegenover ONNX Runtime op de testset | `src/export.py` |
| `variations.json` | accuracy na aanpassingen van licht, hoek, scherpte en ondergrond | `src/variations.py` |
| `variations_bins.json` | dezelfde aanpassingen, per bak | `src/evaluate_bins.py` |

`<model>` is `cnn` (DABI 1.1), `cnn-1.2` (DABI 1.2) of `mobilenet`.

## Testset

| | DABI 1.1 | DABI 1.2 | MobileNetV3-Small |
|---|---|---|---|
| Accuracy | **79,5%** | **84,0%** | **90,8%** |
| Gebalanceerde accuracy | 77,8% | 83,5% | 89,3% |
| Macro-F1 | 0,758 | 0,812 | 0,885 |
| Fouten | 78 van 380 | 61 van 380 | 35 van 380 |
| Gemiddelde zekerheid bij goed / fout | 88% / 59% | 89% / 64% | 98% / 74% |
| Fouten met ≥ 80% zekerheid | 13 (17% van de fouten) | 15 (25% van de fouten) | 15 (43% van de fouten) |
| Gekozen epoch | 168 van 200 | 178 van 200 | 14 van 15 |
| Parameters / ONNX-bestand | 390k / 1,6 MB | 980k / 4,0 MB | 1,5M / 6,4 MB |

DABI 1.2 heeft dezelfde training als DABI 1.1 (200 epochs, lr 2e-3, gewogen cross-entropy), met twee verschillen: een vijfde laag filters (256 kanalen) en RandomErasing bij het trainen, waarbij willekeurige stukjes van de foto worden weggevlakt.

Per klasse (F1):

| Klasse | n | DABI 1.1 | DABI 1.2 | MobileNet |
|---|---|---|---|---|
| papier en karton | 149 | 0,88 | 0,90 | 0,95 |
| glas | 76 | 0,74 | 0,82 | 0,88 |
| plastic | 73 | 0,76 | 0,82 | 0,88 |
| metaal | 62 | 0,77 | 0,80 | 0,90 |
| restafval | 20 | 0,64 | 0,72 | 0,81 |

Confusion matrices (rij = echte klasse, kolom = voorspelling; volgorde zoals hierboven):

```
DABI 1.1                                 DABI 1.2
papier en karton  125   6   4   8   6    128   2   5   7   7
glas                4  62   5   5   0      3  64   5   4   0
plastic             4  12  52   1   4      3   8  61   0   1
metaal              3   7   2  48   2      1   6   4  49   2
restafval           0   4   0   1  15      1   1   0   1  17
```

## Wat opvalt

- **Voorkennis telt**: het voorgetrainde model maakt ruim de helft minder fouten dan DABI 1.1 (35 tegenover 78), met maar 1.768 eigen trainingsfoto's voor allemaal.
- **DABI 1.2 haalt een deel in**: 61 fouten in plaats van 78, beter op elke klasse. Van het verschil met DABI 1.1 komt ongeveer de helft van de extra laag en de helft van het wegvlakken (validatie, gemiddelde van de laatste 20 epochs: DABI 1.1 0,773, alleen extra laag 0,800, extra laag + wegvlakken 0,826).
- **Plastic wordt vaak glas**: DABI 1.1 12 van 73, DABI 1.2 8, MobileNet 4. Doorzichtig, glanzend, vergelijkbare vorm. Transparantie is een eigenschap, geen materiaal.
- **Restafval is het zwakst** bij alle drie. Met 20 testfoto's verschuift één foto de recall al 5 procentpunt; dit cijfer is dus onzeker.
- **Zeker is niet hetzelfde als juist**: DABI 1.1 was 97% zeker dat een tijdschriftpagina restafval was, DABI 1.2 90% zeker dat een doorzichtige plastic fles glas was, MobileNet 94% zeker dat een blik glas was. Hoe beter het model, hoe groter het deel van de fouten dat overtuigd is: 17%, 25%, 43%.

## Wat niet werkte voor DABI 1.2

Vóór de keuze zijn varianten op de validatieset vergeleken (gemiddelde van de laatste 20 epochs; de testset is daarvoor niet gebruikt):

| Variant | Gebalanceerde accuracy | Plastic als glas |
|---|---|---|
| Verder trainen vanaf DABI 1.1, 30 epochs, lr 3e-4 / 1e-4 / 5e-5 | 0,78 | 6–7 |
| **Extra laag + wegvlakken (DABI 1.2)** | **0,826** | **4,0** |
| + verder inzoomen (crops van 30–100% in plaats van 60–100%) | 0,800 | 7,0 |
| + uitsnijden rond het voorwerp | 0,819 | 8,0 |
| + uitsnijden en inzoomen | 0,810 | 9,3 |

Verder trainen vanaf DABI 1.1 bleef rond het startpunt hangen: dat model zat al aan zijn grens. Uitsnijden en inzoomen hielpen niet, en maakten plastic-als-glas juist erger. Ze staan nog in `src/train.py` (`--crop`, `--min-scale`), standaard uit.

## Per Nederlandse bak

De modellen leren vijf materialen, maar in de meeste Nederlandse gemeenten gaan plastic en metaal (blik) samen in de PMD-bak. Het spel op de site telt daarom per bak: de kans op een bak is de som van de kansen op de materialen erin. `src/evaluate_bins.py` rekent dat uit met de ONNX-modellen, precies zoals de browser.

| | DABI 1.1 | DABI 1.2 | MobileNet |
|---|---|---|---|
| Juiste bak | 80,8% | 84,7% | 91,6% |
| Gebalanceerde accuracy per bak | 78,5% | 84,6% | 89,7% |
| Fouten | 73 | 58 | 32 |
| Fouten met ≥ 80% zekerheid | 14 | 16 | 17 |
| Plastic en metaal verwisseld (fout per materiaal, goed per bak) | 3 | 4 | 3 |

Het verschil met de scores per materiaal is klein, onder 1 procentpunt: plastic en metaal werden zelden verwisseld. De grootste fout, plastic dat glas wordt, blijft ook per bak een fout. De uitleg op de site toont de materialen, zodat ook plastic-metaalfouten zichtbaar blijven, met per model de score per bak ernaast.

## Foutanalyse (de meest zekere fouten, bekeken)

- **Twijfelgevallen in de data**: een gewatteerde envelop (papier met plastic noppen) als plastic gelabeld, een tissue als restafval gelabeld (het model zegt papier), een wattenschijfje als papier. Hier twijfelt een mens ook; het label is een keuze.
- **Close-ups zonder vorm**: foto's van alleen een oppervlak of rand (metal331, plastic264, cardboard99) geven het model weinig houvast.
- **Materiaal dat misleidt**: een doorzichtige glazen pot wordt metaal (reflecties), bedrukt tijdschriftpapier wordt restafval.
- **Kleine voorwerpen en doorzichtig plastic** (vooral de plastic-als-glasfouten van DABI 1.2): flesjes die klein in beeld zijn, doorzichtige eierdozen en bakjes, en vormen die op een fles lijken. Op 128 × 128 pixels blijft weinig over van wanddikte, glans en schroefdraad. Geen enkel label leek fout.

## Bijna-duplicaten tussen train en test

Een ruwe hash-vergelijking (dHash 8×8, binnen dezelfde map, afstand ≤ 4) markeerde 18 van de 380 testfoto's. Bij handmatige controle waren er **2 echt hetzelfde voorwerp**: `paper345` ≈ `paper451` (hetzelfde tijdschrift) en `plastic9` ≈ `plastic11` (hetzelfde flesje). De andere 16 waren verschillende voorwerpen op een vergelijkbare achtergrond.

Dit is een **ondergrens**: de methode vindt alleen bijna identieke opnames, niet hetzelfde voorwerp vanuit een andere hoek. De testscores kunnen daardoor iets te rooskleurig zijn. DABI 1.1 had `plastic9` fout, MobileNet had beide goed.

## PyTorch tegenover ONNX

Op alle 380 testfoto's geven de ONNX-modellen (ONNX Runtime, Python) **dezelfde voorspelling** als PyTorch. Het grootste verschil in een kans is 8,3·10⁻⁷ (DABI 1.1), 1,0·10⁻⁶ (DABI 1.2) en 7,6·10⁻⁶ (MobileNet).

## Browser tegenover Python

De explainer controleert dit live in de browser van de bezoeker (sectie "Meer weten") en toont de eigen uitkomst. Gemeten in Brave (Chromium), headless, over meerdere sessies:

- **Verkleinen en uitsnijden**: in JavaScript bit voor bit gelijk aan Pillow/torchvision, op zes beeldformaten (automatische test `tests/test_preprocess_js.py`, ook het afronden bij een halve pixel).
- **Zelfde pixels** (een synthetisch testbeeld uit een formule, zonder JPEG): grootste verschil in een kans 0,0004 tot 0,0006 procentpunt (DABI 1.1 en DABI 1.2) en 0,008 tot 0,16 procentpunt (MobileNet). Dat is afronding van kommagetallen in ONNX Runtime Web; bij MobileNet stapelt die zich op over meer lagen.
- **Echte JPEG-voorbeelden** (16 foto's × 3 modellen): 48 van de 48 voorspellingen gelijk in de laatste meting (eerder, met twee modellen, steeds 32 van de 32). De zekerheid verschilde in die eerdere metingen 0,1 tot 2,6 procentpunt. Browsers openen een JPEG net iets anders dan Pillow (in onze meting was 37 tot 49% van de pixels 1 of 2 op 255 anders). Onzichtbaar, maar het model merkt het.

## Variaties: buiten wit papier

Eén aanpassing tegelijk op alle 380 testfoto's, gemeten met `src/variations.py` (accuracy per materiaal):

| Aanpassing | DABI 1.1 | DABI 1.2 | MobileNet |
|---|---|---|---|
| Origineel | 79,5% | 84,0% | 90,8% |
| Half zoveel licht (0,5×) | 56,3% | 60,8% | 91,1% |
| Twee keer zoveel licht (2×) | 52,9% | 59,2% | 78,9% |
| 45° gedraaid | 73,7% | 77,4% | 84,0% |
| 180° gedraaid | 77,4% | 80,8% | 91,6% |
| Licht vervaagd (straal 4) | 56,1% | 57,6% | 78,2% |
| Sterk vervaagd (straal 16) | 28,4% | 28,2% | 40,0% |
| Op donkere ondergrond, foto 75% | 60,3% | 70,8% | 90,0% |
| Op donkere ondergrond, foto 35% | 40,5% | 51,1% | 51,8% |

- Beide DABI's zijn gevoelig voor licht en achtergrond: ze leerden de witte posterboard mee. MobileNet, met voorkennis uit ImageNet, merkt bijna niets van half licht of een iets kleinere foto op een donkere ondergrond.
- DABI 1.2 houdt op een donkere ondergrond duidelijk beter stand dan DABI 1.1 (70,8% tegen 60,3% bij foto 75%). Bij vervagen helpt de extra laag niet.
- Ondersteboven (180°) gaat beter dan schuin (45°): bij het trainen werden foto's horizontaal en verticaal gespiegeld, en twee keer spiegelen is 180° draaien.
- Vervagen raakt alle modellen het hardst.

## Beperkingen

- Alle foto's: witte achtergrond, binnen, iPhone, één voorwerp in beeld. Deze cijfers zeggen niets over afval in een echte keuken of container.
- Eén training per model; door niet-deterministische GPU-berekeningen geeft opnieuw trainen vergelijkbare maar niet identieke cijfers.
- Klein aantal testfoto's per klasse (20–149); verschillen van een paar procent zijn niet betekenisvol.
- GFT (groente, fruit, tuinafval) zit niet in TrashNet; de modellen kennen het niet.
