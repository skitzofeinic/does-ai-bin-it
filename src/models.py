"""De twee modellen: een compact CNN vanaf nul, en MobileNetV3-Small ter vergelijking."""

from torch import nn
from torchvision import models

from preprocess import CLASSES

# Invoergrootte per model (vierkant, RGB). cnn = DABI 1.1, cnn-1.2 = DABI 1.2 (een blok extra)
IMAGE_SIZE = {"cnn": 128, "cnn-1.2": 128, "mobilenet": 224}

# Kanalen per convolutieblok. Verder trainen vanaf 1.1 met meer epochs of een lagere lr
# bracht val bal. acc niet boven 0,796, dus 1.2 krijgt meer capaciteit en traint vanaf nul.
WIDTHS = {"cnn": (32, 64, 128, 256), "cnn-1.2": (32, 64, 128, 256, 256)}


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WasteCNN(nn.Module):
    """Convolutieblokken (elk halveert de resolutie) + global average pooling.
    Invoer (N, 3, 128, 128), genormaliseerd. Standaard 4 blokken: 128 -> 8."""

    def __init__(self, n_classes: int = len(CLASSES), widths: tuple[int, ...] = WIDTHS["cnn"]):
        super().__init__()
        channels = (3, *widths)
        self.features = nn.Sequential(*(conv_block(c_in, c_out)
                                        for c_in, c_out in zip(channels, channels[1:])))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(widths[-1], n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))  # logits


def mobilenet(n_classes: int = len(CLASSES)) -> nn.Module:
    """MobileNetV3-Small met ImageNet-gewichten; alleen de laatste laag is nieuw."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    return model


def build(name: str) -> nn.Module:
    if name in WIDTHS:
        return WasteCNN(widths=WIDTHS[name])
    if name == "mobilenet":
        return mobilenet()
    raise ValueError(f"onbekend model: {name}")
