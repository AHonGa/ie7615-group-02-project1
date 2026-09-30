"""Model architectures: a custom CNN from scratch, and a ResNet18 transfer-learning
wrapper (feature-extraction or fine-tuning mode)."""
import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """4-block CNN with batchnorm + dropout, designed for 128x128 RGB face crops
    and a small number of classes (4-6 identities)."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 128 -> 64
            block(32, 64),   # 64  -> 32
            block(64, 128),  # 32  -> 16
            block(128, 256), # 16  -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet18(num_classes: int, mode: str = "finetune") -> nn.Module:
    """mode='feature_extract' freezes the backbone and trains only the new head.
    mode='finetune' unfreezes everything (use a lower LR for the backbone)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if mode == "feature_extract":
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # new head is always trainable
    return model


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
