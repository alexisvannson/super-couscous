"""
Implementation of DenseNet-121 following the paper "Densely Connected Convolutional Networks" (Huang et al., CVPR 2017) and torchvision's implementation (to load pretrained weights).

The model is a densely connected network for multi-label classification.

The model is based on the following architecture:

- Stem: 224×224 → 56×56  (names match torchvision: conv0, norm0, relu0, pool0)
- Dense Blocks: 6, 12, 24, 16
- Transition Layers: 3
- Final BN (name matches torchvision: norm5)
- Classifier: 1024 → num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DenseLayer(nn.Module):
    """
    Single Dense Layer: BN → ReLU → Conv(1×1, 4k) → BN → ReLU → Conv(3×3, k).

    Args:
        in_channels: Total input channels (grows as the block deepens).
        growth_rate:  k — number of new feature maps produced by this layer.
        drop_rate:    Dropout probability after the 3×3 conv (0 = disabled).
    """

    def __init__(self, in_channels: int, growth_rate: int = 32, drop_rate: float = 0.0):
        super(DenseLayer, self).__init__()
        bottleneck_channels = 4 * growth_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, growth_rate,
                               kernel_size=3, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out  # shape: (B, growth_rate, H, W) — new channels only


class DenseBlock(nn.ModuleDict):
    """
    Dense Block: each layer receives the concatenation of all preceding layer outputs.

    Output channels = in_channels + num_layers × growth_rate.

    Args:
        num_layers:  Number of Dense Layers in this block.
        in_channels: Channels entering the block.
        growth_rate: k — new channels per layer.
        drop_rate:   Dropout rate forwarded to each Dense Layer.
    """

    def __init__(self, num_layers: int, in_channels: int,
                 growth_rate: int = 32, drop_rate: float = 0.0):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate)
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = [x]
        for layer in self.values():
            new_features = layer(torch.cat(feature_maps, dim=1))
            feature_maps.append(new_features)
        return torch.cat(feature_maps, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition Layer: compresses channels and halves spatial dimensions between Dense Blocks.

    Structure: BN → ReLU → Conv(1×1, floor(θ×C)) → AvgPool(2×2, stride=2).

    Args:
        in_channels:  Channels from the preceding Dense Block.
        compression:  θ ∈ (0, 1] — channel reduction factor (0.5 halves channels).
    """

    def __init__(self, in_channels: int, compression: float = 0.5):
        super(TransitionLayer, self).__init__()
        out_channels = int(in_channels * compression)
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x), inplace=True))
        return self.pool(x)


class DenseNet121(nn.Module):
    """
    DenseNet-121 (DenseNet-BC): densely connected network for multi-label classification.

    Tensor shapes through the network (B = batch, k = 32):
        input              (B,    3, 224, 224)
        after stem         (B,   64,  56,  56)   conv0/norm0/relu0/pool0
        after block 1      (B,  256,  56,  56)   64  + 6×32
        after transition 1 (B,  128,  28,  28)   256×0.5, pool÷2
        after block 2      (B,  512,  28,  28)   128 + 12×32
        after transition 2 (B,  256,  14,  14)   512×0.5, pool÷2
        after block 3      (B, 1024,  14,  14)   256 + 24×32
        after transition 3 (B,  512,   7,   7)   1024×0.5, pool÷2
        after block 4      (B, 1024,   7,   7)   512 + 16×32
        after global pool  (B, 1024)
        logits             (B, num_classes)

    Args:
        num_classes:  Number of output logits.
        in_channels:  Input image channels (3 = RGB).
        growth_rate:  k — new feature maps per Dense Layer.
        compression:  θ — Transition channel reduction factor.
        drop_rate:    Dropout inside Dense Layers.
        block_config: Dense layers per block. (6,12,24,16) defines DenseNet-121.
    """

    DENSENET121_BLOCKS: tuple = (6, 12, 24, 16)

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        growth_rate: int = 32,
        compression: float = 0.5,
        drop_rate: float = 0.0,
        block_config: tuple = DENSENET121_BLOCKS,
    ):
        super(DenseNet121, self).__init__()

        initial_channels = 2 * growth_rate  # 64 for k=32

        # Stem: 224×224 → 56×56  (names match torchvision: conv0, norm0, relu0, pool0)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, initial_channels,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(initial_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        current_channels = initial_channels

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, current_channels, growth_rate, drop_rate)
            self.features.add_module(f'denseblock{i + 1}', block)
            current_channels += num_layers * growth_rate

            if i < len(block_config) - 1:
                trans = TransitionLayer(current_channels, compression)
                self.features.add_module(f'transition{i + 1}', trans)
                current_channels = int(current_channels * compression)

        # Final BN (name matches torchvision: norm5)
        self.features.add_module('norm5', nn.BatchNorm2d(current_channels))

        self.classifier = nn.Linear(current_channels, num_classes)
        self.out_channels = current_channels  # 1024 for standard DenseNet-121

        self._init_weights()

    def load_imagenet_weights(self, strict_backbone: bool = True):
        """
        Load ImageNet-pretrained weights from torchvision's DenseNet-121 into this model.
        The classifier head is always skipped (different num_classes).

        Module names in this model mirror torchvision's DenseNet-121, so backbone
        weights are loaded directly without key remapping:
            features.conv0                             (stem conv)
            features.norm0                             (stem BN)
            features.denseblock{i}.denselayer{j}.norm1
            features.denseblock{i}.denselayer{j}.conv1
            features.denseblock{i}.denselayer{j}.norm2
            features.denseblock{i}.denselayer{j}.conv2
            features.transition{i}.norm
            features.transition{i}.conv
            features.norm5                             (final BN)
        """
        from torchvision.models import densenet121, DenseNet121_Weights
        pretrained_sd = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).state_dict()
        backbone_sd = {k: v for k, v in pretrained_sd.items() if not k.startswith('classifier')}
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)

        backbone_missing = [k for k in missing if not k.startswith('classifier')]
        if strict_backbone and backbone_missing:
            raise RuntimeError(f"Failed to load backbone weights: {backbone_missing}")

        print("Pretrained ImageNet weights loaded. Classifier head re-initialised from scratch.")
        if backbone_missing:
            print(f"  Missing  : {backbone_missing}")
        if unexpected:
            print(f"  Unexpected: {unexpected}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(start_dim=1)
        return self.classifier(x)
