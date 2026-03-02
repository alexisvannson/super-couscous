import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from models.DenseNet import DenseLayer, DenseBlock, TransitionLayer, DenseNet121


# ── DenseLayer ────────────────────────────────────────────────────────────────

class TestDenseLayer:
    def test_output_channels_equal_growth_rate(self):
        layer = DenseLayer(in_channels=64, growth_rate=32)
        x = torch.zeros(2, 64, 8, 8)
        out = layer(x)
        assert out.shape == (2, 32, 8, 8)

    def test_spatial_size_preserved(self):
        layer = DenseLayer(in_channels=128, growth_rate=16)
        x = torch.zeros(1, 128, 14, 14)
        out = layer(x)
        assert out.shape[2:] == (14, 14)

    def test_attribute_names_match_torchvision(self):
        """norm1/conv1/norm2/conv2 must match torchvision's _DenseLayer state-dict keys."""
        layer = DenseLayer(in_channels=64, growth_rate=32)
        for attr in ('norm1', 'conv1', 'norm2', 'conv2'):
            assert hasattr(layer, attr), f"DenseLayer missing attribute '{attr}'"

    def test_dropout_disabled_by_default(self):
        layer = DenseLayer(in_channels=64, growth_rate=32)
        assert layer.drop_rate == 0.0


# ── DenseBlock ────────────────────────────────────────────────────────────────

class TestDenseBlock:
    def test_output_channels(self):
        # out = in_channels + num_layers * growth_rate
        block = DenseBlock(num_layers=6, in_channels=64, growth_rate=32)
        x = torch.zeros(2, 64, 56, 56)
        out = block(x)
        assert out.shape == (2, 64 + 6 * 32, 56, 56)

    def test_spatial_size_preserved(self):
        block = DenseBlock(num_layers=4, in_channels=32, growth_rate=8)
        x = torch.zeros(1, 32, 10, 10)
        out = block(x)
        assert out.shape[2:] == (10, 10)

    def test_layer_names_match_torchvision(self):
        """Sub-modules must be named denselayer1, denselayer2, … to match torchvision."""
        block = DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
        for i in range(1, 5):
            assert f'denselayer{i}' in block, \
                f"DenseBlock missing sub-module 'denselayer{i}'"

    def test_number_of_layers(self):
        block = DenseBlock(num_layers=12, in_channels=128, growth_rate=32)
        assert len(block) == 12


# ── TransitionLayer ───────────────────────────────────────────────────────────

class TestTransitionLayer:
    def test_output_shape_default_compression(self):
        # θ=0.5 → channels halved, spatial halved by AvgPool
        trans = TransitionLayer(in_channels=256, compression=0.5)
        x = torch.zeros(2, 256, 28, 28)
        out = trans(x)
        assert out.shape == (2, 128, 14, 14)

    def test_output_channels_custom_compression(self):
        trans = TransitionLayer(in_channels=512, compression=0.25)
        x = torch.zeros(1, 512, 8, 8)
        out = trans(x)
        assert out.shape[1] == 128

    def test_attribute_names_match_torchvision(self):
        """norm/conv/pool must match torchvision's _Transition state-dict keys."""
        trans = TransitionLayer(in_channels=256)
        for attr in ('norm', 'conv', 'pool'):
            assert hasattr(trans, attr), f"TransitionLayer missing attribute '{attr}'"


# ── DenseNet121 architecture ──────────────────────────────────────────────────

class TestDenseNet121Architecture:
    def test_output_shape_standard_input(self):
        model = DenseNet121(num_classes=14)
        model.eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 14)

    def test_output_shape_small_input(self):
        """adaptive_avg_pool allows inputs smaller than 224×224."""
        model = DenseNet121(num_classes=5)
        model.eval()
        x = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_out_channels_default_config(self):
        model = DenseNet121(num_classes=10)
        assert model.out_channels == 1024

    def test_no_nan_in_output(self):
        model = DenseNet121(num_classes=8)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_features_sequential_exists(self):
        model = DenseNet121(num_classes=2)
        assert hasattr(model, 'features')
        assert isinstance(model.features, torch.nn.Sequential)

    def test_stem_module_names(self):
        """conv0/norm0/relu0/pool0 must exist inside self.features."""
        model = DenseNet121(num_classes=2)
        for name in ('conv0', 'norm0', 'relu0', 'pool0'):
            assert hasattr(model.features, name), \
                f"model.features missing '{name}'"

    def test_denseblock_and_transition_names(self):
        """denseblock1-4 and transition1-3 must exist inside self.features."""
        model = DenseNet121(num_classes=2)
        for i in range(1, 5):
            assert hasattr(model.features, f'denseblock{i}')
        for i in range(1, 4):
            assert hasattr(model.features, f'transition{i}')

    def test_norm5_exists(self):
        model = DenseNet121(num_classes=2)
        assert hasattr(model.features, 'norm5')

    def test_classifier_output_dim(self):
        model = DenseNet121(num_classes=42)
        assert model.classifier.out_features == 42


# ── Torchvision key compatibility ─────────────────────────────────────────────

class TestTorchvisionKeyCompatibility:
    """Verify state-dict keys align with torchvision's DenseNet-121 exactly,
    so load_state_dict works without any key remapping."""

    @pytest.fixture(scope='class')
    def torchvision_backbone_keys(self):
        pytest.importorskip('torchvision')
        from torchvision.models import densenet121
        sd = densenet121(weights=None).state_dict()
        return {k for k in sd if not k.startswith('classifier')}

    def test_all_torchvision_backbone_keys_present(self, torchvision_backbone_keys):
        model = DenseNet121(num_classes=1000)
        our_keys = set(model.state_dict().keys())
        missing = torchvision_backbone_keys - our_keys
        assert not missing, f"Keys in torchvision but not in our model:\n{sorted(missing)}"

    def test_no_extra_backbone_keys(self, torchvision_backbone_keys):
        model = DenseNet121(num_classes=1000)
        our_backbone_keys = {k for k in model.state_dict() if not k.startswith('classifier')}
        extra = our_backbone_keys - torchvision_backbone_keys
        assert not extra, f"Extra keys in our model not in torchvision:\n{sorted(extra)}"


# ── Pretrained weight loading ─────────────────────────────────────────────────

class TestLoadImagenetWeights:
    @pytest.fixture(scope='class')
    def pretrained_sd(self):
        pytest.importorskip('torchvision')
        from torchvision.models import densenet121, DenseNet121_Weights
        return densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).state_dict()

    def test_load_does_not_raise(self):
        model = DenseNet121(num_classes=14)
        model.load_imagenet_weights(strict_backbone=True)  # must not raise

    def test_backbone_weights_are_loaded(self, pretrained_sd):
        """A spot-checked backbone parameter must equal the torchvision value."""
        model = DenseNet121(num_classes=14)
        model.load_imagenet_weights()
        our_w = model.state_dict()['features.conv0.weight']
        tv_w = pretrained_sd['features.conv0.weight']
        assert torch.allclose(our_w, tv_w), "features.conv0.weight does not match torchvision"

    def test_classifier_is_not_overwritten(self):
        """Classifier head must differ from the ImageNet 1000-class head."""
        model = DenseNet121(num_classes=14)
        model.load_imagenet_weights()
        assert model.classifier.out_features == 14

    def test_norm5_weights_loaded(self, pretrained_sd):
        model = DenseNet121(num_classes=14)
        model.load_imagenet_weights()
        our_w = model.state_dict()['features.norm5.weight']
        tv_w = pretrained_sd['features.norm5.weight']
        assert torch.allclose(our_w, tv_w), "features.norm5.weight does not match torchvision"

    def test_denselayer_weights_loaded(self, pretrained_sd):
        model = DenseNet121(num_classes=14)
        model.load_imagenet_weights()
        key = 'features.denseblock1.denselayer1.conv1.weight'
        our_w = model.state_dict()[key]
        tv_w = pretrained_sd[key]
        assert torch.allclose(our_w, tv_w), f"{key} does not match torchvision"

    def test_transition_weights_loaded(self, pretrained_sd):
        model = DenseNet121(num_classes=14)
        model.load_imagenet_weights()
        key = 'features.transition2.conv.weight'
        our_w = model.state_dict()[key]
        tv_w = pretrained_sd[key]
        assert torch.allclose(our_w, tv_w), f"{key} does not match torchvision"
