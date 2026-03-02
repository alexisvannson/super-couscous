import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from models.CNN import CNN, ConvolutionBlock


# ── ConvolutionBlock ──────────────────────────────────────────────────────────

class TestConvolutionBlock:
    def test_output_shape_no_padding(self):
        # (H - K) / S + 1 = (8 - 3) / 1 + 1 = 6, then maxpool /2 → 3
        block = ConvolutionBlock(in_dim=3, out_dim=16, kernel_size=3)
        x = torch.zeros(2, 3, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 3, 3)

    def test_output_shape_same_padding(self):
        # padding=1, kernel=3, stride=1 → same size → 8, maxpool /2 → 4
        block = ConvolutionBlock(in_dim=3, out_dim=16, kernel_size=3, padding=1)
        x = torch.zeros(2, 3, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 4, 4)

    def test_channels_change(self):
        block = ConvolutionBlock(in_dim=1, out_dim=64, kernel_size=3, padding=1)
        x = torch.zeros(1, 1, 16, 16)
        out = block(x)
        assert out.shape[1] == 64


# ── CNN flattened size ────────────────────────────────────────────────────────

class TestCNNFlattenedSize:
    def test_default_config_flattened_size(self):
        """
        kernel=3, stride=1, padding=1 → same-size conv each block, maxpool /2.
        224 → 112 → 56 → 28, last channels=128 → 128*28*28 = 100352
        """
        model = CNN(
            nblocks=3, in_dim=3, out_dim=10,
            kernel_size=3, stride=1, padding=1,
            layers_out=[32, 64, 128],
            input_image_size=(224, 224),
        )
        expected = 128 * 28 * 28  # 100352
        assert model.fc.in_features == expected

    def test_two_blocks_no_padding(self):
        """
        kernel=3, stride=1, padding=0, input=32x32.
        Block1: (32-3+1)=30 → maxpool → 15
        Block2: (15-3+1)=13 → maxpool → 6 (floor)
        channels=64 → 64*6*6 = 2304
        """
        model = CNN(
            nblocks=2, in_dim=1, out_dim=5,
            kernel_size=3, stride=1, padding=0,
            layers_out=[32, 64],
            input_image_size=(32, 32),
        )
        expected = 64 * 6 * 6
        assert model.fc.in_features == expected

    def test_single_block(self):
        """Single block: kernel=3, padding=1, input=16x16 → same → maxpool → 8x8, channels=8"""
        model = CNN(
            nblocks=1, in_dim=3, out_dim=4,
            kernel_size=3, stride=1, padding=1,
            layers_out=[8],
            input_image_size=(16, 16),
        )
        expected = 8 * 8 * 8
        assert model.fc.in_features == expected


# ── CNN forward pass ──────────────────────────────────────────────────────────

class TestCNNForward:
    def test_output_shape_default_config(self):
        model = CNN(
            nblocks=3, in_dim=3, out_dim=10,
            kernel_size=3, stride=1, padding=1,
            layers_out=[32, 64, 128],
            input_image_size=(224, 224),
        )
        model.eval()
        x = torch.zeros(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)

    def test_output_shape_small_input(self):
        model = CNN(
            nblocks=2, in_dim=1, out_dim=5,
            kernel_size=3, stride=1, padding=0,
            layers_out=[32, 64],
            input_image_size=(32, 32),
        )
        model.eval()
        x = torch.zeros(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_batch_size_one(self):
        model = CNN(
            nblocks=1, in_dim=3, out_dim=2,
            kernel_size=3, stride=1, padding=1,
            layers_out=[16],
            input_image_size=(16, 16),
        )
        model.eval()
        x = torch.zeros(1, 3, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_output_is_tensor(self):
        model = CNN(
            nblocks=1, in_dim=3, out_dim=3,
            kernel_size=3, stride=1, padding=1,
            layers_out=[8],
            input_image_size=(16, 16),
        )
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert not torch.isnan(out).any()
