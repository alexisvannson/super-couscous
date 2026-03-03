"""
Class Activation Map (CAM) generation for DenseNet-121.

Implements the weighted-sum formula from Rajpurkar et al. / Zhou et al. (2016):

    Mc = Σk  wc,k · fk

where fk  is the k-th feature map of the final conv block  (shape 7×7 for 224px input)
and   wc,k is the classifier weight for class c and feature map k.

A forward hook captures fk just before Global Average Pooling, so no model
surgery is needed.

Usage:
    python scripts/cam.py densenet \\
        --weights models/checkpoints/best.pth \\
        --image   data/sample/images/00000013_005.png \\
        --labels  Pneumothorax Infiltration        # leave out to show all
        --out     cam_output/
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.training import create_model, load_config, get_transforms


# ── Hook ─────────────────────────────────────────────────────────────────────

class CAMGenerator:
    """
    Attaches a forward hook to `model.features` to capture the final
    convolutional feature maps (before GAP) and computes CAMs.
    """

    def __init__(self, model):
        self.model = model
        self._feature_maps = None  # (1, C, H, W)
        self._hook = model.features.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        # output is after norm5 (BN), before relu — apply relu to match forward()
        self._feature_maps = F.relu(output, inplace=False).detach()

    def generate(self, image_tensor: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Run a forward pass and return a CAM for every class.

        Args:
            image_tensor: (1, 3, H, W) preprocessed image on the same device as the model.

        Returns:
            dict mapping label_name → cam (np.ndarray, shape H×W, values in [0,1])
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor)           # (1, num_classes)
            probs  = torch.sigmoid(logits).squeeze(0)   # (num_classes,)

        feature_maps = self._feature_maps.squeeze(0)    # (C, h, w)  e.g. (1024, 7, 7)
        weights = self.model.classifier.weight           # (num_classes, C)

        # Mc = Σk  wc,k · fk  →  matmul: (num_classes, C) × (C, h*w) → (num_classes, h*w)
        h, w = feature_maps.shape[1], feature_maps.shape[2]
        cam_flat = weights @ feature_maps.flatten(1)    # (num_classes, h*w)
        cam_flat = cam_flat.reshape(-1, h, w)           # (num_classes, h, w)

        # Upsample to input resolution
        input_h, input_w = image_tensor.shape[2], image_tensor.shape[3]
        cam_up = F.interpolate(
            cam_flat.unsqueeze(0),                      # (1, num_classes, h, w)
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                    # (num_classes, H, W)

        label_names = self.model._label_names if hasattr(self.model, "_label_names") else \
                      [str(i) for i in range(cam_up.shape[0])]

        cams = {}
        for i, name in enumerate(label_names):
            c = cam_up[i].cpu().numpy()
            c = np.maximum(c, 0)                        # ReLU
            if c.max() > 0:
                c = c / c.max()                         # normalise to [0, 1]
            cams[name] = c

        return cams, probs.cpu().numpy()

    def remove(self):
        self._hook.remove()


# ── Visualisation ─────────────────────────────────────────────────────────────

def overlay_cam(original: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a CAM heatmap on the original image (PIL → PIL)."""
    try:
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for CAM visualisation: pip install matplotlib")

    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = Image.fromarray(cm.jet(cam_uint8)[:, :, :3].astype(np.uint8) * 1)

    # cm.jet returns floats in [0,1] — convert properly
    heatmap_rgb = Image.fromarray((cm.jet(cam_uint8)[:, :, :3] * 255).astype(np.uint8))
    heatmap_rgb = heatmap_rgb.resize(original.size, Image.BILINEAR)

    original_rgb = original.convert("RGB")
    blended = Image.blend(original_rgb, heatmap_rgb, alpha=alpha)
    return blended


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CAMs for a chest X-ray image")
    parser.add_argument("model",     type=str, help="Model name (must match configs/)")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image",   type=str, required=True, help="Path to input image")
    parser.add_argument("--labels",  type=str, nargs="*",     help="Label(s) to visualise (default: all)")
    parser.add_argument("--out",     type=str, default="cam_output", help="Output directory")
    parser.add_argument("--alpha",   type=float, default=0.5, help="Heatmap blend factor")
    args = parser.parse_args()

    config = load_config(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(args.model, config)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # Attach label names to model so CAMGenerator can use them
    from scripts.thedataloader import ChestXrayDataset
    import itertools, pandas as pd
    meta = pd.read_csv(config["data"]["labels_csv"])
    label_names = sorted(set(itertools.chain.from_iterable(meta["Finding Labels"].str.split("|"))))
    model._label_names = label_names

    # Preprocess image
    transform = get_transforms(config.get("transform", {}))
    original  = Image.open(args.image)
    tensor    = transform(original.convert("RGB")).unsqueeze(0).to(device)

    # Generate CAMs
    gen = CAMGenerator(model)
    cams, probs = gen.generate(tensor)
    gen.remove()

    # Filter to requested labels
    targets = args.labels if args.labels else label_names

    os.makedirs(args.out, exist_ok=True)
    print(f"\n{'Label':<30}  {'P(y=1)':>8}  {'Saved'}")
    print("-" * 60)
    for label in targets:
        if label not in cams:
            print(f"  Warning: '{label}' not in label set — skipping")
            continue
        prob = probs[label_names.index(label)]
        blended = overlay_cam(original, cams[label], alpha=args.alpha)
        fname = os.path.join(args.out, f"cam_{label.replace(' ', '_')}.png")
        blended.save(fname)
        print(f"  {label:<30}  {prob:>8.3f}  {fname}")
    print()


if __name__ == "__main__":
    main()
