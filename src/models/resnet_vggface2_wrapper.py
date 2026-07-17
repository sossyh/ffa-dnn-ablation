# src/models/resnet_vggface2_wrapper.py

import os
import pickle

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# VGGFace2 weights are Caffe conversions. They expect BGR channel order,
# 0-255 scale, per-channel mean subtraction, and NO std division.

# VERIFY these values against datasets/vgg_face2.py in cydonia999/VGGFace2-pytorch
# before trusting any downstream result.
VGGFACE2_MEAN_BGR = [91.4953, 103.8827, 131.0912]
VGGFACE2_STD = [1.0, 1.0, 1.0]  # mean subtraction only, no scaling

N_CLASSES_VGGFACE2 = 8631

# Google Drive file IDs, from the cydonia999/VGGFace2-pytorch README.
VGGFACE2_WEIGHT_IDS = {
    "resnet50_scratch": "1gy9OJlVfBulWkIEnZhGpOLu084RgHw39",
    "resnet50_ft": "1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU",
}

# Weights are cached relative to the repo root, not the caller's working
# directory, so running from src/models does not scatter checkpoints.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WEIGHT_DIR = os.path.join(_REPO_ROOT, "data", "weights")


def get_resnet50_encoder(device: str = "cpu"):
    """
    Returns a pretrained ResNet50 model configured as a feature encoder,
    plus a preprocessing transform suitable for ImageNet-pretrained models.
    """
    # Load pretrained ResNet50 weights from TorchVision.
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)

    # Replace the final classification layer with Identity to get feature vectors.
    model.fc = torch.nn.Identity()

    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return model, transform


def _download_vggface2_weights(arch: str = "resnet50_scratch",
                               out_dir: str = None):
    """
    Downloads Caffe-converted VGGFace2 weights from Google Drive.
    """
    import gdown

    out_dir = out_dir or DEFAULT_WEIGHT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{arch}.pkl")
    if not os.path.exists(path):
        gdown.download(id=VGGFACE2_WEIGHT_IDS[arch], output=path, quiet=False)
    return path


def _load_vggface2_state_dict(weight_path: str):
    """
    Loads a converted Caffe checkpoint and normalizes it into a state dict.

    These conversions date from 2018 and their packing varies: raw state dict
    versus {"state_dict": ...}, numpy arrays versus tensors, "module." prefixes.
    This inspects the object rather than assuming a layout.

    Parameters
    ----------
    weight_path : str
        Path to the .pkl checkpoint.

    Returns
    -------
    dict
        State dict with tensor values and normalized key names.
    """
    with open(weight_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {
        k.replace("module.", ""): (
            torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        )
        for k, v in sd.items()
    }


def get_vggface2_encoder(device: str = "cpu", arch: str = "resnet50_scratch",
                         weight_path: str = None):
    """
    Returns a VGGFace2-pretrained ResNet50 configured as a feature encoder,
    plus a preprocessing transform suitable for the Caffe-converted weights.
    """
    weight_path = weight_path or _download_vggface2_weights(arch)
    state_dict = _load_vggface2_state_dict(weight_path)

    model = models.resnet50(weights=None, num_classes=N_CLASSES_VGGFACE2)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    conv_missing = [k for k in missing if not k.startswith("fc.")]
    if conv_missing:
        raise RuntimeError(
            f"{len(conv_missing)} conv/bn weights failed to load, e.g. "
            f"{conv_missing[:5]}. The checkpoint's key naming does not match "
            f"torchvision's resnet50. Inspect the checkpoint keys and remap "
            f"them before proceeding."
        )
    if unexpected:
        print(f"[note] {len(unexpected)} unexpected keys ignored: {unexpected[:5]}")

    print(f"[ok] VGGFace2 weights loaded from {weight_path} "
          f"({len(state_dict)} tensors)")

    # Replace the final classification layer with Identity to get feature vectors.
    model.fc = torch.nn.Identity()

    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),                                    # RGB, [0, 1]
            transforms.Lambda(lambda t: t[[2, 1, 0], :, :] * 255.0),  # BGR, [0, 255]
            transforms.Normalize(mean=VGGFACE2_MEAN_BGR, std=VGGFACE2_STD),
        ]
    )

    return model, transform


def assert_not_imagenet(model):
    """
    Fails if `model` carries torchvision's ImageNet weights.
    """
    for version in ("IMAGENET1K_V1", "IMAGENET1K_V2"):
        ref = models.resnet50(weights=getattr(models.ResNet50_Weights, version))
        if torch.allclose(model.conv1.weight.cpu(), ref.conv1.weight):
            raise RuntimeError(
                f"This model carries torchvision {version} weights, not "
                f"VGGFace2. The face-versus-ImageNet contrast would be "
                f"ImageNet versus ImageNet."
            )
    print("[ok] weights are not torchvision ImageNet")