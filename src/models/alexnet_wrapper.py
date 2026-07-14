# src/models/alexnet_wrapper.py

import torch
import torchvision.models as models
from torchvision import transforms

# Standard ImageNet normalization used by TorchVision pretrained models
# (same for AlexNet, ResNet, etc.).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_alexnet_encoder(device: str = "cpu"):
    """
    Returns a pretrained AlexNet model configured as a feature encoder,
    plus a preprocessing transform suitable for ImageNet-pretrained models.

    Parameters
    ----------
    device : str
        "cpu" or "cuda" depending on available hardware.

    Returns
    -------
    model : torch.nn.Module
        AlexNet with the classifier replaced by Identity, outputs feature vectors.
    transform : torchvision.transforms.Compose
        Preprocessing pipeline: resize, crop, to tensor, normalize.
    """
    # Load pretrained AlexNet weights from TorchVision.
    # For latest TorchVision, the recommended API uses `weights=...`.
    # If your installed version doesn’t support this, fall back to pretrained=True.
    try:
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)
    except AttributeError:
        model = models.alexnet(pretrained=True)

    # Replace the classifier block with Identity so we get high-level features.
    model.classifier = torch.nn.Identity()

    model.eval()
    model.to(device)

    # Preprocessing: resize, center crop, convert to tensor, normalize.
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return model, transform