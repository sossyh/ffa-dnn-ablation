# src/models/resnet_vggface2_wrapper.py

import torch
import torchvision.models as models
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_resnet50_encoder(device: str = "cpu"):
    """
    Returns a pretrained ResNet50 model configured as a feature encoder,
    plus a preprocessing transform suitable for ImageNet-pretrained models.

    Parameters
    ----------
    device : str
        "cpu" or "cuda" depending on available hardware.

    Returns
    -------
    model : torch.nn.Module
        ResNet50 with the final fully-connected layer replaced by Identity.
    transform : torchvision.transforms.Compose
        Preprocessing pipeline: resize, crop, to tensor, normalize.
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


def get_vggface2_encoder(device: str = "cpu"):
    """
    Placeholder for VGGFace2-based encoder.

    This function should eventually load a face-specific model (e.g. SENet50
    trained on VGGFace2) and return it as a feature encoder, along with an
    appropriate preprocessing transform.

    For now, it raises NotImplementedError so the rest of the code can
    refer to it without silently doing the wrong thing.
    """
    raise NotImplementedError(
        "VGGFace2 encoder not implemented yet. "
        "Decide on a specific VGGFace2 PyTorch implementation and weights."
    )