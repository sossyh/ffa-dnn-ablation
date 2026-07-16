import torch
import torchvision.models as models
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_resnet50_encoder(device: str = "cpu"):
    """
    Returns a pretrained ResNet50 model configured as a feature encoder,
    plus a preprocessing transform suitable for ImageNet-pretrained models.
    """
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)

    # Remove final classifier so output is a 2048-d feature vector
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
    raise NotImplementedError(
        "VGGFace2 encoder not implemented yet. "
        "Decide on a specific VGGFace2 PyTorch implementation and weights."
    )