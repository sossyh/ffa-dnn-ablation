# test_models.py

import torch

from src.models.alexnet_wrapper import get_alexnet_encoder
from src.models.resnet_vggface2_wrapper import get_resnet50_encoder

def main():
    device = "cpu"

    alexnet, alex_transform = get_alexnet_encoder(device=device)
    resnet, resnet_transform = get_resnet50_encoder(device=device)

    x = torch.randn(1, 3, 224, 224).to(device)

    alex_features = alexnet(x)
    resnet_features = resnet(x)

    print("AlexNet features shape:", alex_features.shape)
    print("ResNet50 features shape:", resnet_features.shape)


if __name__ == "__main__":
    main()