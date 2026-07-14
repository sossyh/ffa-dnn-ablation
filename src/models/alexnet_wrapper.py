"""
AlexNet wrapper for extracting layer-wise activations.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


LAYER_MAP = {
    "conv1": ("features", 0),
    "conv2": ("features", 3),
    "conv3": ("features", 6),
    "conv4": ("features", 8),
    "conv5": ("features", 10),
    "fc6": ("classifier", 1),
    "fc7": ("classifier", 4),
    "fc8": ("classifier", 6),
}


class AlexNetWrapper:
    def __init__(self, layers=None, device=None):
        """
        Parameters
        ----------
        layers : list of str, optional
            Which layers to extract activations from.
            Defaults to all 8 standard layers.
        device : str, optional
            "cuda" or "cpu". Auto-detects if not given.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers or list(LAYER_MAP.keys())

        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)

        self.activations = {}
        self._register_hooks()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu()
            return hook

        for layer_name in self.layers:
            submodule_name, idx = LAYER_MAP[layer_name]
            submodule = getattr(self.model, submodule_name)
            submodule[idx].register_forward_hook(get_hook(layer_name))

    def extract(self, frames):
        """
        Extracts activations for a batch of frames.

        Parameters
        ----------
        frames : np.ndarray
            Array of shape (num_frames, H, W, 3), dtype uint8.

        Returns
        -------
        dict
            Maps layer name -> activation tensor averaged across frames,
            flattened to shape (num_units,).
        """
        batch = torch.stack([self.preprocess(f) for f in frames])
        batch = batch.to(self.device)

        with torch.no_grad():
            self.model(batch)

        result = {}
        for layer_name in self.layers:
            act = self.activations[layer_name]
            # average across the sampled frames, then flatten
            act = act.mean(dim=0).flatten().numpy()
            result[layer_name] = act

        return result
