"""
ResNet50-equivalent (VGGFace2-trained) wrapper for extracting
layer-wise activations, via facenet-pytorch's InceptionResnetV1.

Note on architecture: InceptionResnetV1 is an Inception-based
architecture, not a literal ResNet50, but it is the standard,
easiest-to-access VGGFace2-pretrained model in Python. Layer names
below are based on the known facenet-pytorch source structure.

Before trusting results, run this diagnostic once and compare against
ASSUMED_LAYERS below:

    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2")
    for name, module in model.named_children():
        print(name, "->", type(module).__name__)

If names differ, edit ASSUMED_LAYERS to match.
"""

import torch
import torchvision.transforms as transforms


ASSUMED_LAYERS = ["repeat_1", "repeat_2", "repeat_3", "block8", "avgpool_1a"]


class ResNetVGGFace2Wrapper:
    def __init__(self, layers=None, device=None):
        """
        Parameters
        ----------
        layers : list of str, optional
            Which named modules to hook. Defaults to ASSUMED_LAYERS.
        device : str, optional
            "cuda" or "cpu". Auto-detects if not given.
        """
        from facenet_pytorch import InceptionResnetV1

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = InceptionResnetV1(pretrained="vggface2")
        self.model.eval()
        self.model.to(self.device)

        available_modules = dict(self.model.named_modules())
        requested_layers = layers or ASSUMED_LAYERS

        self.layers = []
        self.activations = {}

        for layer_name in requested_layers:
            if layer_name in available_modules:
                available_modules[layer_name].register_forward_hook(
                    self._make_hook(layer_name)
                )
                self.layers.append(layer_name)
            else:
                print(
                    f"WARNING: layer '{layer_name}' not found in model. "
                    f"Run the diagnostic in this file's docstring and "
                    f"update ASSUMED_LAYERS. Skipping this layer."
                )

        if len(self.layers) == 0:
            raise RuntimeError(
                "No valid layers were hooked. Run the diagnostic in "
                "this file's docstring and fix ASSUMED_LAYERS."
            )

        print(f"ResNetVGGFace2Wrapper: hooked layers {self.layers}")

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),  # facenet-pytorch's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

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
            Maps layer name -> activation vector averaged across
            frames, flattened to shape (num_units,).
        """
        batch = torch.stack([self.preprocess(f) for f in frames])
        batch = batch.to(self.device)

        with torch.no_grad():
            self.model(batch)

        result = {}
        for layer_name in self.layers:
            act = self.activations[layer_name]
            act = act.mean(dim=0).flatten().numpy()
            result[layer_name] = act

        return result
