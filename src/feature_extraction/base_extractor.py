from abc import ABC, abstractmethod
import torch
from PIL import Image

class BaseExtractor(ABC):
    """
    Abstract base class for feature extractors using pretrained CNN models.
    Handles device selection, model initialization, and image preprocessing.
    """

    def __init__(self, device=None):
        """
        Initializes the computation device, model, and preprocessing pipeline.

        Args:
            device (str, optional): Target device for computation.
                If None, defaults to GPU if available, otherwise CPU.
                Supported values include:
                - "cpu": Central Processing Unit
                - "cuda": NVIDIA GPU with CUDA support
                - "cuda:N": Specific GPU index (e.g., "cuda:0")
                - "mps": Apple Silicon GPU via Metal (macOS only)
                - "xpu": Intel GPU via oneAPI (experimental)
                - "hip": AMD GPU via ROCm (Linux only)
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.build_model().to(self.device)
        self.transform = self.build_transform()

    @abstractmethod
    def build_model(self):
        """
        Constructs the CNN model for feature extraction.
        Must remove the final classification layer to return raw feature vectors.

        Returns:
            nn.Module: A PyTorch model without the final classifier.
        """
        pass

    @abstractmethod
    def build_transform(self):
        """
        Defines the preprocessing pipeline for input images.
        Should include resizing, cropping, normalization, and tensor conversion.

        Returns:
            torchvision.transforms.Compose: A composed transform for image preprocessing.
        """
        pass

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Applies preprocessing to a batch of images and stacks them into a tensor.

        Args:
            images (list of PIL.Image): List of input images.

        Returns:
            torch.Tensor: A batch tensor of shape [B, 3, H, W] on the selected device.
        """
        if self.transform is None:
            raise NotImplementedError("Transform has not been defined.")

        tensors = [self.transform(img) for img in images]
        batch_tensor = torch.stack(tensors).to(self.device)
        return batch_tensor

    def extract_features_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Extracts feature vectors from a batch of preprocessed images.

        Args:
            images (list of PIL.Image): List of input images.

        Returns:
            numpy.ndarray: Feature vectors with shape [B, D] or [B, C, H, W], depending on the model.
        """
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess_batch(images)
            features = self.model(x)
        return features.squeeze().cpu().numpy()



