from abc import ABC, abstractmethod
import torch
from PIL import Image
import numpy as np

class BaseExtractor(ABC):
    """
    Abstract base class for feature extractors using pretrained CNN models.
    Handles device selection, model initialization, and image preprocessing.
    """

    def __init__(self, device=None):
        """
        Initializes the computation device, model, and preprocessing pipeline.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.build_model().to(self.device)
        self.transform = self.build_transform()

    @abstractmethod
    def build_model(self):
        """
        Constructs the CNN model for feature extraction.
        Must remove the final classification layer to return raw feature vectors.
        """
        pass

    @abstractmethod
    def build_transform(self):
        """
        Defines the preprocessing pipeline for input images.
        Should include resizing, cropping, normalization, and tensor conversion.
        """
        pass

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Applies preprocessing to a batch of images and stacks them into a tensor.
        """
        if self.transform is None:
            raise NotImplementedError("Transform has not been defined.")
        tensors = [self.transform(img) for img in images]
        batch_tensor = torch.stack(tensors)  # chưa chuyển device ở đây
        return batch_tensor

    def extract_features_batch(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """
        Extracts feature vectors from a batch of images using mini-batches.
        Supports GPU if available, else CPU.

        Args:
            images (list of PIL.Image): List of input images.
            batch_size (int): Number of images per mini-batch.

        Returns:
            np.ndarray: Feature vectors with shape [B, D] or [B, C, H, W], depending on the model.
        """
        self.model.eval()
        features_list = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                x = self.preprocess_batch(batch).to(self.device)  # chuyển lên đúng device
                batch_features = self.model(x)
                features_list.append(batch_features.cpu())  # lưu về CPU để ghép

        features = torch.cat(features_list, dim=0)
        return features.squeeze().numpy()
