from src.feature_extraction.base_extractor import BaseExtractor
from torchvision.models import ConvNeXt_Tiny_Weights
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms



class ConvNeXtTinyExtractor(BaseExtractor):
    """
    Feature extractor using a pretrained ConvNeXt-Tiny model.
    """

    def build_model(self):
        """
        Loads the ConvNeXt-Tiny model and removes the final classifier.

        Returns:
            nn.Sequential: ConvNeXt-Tiny backbone without the classifier.
        """
        model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        return nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def build_transform(self):
        """
        Defines preprocessing for ConvNeXt-Tiny using ImageNet normalization.

        Returns:
            torchvision.transforms.Compose: Transform pipeline for ConvNeXt-Tiny.
        """
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        return weights.transforms()
