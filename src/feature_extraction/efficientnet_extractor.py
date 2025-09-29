from torchvision.models import EfficientNet_B0_Weights
from src.feature_extraction.base_extractor import BaseExtractor
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class EfficientNetExtractor(BaseExtractor):
    """
    Feature extractor using a pretrained EfficientNet-B0 model.
    """

    def build_model(self):
        """
        Loads EfficientNet-B0 and returns the feature extraction layers.

        Returns:
            nn.Sequential: EfficientNet-B0 backbone with avgpool.
        """
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        return nn.Sequential(
            model.features,
            model.avgpool
        )

    def build_transform(self):
        """
        Defines preprocessing for EfficientNet-B0 with 240x240 cropping.

        Returns:
            torchvision.transforms.Compose: Transform pipeline for EfficientNet-B0.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
