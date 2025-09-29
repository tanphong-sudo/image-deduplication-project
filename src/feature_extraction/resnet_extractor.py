from torchvision.models import ResNet50_Weights
from src.feature_extraction.base_extractor import BaseExtractor
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ResNetExtractor(BaseExtractor):
    """
    Feature extractor using a pretrained ResNet50 model.
    """

    def build_model(self):
        """
        Loads the ResNet50 model and removes the final fully connected layer.

        Returns:
            nn.Sequential: ResNet50 backbone without the classifier.
        """
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        return nn.Sequential(*list(model.children())[:-1])

    def build_transform(self):
        """
        Defines preprocessing for ResNet50 using ImageNet normalization.

        Returns:
            torchvision.transforms.Compose: Transform pipeline for ResNet50.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
