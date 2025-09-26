from torchvision.models import GoogLeNet_Weights
from base_extractor import BaseExtractor
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class GoogLeNetExtractor(BaseExtractor):
    """
    Feature extractor using a pretrained GoogLeNet model.
    """

    def build_model(self):
        """
        Loads GoogLeNet, disables auxiliary classifiers, and removes the final layer.

        Returns:
            nn.Sequential: GoogLeNet backbone without the classifier.
        """
        model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=False, transform_input=False)
        return nn.Sequential(*list(model.children())[:-1])

    def build_transform(self):
        """
        Defines preprocessing for GoogLeNet using ImageNet normalization.

        Returns:
            torchvision.transforms.Compose: Transform pipeline for GoogLeNet.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])