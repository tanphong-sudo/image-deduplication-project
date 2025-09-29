from src.feature_extraction.base_extractor import BaseExtractor
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class ViTExtractor(BaseExtractor):
    """
    Feature extractor using a pretrained Vision Transformer (ViT-B/16) model.
    """

    def build_model(self):
        """
        Loads the ViT-B/16 model and removes the final classification head.

        Returns:
            torch.nn.Module: ViT backbone without the classifier head.
        """
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = torch.nn.Identity() 
        return model

    def build_transform(self):
        """
        Defines preprocessing for ViT-B/16.

        Returns:
            torchvision.transforms.Compose: Transform pipeline for ViT.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    