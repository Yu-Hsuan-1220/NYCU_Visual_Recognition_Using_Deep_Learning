import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt101_32X8D_Weights,
    resnet50,
    resnet101,
    resnet152,
    resnext101_32x8d
)


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier with modified final FC layer.
    Supports ResNet50, ResNet101, and ResNet152 with V2 weights.
    """

    def __init__(
        self,
        backbone='resnet50',
        num_classes=100,
        pretrained=True,
        dropout=0.5,
        use_deeper_fc=False
    ):
        """
        Args:
            backbone (str): 'resnet50', 'resnet101', or 'resnet152'
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet V2 pretrained weights
            dropout (float): Dropout rate before final FC layer
            use_deeper_fc (bool): Whether to use a deeper FC layer
        """
        super(ResNetClassifier, self).__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
                print("Loading ResNet50 with ImageNet V2 pretrained weights")
            else:
                weights = None
                print("Loading ResNet50 without pretrained weights")
            self.backbone = resnet50(weights=weights)

        elif backbone == 'resnet101':
            if pretrained:
                weights = ResNet101_Weights.IMAGENET1K_V2
                print("Loading ResNet101 with ImageNet V2 pretrained weights")
            else:
                weights = None
                print("Loading ResNet101 without pretrained weights")
            self.backbone = resnet101(weights=weights)

        elif backbone == 'resnet152':
            if pretrained:
                weights = ResNet152_Weights.IMAGENET1K_V2
                print("Loading ResNet152 with ImageNet V2 pretrained weights")
            else:
                weights = None
                print("Loading ResNet152 without pretrained weights")
            self.backbone = resnet152(weights=weights)

        elif backbone == 'resnext101':
            if pretrained:
                weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2
                print("Loading ResNeXt101_32x8d  \
                with ImageNet V2 pretrained weights")
            else:
                weights = None
                print("Loading ResNeXt101_32x8d without pretrained weights")
            self.backbone = resnext101_32x8d(weights=weights)
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                "Choose 'resnet50', 'resnet101', 'resnet152', or 'resnext101'"
            )

        in_features = self.backbone.fc.in_features

        if use_deeper_fc:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features // 2, num_classes)
            )
        else:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        print(f"Modified FC layer: {in_features} -> {num_classes}")

    def forward(self, x):
        return self.backbone(x)

    def get_params_for_optimizer(self, lr_backbone=1e-4, lr_fc=1e-3):
        """
        Get parameter groups with different learning rates.
        Lower LR for backbone, higher LR for new FC layer.
        """
        backbone_params = []
        fc_params = []

        for name, param in self.backbone.named_parameters():
            if 'fc' in name:
                fc_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': fc_params, 'lr': lr_fc}
        ]

        return param_groups

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("Backbone frozen. Only FC layer will be trained.")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. All parameters will be trained.")

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_p = sum(p.numel() for p in self.parameters())
        trainable_p = sum(p.numel() for p in self.parameters()
                          if p.requires_grad)
        return total_p, trainable_p


def get_model(
    backbone='resnet50',
    num_classes=100,
    pretrained=True,
    dropout=0.5,
    use_deeper_fc=False
):
    """
    Factory function to create a ResNet classifier.
    """
    model = ResNetClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        use_deeper_fc=use_deeper_fc
    )

    total, trainable = model.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    return model
