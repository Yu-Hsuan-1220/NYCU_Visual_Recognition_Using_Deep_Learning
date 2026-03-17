import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights, 
    ResNet101_Weights,
    ResNet152_Weights,
    resnet50, 
    resnet101,
    resnet152
)


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier with modified final FC layer for custom number of classes.
    Supports ResNet50 and ResNet101 with ImageNet V2 pretrained weights.
    """
    
    def __init__(self, backbone='resnet50', num_classes=100, pretrained=True, dropout=0.5, use_deeper_fc=False):
        """
        Args:
            backbone (str): 'resnet50' or 'resnet101'
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use ImageNet V2 pretrained weights
            dropout (float): Dropout rate before final FC layer
            use_deeper_fc (bool): Whether to use a deeper FC layer
        """
        super(ResNetClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone with ImageNet V2 pretrained weights
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
                print(f"Loading ResNet50 with ImageNet V2 pretrained weights")
            else:
                weights = None
                print(f"Loading ResNet50 without pretrained weights")
            self.backbone = resnet50(weights=weights)
            
        elif backbone == 'resnet101':
            if pretrained:
                weights = ResNet101_Weights.IMAGENET1K_V2
                print(f"Loading ResNet101 with ImageNet V2 pretrained weights")
            else:
                weights = None
                print(f"Loading ResNet101 without pretrained weights")
            self.backbone = resnet101(weights=weights)
        elif backbone == 'resnet152':
            if pretrained:
                weights = ResNet152_Weights.IMAGENET1K_V2
                print(f"Loading ResNet152 with ImageNet V2 pretrained weights")
            else:
                weights = None
                print(f"Loading ResNet152 without pretrained weights")
            self.backbone = resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'resnet101'")
        
        # Get the number of features from the original FC layer
        in_features = self.backbone.fc.in_features
        
        # Replace the final FC layer with custom classifier
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
        Lower learning rate for pretrained backbone, higher for new FC layer.
        
        Args:
            lr_backbone: Learning rate for backbone layers
            lr_fc: Learning rate for FC layer
            
        Returns:
            List of parameter groups for optimizer
        """
        # Get backbone parameters (excluding fc)
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def get_model(backbone='resnet50', num_classes=100, pretrained=True, dropout=0.5, use_deeper_fc=False):
    """
    Factory function to create a ResNet classifier.
    
    Args:
        backbone (str): 'resnet50' or 'resnet101'
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate
        
    Returns:
        ResNetClassifier model
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


if __name__ == "__main__":
    # Test model creation
    print("=" * 50)
    print("Testing ResNet50")
    print("=" * 50)
    model_50 = get_model(backbone='resnet50', num_classes=100)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model_50(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    print("\n" + "=" * 50)
    print("Testing ResNet101")
    print("=" * 50)
    model_101 = get_model(backbone='resnet101', num_classes=100)
    
    out = model_101(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test parameter groups
    print("\n" + "=" * 50)
    print("Testing parameter groups")
    print("=" * 50)
    param_groups = model_50.get_params_for_optimizer(lr_backbone=1e-4, lr_fc=1e-3)
    for i, group in enumerate(param_groups):
        print(f"Group {i}: {len(group['params'])} params, lr={group['lr']}")
