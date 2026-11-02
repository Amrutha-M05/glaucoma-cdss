"""Image-only baseline classifiers (B3: CNNs, B4: plain ViTs). All return {"logits": [B, 2]}."""
import timm
import torch.nn as nn
import torchvision

BASELINES = {
    "resnet50": "B3", "densenet121": "B3",
    "deit_tiny": "B4", "vit_small": "B4",
}


class _Wrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return {"logits": self.net(x)}


def build_baseline(name):
    if name == "resnet50":
        net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif name == "densenet121":
        net = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        net.classifier = nn.Linear(net.classifier.in_features, 2)
    elif name == "deit_tiny":
        net = timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=True, num_classes=2)
    elif name == "vit_small":
        net = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=2)
    else:
        raise ValueError(name)
    return _Wrap(net)
