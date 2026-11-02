"""Pure ViT baseline: the same pretrained DeiT backbone BT-ViT starts from, used as a plain
image classifier (16x16 linear patch embedding, all 12 blocks, CLS -> linear head).

It is trained with exactly BT-ViT's protocol (data, splits, preprocessing, augmentation,
sampler, optimiser, learning rates, epochs, early stopping, thresholds, TTA/EMA when used), so
the comparison isolates the architecture. The new classification head gets lr_new, like
BT-ViT's new layers.
"""
import timm
import torch.nn as nn


class PureViT(nn.Module):
    def __init__(self, backbone="deit_tiny_patch16_224.fb_in1k", pretrained=True, drop_path=0.1):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=2, drop_path_rate=drop_path)

    def param_groups(self, lr_new, lr_pretrained):
        head = {id(p) for p in self.net.get_classifier().parameters()}
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": [p for p in params if id(p) not in head], "lr": lr_pretrained},
                {"params": [p for p in params if id(p) in head], "lr": lr_new}]

    def forward(self, x):
        return {"logits": self.net(x)}
