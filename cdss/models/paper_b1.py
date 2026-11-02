"""B1 - re-implementation of the reference paper's model (Nasrin et al., J. Clin. Med. 2026).

Section 3.4 / Hyperparameter Details: custom encoder-decoder CNN with 4 encoder and 4 decoder
blocks (2x conv3x3 + BN + ReLU each), 2x2 max-pooling, 2x2 transposed-conv upsampling, skip
connections, 1x1 conv + softmax over background/disc/cup, deep supervision, Dice + CE loss.

Section 3.7-3.9: Lightweight ViT on 224x224 input with 8x8 patches, a learnable CLS token and
positional embeddings, 4-6 encoder layers (we use 6) of multi-head self-attention, LayerNorm,
residuals and a GELU MLP. The CLS feature is concatenated with the biomarker vector and passed
through fully connected layers with ReLU and a softmax output.
Unspecified by the paper (our choices): base width 32, embedding 192, 3 heads, MLP 768,
fusion hidden layer 128, dropout 0.1, single (grayscale) input channel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PaperSegCNN(nn.Module):
    def __init__(self, in_ch=1, classes=3, base=32):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]  # 32, 64, 128, 256, 512 (bottleneck)
        self.enc = nn.ModuleList([_double_conv(in_ch, ch[0])] + [_double_conv(ch[i], ch[i + 1]) for i in range(3)])
        self.bottleneck = _double_conv(ch[3], ch[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([_double_conv(ch[i] * 2, ch[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(ch[0], classes, 1)
        # Deep supervision: auxiliary 1x1 heads on the three coarser decoder outputs.
        self.aux = nn.ModuleList([nn.Conv2d(ch[i], classes, 1) for i in (3, 2, 1)])

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.bottleneck(F.max_pool2d(x, 2))
        aux = []
        for k, (up, dec, skip) in enumerate(zip(self.up, self.dec, reversed(skips))):
            x = dec(torch.cat([up(x), skip], 1))
            if k < 3:
                aux.append(self.aux[k](x))
        return {"seg_logits": self.head(x), "aux_logits": aux}


def seg_loss(out, target, num_classes=3, aux_weight=0.4):
    """Dice + CE on the main head, plus down-weighted deep-supervision terms."""
    def dice_ce(logits, tgt):
        if logits.shape[-1] != tgt.shape[-1]:
            tgt = F.interpolate(tgt[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
        prob = logits.softmax(1)
        onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
        inter = (prob * onehot).sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1))[1:].mean()
        return F.cross_entropy(logits, tgt) + dice
    loss = dice_ce(out["seg_logits"], target)
    for a in out["aux_logits"]:
        loss = loss + aux_weight * dice_ce(a, target)
    return loss


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class PaperViTFusion(nn.Module):
    def __init__(self, n_biomarkers, in_ch=1, img=224, patch=8, dim=192, depth=6, heads=3, mlp=768,
                 hidden=128, drop=0.1):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, dim, patch, stride=patch)  # linear projection of flattened patches
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim + n_biomarkers, hidden), nn.ReLU(inplace=True), nn.Dropout(drop),
                                  nn.Linear(hidden, 2))

    def forward(self, x, biomarkers):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        for blk in self.blocks:
            t = blk(t)
        cls = self.norm(t[:, 0])
        return {"logits": self.head(torch.cat([cls, biomarkers], 1))}
