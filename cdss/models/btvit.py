"""BT-ViT: Biomarker-Token Vision Transformer (METHODOLOGY.md section 4).

ROI 224x224 RGB
 -> conv stem (3 stride-2 conv blocks + 2x2 projection) -> 14x14 = 196 patch tokens, D=192
 -> [CLS] + patches + positional embedding -> transformer blocks 1-4
 -> segmentation decoder (tokens + stem skips) -> background / disc / cup
 -> differentiable biomarkers from the soft masks, in 4 groups -> 4 clinical tokens
 -> blocks 5-8 over [CLS | patches | clinical tokens], with a rim-guided attention bias on the
    CLS query (CLS->patch logits += alpha * log(rim prior))
 -> head on concat(CLS, mean clinical token) -> 2 logits.

Transformer blocks, CLS token and positional embedding are initialised from ImageNet DeiT-Tiny
(its first 8 of 12 blocks; its 14x14 position grid matches the stem). Everything else is new.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = {
    "cup_disc": ["cdr_area", "vcdr", "hcdr"],
    "rim": ["rim_inferior", "rim_superior", "rim_nasal_temporal", "nrr", "isnt_violation"],
    "size": ["disc_area", "disc_ellipticity"],
    "edge": ["disc_edge", "cup_edge"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
EPS = 1e-6


def _conv_bn(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


def _quadrant_masks(size, device):
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
    dy, dx = (y - (size - 1) / 2) / size, (x - (size - 1) / 2) / size
    superior, inferior = (dy < -dx.abs()).float(), (dy > dx.abs()).float()
    return inferior, superior, 1 - superior - inferior


def soft_biomarkers(disc, cup, gray):
    """Differentiable biomarkers from disc/cup probability maps [B, H, W] (cup inside disc).

    The same function applied to one-hot expert masks gives the supervision targets, so
    predicted and target biomarkers share one definition. Extents use row/column maxima.
    """
    b, h, w = disc.shape
    da, ca = disc.sum((1, 2)), cup.sum((1, 2))
    dv, cv = disc.amax(2).sum(1), cup.amax(2).sum(1)  # vertical extent (rows touched)
    dh, ch = disc.amax(1).sum(1), cup.amax(1).sum(1)  # horizontal extent
    rim = (disc - cup).clamp(min=0)
    inf_m, sup_m, nt_m = _quadrant_masks(h, disc.device)
    ri, rs, rnt = (rim * inf_m).sum((1, 2)), (rim * sup_m).sum((1, 2)), (rim * nt_m).sum((1, 2))
    rt = ri + rs + rnt + EPS
    fi, fs, fnt = ri / rt, rs / rt, rnt / rt
    gx = F.conv2d(gray[:, None], gray.new_tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]), padding=1)
    gy = F.conv2d(gray[:, None], gray.new_tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]), padding=1)
    grad = (gx ** 2 + gy ** 2 + EPS).sqrt()[:, 0]

    def boundary_strength(p):
        px = F.pad(p[:, None], (1, 1, 1, 1), mode="replicate")
        edge = ((px[..., 1:-1, 2:] - px[..., 1:-1, :-2]) ** 2 + (px[..., 2:, 1:-1] - px[..., :-2, 1:-1]) ** 2 + EPS).sqrt()[:, 0]
        return (edge * grad).sum((1, 2)) / (edge.sum((1, 2)) + EPS)

    feats = {
        "cdr_area": ca / (da + EPS), "vcdr": cv / (dv + EPS), "hcdr": ch / (dh + EPS),
        "rim_inferior": fi, "rim_superior": fs, "rim_nasal_temporal": fnt,
        "nrr": (fi + fs + 0.01) / (fnt + 0.01),
        "isnt_violation": F.relu(fs - fi) + F.relu(fnt / 2 - fs) + F.relu(fnt / 2 - fi),
        "disc_area": da / (h * w), "disc_ellipticity": dv / (dh + EPS),
        "disc_edge": boundary_strength(disc), "cup_edge": boundary_strength(cup),
    }
    return torch.stack([feats[f] for f in FEATURES], 1)


class _BiasedBlock(nn.Module):
    """A timm ViT block whose attention accepts an additive bias [B, 1, N, N]."""

    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, x, bias=None):
        blk, attn = self.blk, self.blk.attn
        h = blk.norm1(x)
        b, n, c = h.shape
        qkv = attn.qkv(h).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=attn.attn_drop.p if self.training else 0.0)
        a = attn.proj_drop(attn.proj(a.transpose(1, 2).reshape(b, n, c)))
        x = x + blk.drop_path1(blk.ls1(a))
        return x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))


class BTViT(nn.Module):
    def __init__(self, pretrained=True, depth_pre=4, depth_post=4, drop_path=0.1, head_drop=0.2,
                 clinical_tokens=True, rim_bias=True, backbone="deit_tiny_patch16_224.fb_in1k"):
        super().__init__()
        self.clinical_tokens, self.rim_bias = clinical_tokens, rim_bias
        deit = timm.create_model(backbone, pretrained=pretrained, drop_path_rate=drop_path)
        dim = deit.embed_dim  # 192 (DeiT-Tiny) or 384 (DeiT-Small); both use a 14x14 patch grid at 224 px
        # Conv stem: 224 -> 112 -> 56 -> 28 -> (2x2 proj) 14
        self.stem = nn.ModuleList([_conv_bn(3, 32, 2), _conv_bn(32, 64, 2), _conv_bn(64, 128, 2)])
        self.proj = nn.Conv2d(128, dim, 2, 2)
        self.cls_token, self.pos_embed = deit.cls_token, deit.pos_embed  # [1, 197, 192]: CLS + 14x14
        blocks = list(deit.blocks)[:depth_pre + depth_post]
        self.pre = nn.ModuleList(blocks[:depth_pre])
        self.post = nn.ModuleList([_BiasedBlock(b) for b in blocks[depth_pre:]])
        self.norm = deit.norm
        # Segmentation decoder: 14 -> 28 (+stem128) -> 56 (+stem64) -> 112 (+stem32) -> 224
        self.dec = nn.ModuleList([_conv_bn(dim + 128, 128, 1), _conv_bn(128 + 64, 64, 1), _conv_bn(64 + 32, 32, 1)])
        self.seg_head = nn.Conv2d(32, 3, 1)
        # Clinical tokens: one MLP per biomarker group + a type embedding.
        self.group_slices, start = [], 0
        for g in GROUPS.values():
            self.group_slices.append(slice(start, start + len(g)))
            start += len(g)
        self.token_mlps = nn.ModuleList([nn.Sequential(nn.Linear(len(g), dim), nn.GELU(), nn.Linear(dim, dim))
                                         for g in GROUPS.values()])
        self.type_embed = nn.Parameter(torch.zeros(1, len(GROUPS), dim))
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        self.alpha = nn.Parameter(torch.full((len(self.post),), 0.5))
        self.head = nn.Sequential(nn.Linear(2 * dim if clinical_tokens else dim, dim), nn.GELU(),
                                  nn.Dropout(head_drop), nn.Linear(dim, 2))
        # Biomarker normalisation (set from training-set expert masks before training).
        self.register_buffer("bio_mean", torch.zeros(len(FEATURES)))
        self.register_buffer("bio_std", torch.ones(len(FEATURES)))

    def param_groups(self, lr_new, lr_pretrained):
        pretrained = {id(p) for m in (self.pre, self.post, self.norm) for p in m.parameters()}
        pretrained |= {id(self.cls_token), id(self.pos_embed)}
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": [p for p in params if id(p) in pretrained], "lr": lr_pretrained},
                {"params": [p for p in params if id(p) not in pretrained], "lr": lr_new}]

    @staticmethod
    def gray(x):
        return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]

    def normalise(self, feats):
        return (feats - self.bio_mean) / self.bio_std

    def forward(self, x):
        skips, h = [], x
        for block in self.stem:
            h = block(h)
            skips.append(h)
        t = self.proj(h).flatten(2).transpose(1, 2)  # [B, 196, D]
        t = torch.cat([self.cls_token.expand(len(t), -1, -1), t], 1) + self.pos_embed
        for blk in self.pre:
            t = blk(t)

        out = {}
        # Segmentation branch from the patch tokens of block 4.
        g = t[:, 1:].transpose(1, 2).reshape(len(t), -1, 14, 14)
        for dec, skip in zip(self.dec, reversed(skips)):
            g = dec(torch.cat([F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        seg_logits = F.interpolate(self.seg_head(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        out["seg_logits"] = seg_logits
        p = seg_logits.softmax(1)
        disc, cup = p[:, 1] + p[:, 2], p[:, 2]
        bio = soft_biomarkers(disc, cup, self.gray(x))
        out["biomarkers"] = bio

        bias = None
        if self.clinical_tokens:
            z = self.normalise(bio)
            clin = torch.stack([mlp(z[:, s]) for mlp, s in zip(self.token_mlps, self.group_slices)], 1) + self.type_embed
            t = torch.cat([t, clin], 1)
        if self.rim_bias:
            rim = F.adaptive_avg_pool2d((disc - cup).clamp(min=0)[:, None].detach(), 14).flatten(1)  # [B, 196]
            rim = rim / (rim.amax(1, keepdim=True) + EPS)
            n = t.shape[1]
            bias = t.new_zeros(len(t), 1, n, n)
            bias[:, 0, 0, 1:197] = torch.log(rim + 0.05)
        for i, blk in enumerate(self.post):
            t = blk(t, None if bias is None else bias * self.alpha[i])
        t = self.norm(t)
        feat = t[:, 0] if not self.clinical_tokens else torch.cat([t[:, 0], t[:, 197:].mean(1)], 1)
        out["logits"] = self.head(feat)
        return out


class BTViTv2(BTViT):
    """BT-ViT v2: a strict superset of the pretrained DeiT classifier.

    Classification path = DeiT's own pretrained 16x16 patch embedding and all 12 blocks, exactly
    as in the pure-ViT baseline. Additions:
      - segmentation decoder on the patch tokens after block `seg_from` (default 4), with skips
        from a small conv branch that only feeds the decoder;
      - 4 clinical tokens appended before the last `depth_post` blocks (default 9-12) and a
        rim-guided bias on the CLS query in those blocks;
      - head: linear(CLS) + gate * linear(mean clinical token).
    Initialised to behave like the pure ViT: alpha = 0, zero-initialised token-MLP outputs and
    clinical gate = 0, so training only departs from the baseline where the additions help.
    """

    def __init__(self, pretrained=True, seg_from=4, depth_post=4, drop_path=0.1, head_drop=0.0,
                 clinical_tokens=True, rim_bias=True, backbone="deit_tiny_patch16_224.fb_in1k",
                 beta_init=-4.0, gate_weight_decay=True):
        nn.Module.__init__(self)
        self.gate_weight_decay = gate_weight_decay
        self.clinical_tokens, self.rim_bias, self.seg_from = clinical_tokens, rim_bias, seg_from
        deit = timm.create_model(backbone, pretrained=pretrained, drop_path_rate=drop_path)
        dim = deit.embed_dim
        self.patch_embed, self.cls_token, self.pos_embed = deit.patch_embed, deit.cls_token, deit.pos_embed
        blocks = list(deit.blocks)
        self.pre = nn.ModuleList(blocks[:len(blocks) - depth_post])
        self.post = nn.ModuleList([_BiasedBlock(b) for b in blocks[len(blocks) - depth_post:]])
        self.norm = deit.norm
        # Conv branch used only by the segmentation decoder (skips at 112 / 56 / 28 px).
        self.stem = nn.ModuleList([_conv_bn(3, 32, 2), _conv_bn(32, 64, 2), _conv_bn(64, 128, 2)])
        self.dec = nn.ModuleList([_conv_bn(dim + 128, 128, 1), _conv_bn(128 + 64, 64, 1), _conv_bn(64 + 32, 32, 1)])
        self.seg_head = nn.Conv2d(32, 3, 1)
        self.group_slices, start = [], 0
        for g in GROUPS.values():
            self.group_slices.append(slice(start, start + len(g)))
            start += len(g)
        self.token_mlps = nn.ModuleList([nn.Sequential(nn.Linear(len(g), dim), nn.GELU(), nn.Linear(dim, dim))
                                         for g in GROUPS.values()])
        for mlp in self.token_mlps:
            nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
        self.type_embed = nn.Parameter(torch.zeros(1, len(GROUPS), dim))
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        self.alpha = nn.Parameter(torch.zeros(len(self.post)))
        # Learnable key bias on the clinical tokens (per block): starts nearly closed (e^-4) so the
        # initial model behaves like the pure ViT; opens only if the tokens help.
        self.beta = nn.Parameter(torch.full((len(self.post),), float(beta_init)))
        self.head_drop = nn.Dropout(head_drop)
        self.head_cls = nn.Linear(dim, 2)
        self.head_clin = nn.Linear(dim, 2)
        self.clin_gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("bio_mean", torch.zeros(len(FEATURES)))
        self.register_buffer("bio_std", torch.ones(len(FEATURES)))

    def param_groups(self, lr_new, lr_pretrained):
        pretrained = {id(p) for m in (self.patch_embed, self.pre, self.post, self.norm) for p in m.parameters()}
        pretrained |= {id(self.cls_token), id(self.pos_embed)}
        gates = {id(self.alpha), id(self.beta), id(self.clin_gate)} if not self.gate_weight_decay else set()
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [{"params": [p for p in params if id(p) in pretrained], "lr": lr_pretrained},
                  {"params": [p for p in params if id(p) not in pretrained and id(p) not in gates], "lr": lr_new}]
        if gates:  # gate scalars free of weight decay, so they are not pulled shut
            groups.append({"params": [p for p in params if id(p) in gates], "lr": lr_new, "weight_decay": 0.0})
        return groups

    def forward(self, x):
        t = self.patch_embed(x)  # [B, 196, D]
        t = torch.cat([self.cls_token.expand(len(t), -1, -1), t], 1) + self.pos_embed
        seg_tokens = None
        for i, blk in enumerate(self.pre):
            t = blk(t)
            if i + 1 == self.seg_from:
                seg_tokens = t[:, 1:197]
        out = {}
        skips, h = [], x
        for block in self.stem:
            h = block(h)
            skips.append(h)
        g = seg_tokens.transpose(1, 2).reshape(len(t), -1, 14, 14)
        for dec, skip in zip(self.dec, reversed(skips)):
            g = dec(torch.cat([F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        seg_logits = F.interpolate(self.seg_head(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        out["seg_logits"] = seg_logits
        p = seg_logits.softmax(1)
        disc, cup = p[:, 1] + p[:, 2], p[:, 2]
        bio = soft_biomarkers(disc, cup, self.gray(x))
        out["biomarkers"] = bio

        if self.clinical_tokens:
            z = self.normalise(bio)
            clin = torch.stack([mlp(z[:, s]) for mlp, s in zip(self.token_mlps, self.group_slices)], 1) + self.type_embed
            t = torch.cat([t, clin], 1)
        n = t.shape[1]
        rim_bias = clin_keys = None
        if self.rim_bias:
            rim = F.adaptive_avg_pool2d((disc - cup).clamp(min=0)[:, None].detach(), 14).flatten(1)
            rim = rim / (rim.amax(1, keepdim=True) + EPS)
            rim_bias = t.new_zeros(len(t), 1, n, n)
            rim_bias[:, 0, 0, 1:197] = torch.log(rim + 0.05)
        if self.clinical_tokens:
            clin_keys = t.new_zeros(1, 1, n, n)
            clin_keys[..., 197:] = 1.0
        for i, blk in enumerate(self.post):
            bias = None
            if rim_bias is not None:
                bias = rim_bias * self.alpha[i]
            if clin_keys is not None:
                bias = clin_keys * self.beta[i] if bias is None else bias + clin_keys * self.beta[i]
            t = blk(t, bias)
        t = self.norm(t)
        logits = self.head_cls(self.head_drop(t[:, 0]))
        if self.clinical_tokens:
            logits = logits + self.clin_gate * self.head_clin(self.head_drop(t[:, 197:].mean(1)))
        out["logits"] = logits
        return out


def btvit_aux_loss(model, lambda_seg=1.0, lambda_bio=0.5):
    """Segmentation (Dice + CE over 3 classes) + biomarker supervision against expert masks."""
    def loss(out, batch, device):
        mask = batch["mask"].to(device)
        logits = out["seg_logits"]
        prob = logits.softmax(1)
        onehot = F.one_hot(mask, 3).permute(0, 3, 1, 2).float()
        inter = (prob * onehot).sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1))[1:].mean()
        l_seg = F.cross_entropy(logits, mask) + dice
        x = batch["image"].to(device)
        with torch.no_grad():
            target = soft_biomarkers(onehot[:, 1] + onehot[:, 2], onehot[:, 2], BTViT.gray(x))
        l_bio = F.smooth_l1_loss(model.normalise(out["biomarkers"]), model.normalise(target))
        return lambda_seg * l_seg + lambda_bio * l_bio
    return loss
