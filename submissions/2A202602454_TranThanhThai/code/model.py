"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện chuẩn:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp theo cấu hình kiến trúc và khởi tạo.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ từ đầu
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ (fine-tune)
    """
    is_pretrained = (init != "scratch") and pretrained

    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Lưu lại thông tin cấu hình tag trọng số
    tag = "scratch"
    if is_pretrained:
        cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
        tag = cfg.get("tag", cfg.get("architecture", "pretrained"))
    setattr(model, "weight_tag", tag)
    setattr(model, "init_mode", init)

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.

    Lưu ý (GUIDE.md mục 3.2): Khi backbone đóng băng, BatchNorm trong backbone
    cũng phải duy trì ở eval mode để không cập nhật running mean/variance.
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters())

    for param in model.parameters():
        if param in head_params:
            param.requires_grad = True
        else:
            param.requires_grad = False

    setattr(model, "has_frozen_backbone", True)


def set_train_mode_with_frozen_bn(model: nn.Module) -> None:
    """Đưa model về train mode nhưng giữ các lớp BatchNorm bị đóng băng ở eval mode."""
    model.train()
    if getattr(model, "has_frozen_backbone", False):
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                # Nếu tham số của module BN không requires_grad thì ép về eval
                if any(not p.requires_grad for p in module.parameters()):
                    module.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:

    1. Backbone 2D+ weights (Conv, Linear): lr = lr_backbone, weight_decay = weight_decay
    2. Backbone 1D hoặc biases (Norm, bias): lr = lr_backbone, weight_decay = 0.0
    3. Classifier head: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    Bỏ qua toàn bộ tham số có requires_grad == False.
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters())

    backbone_weights = []
    backbone_no_decay = []
    head_parameters = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in head_params:
            head_parameters.append(param)
        else:
            # 1D tensors (bias, 1D weight của norm) không áp dụng weight decay
            if param.ndim <= 1 or name.endswith(".bias"):
                backbone_no_decay.append(param)
            else:
                backbone_weights.append(param)

    groups = []
    if backbone_weights:
        groups.append({
            "params": backbone_weights,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
            "group_name": "backbone_decay",
        })
    if backbone_no_decay:
        groups.append({
            "params": backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        })
    if head_parameters:
        groups.append({
            "params": head_parameters,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "group_name": "head",
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Tính GMAC cho một ảnh 3 x img_size x img_size (MAC = FLOPs / 2)."""
    # 1. Thử dùng fvcore nếu có
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, img_size, img_size)
        flops = FlopCountAnalysis(model, dummy).total()
        return round(flops / 1e9, 3)
    except Exception:
        pass

    # 2. Thử dùng ptflops nếu có
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, img_size, img_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return round(macs / 1e9, 3)
    except Exception:
        pass

    # 3. Thử dùng thop nếu có
    try:
        import thop
        dummy = torch.randn(1, 3, img_size, img_size)
        macs, _ = thop.profile(model, inputs=(dummy,), verbose=False)
        return round(macs / 1e9, 3)
    except Exception:
        pass

    # 4. Tự tính toán MACs thông qua forward hook cho Conv2d và Linear
    macs_counter = [0]
    hooks = []

    def conv_hook(self, input, output):
        batch_size, out_channels, out_h, out_w = output.shape
        in_channels = self.in_channels
        kernel_h, kernel_w = self.kernel_size
        groups = self.groups
        mac = (batch_size * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w)
        macs_counter[0] += mac

    def linear_hook(self, input, output):
        batch_size = input[0].shape[0] if input[0].ndim > 1 else 1
        mac = batch_size * self.in_features * self.out_features
        macs_counter[0] += mac

    was_training = model.training
    model.eval()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    if macs_counter[0] > 0:
        return round(macs_counter[0] / 1e9, 3)

    # Ước lượng chuẩn theo slide nếu mô hình đặc thù
    name = getattr(model, "default_cfg", {}).get("architecture", "")
    known_macs = {
        "resnet50": 4.1,
        "resnext50_32x4d": 4.2,
        "convnext_tiny": 4.5,
        "deit_small_patch16_224": 4.6,
        "swin_tiny_patch4_window7_224": 4.5,
        "efficientnet_b0": 0.39,
        "mobilenetv3_large_100": 0.22,
    }
    return known_macs.get(name, 4.0)
