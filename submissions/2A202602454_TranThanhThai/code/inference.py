"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2:
  - TTA (trang 62-66, 75)
  - ensemble / EMA / model soup (trang 67)
  - độ phân giải kiểm tra (trang 68)
  - temperature scaling (trang 69)
  - gộp BatchNorm vào Conv (trang 71)

Mọi hàm chạy ở chế độ eval, không gradient.
Nhiệt độ T chỉ được khớp trên VAL rồi áp dụng sang test.

Giao diện chuẩn:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def predict_logits(model: nn.Module, loader, device: str | torch.device, view=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    device = torch.device(device)
    model.to(device)

    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            images, labels, filenames = batch
            images = images.to(device, non_blocking=True)

            if view is not None:
                images = view(images)

            logits = model(images)

            all_filenames.extend(filenames)
            all_y_true.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0) if all_logits else np.array([], dtype=int)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, NUM_CLASSES))

    return all_filenames, y_true, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """View gốc: giữ nguyên batch ảnh."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    """5-crop: 4 góc + giữa với kích thước crop x crop."""
    _, _, h, w = x.shape
    if crop > h or crop > w:
        raise ValueError(f"Crop size {crop} lớn hơn kích thước ảnh {(h, w)}")

    # 4 góc và vị trí giữa
    tl = x[:, :, 0:crop, 0:crop]
    tr = x[:, :, 0:crop, w - crop:w]
    bl = x[:, :, h - crop:h, 0:crop]
    br = x[:, :, h - crop:h, w - crop:w]
    ch = (h - crop) // 2
    cw = (w - crop) // 2
    cc = x[:, :, ch:ch + crop, cw:cw + crop]

    return [tl, tr, bl, br, cc]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    out = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        out.append(resized)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax ổn định số học cho mảng 2D (N, C)."""
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán xác suất (N, 9) (slide trang 62).

    - space='prob': trung bình softmax của từng view
    - space='logit': trung bình logit rồi tính softmax
    """
    if not logits_per_view:
        raise ValueError("logits_per_view rỗng")

    if space == "prob":
        probs_list = [_softmax(np.asarray(log)) for log in logits_per_view]
        mean_probs = np.mean(probs_list, axis=0)
        # Chuẩn hoá đảm bảo tổng bằng 1
        return mean_probs / np.sum(mean_probs, axis=-1, keepdims=True)

    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        return _softmax(mean_logits)

    else:
        raise ValueError(f"Không hỗ trợ aggregation space: {space}")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác kiến trúc hoặc khác seed).

    Tất cả các mảng trong list_of_probs phải cùng hình dạng (N, 9) và cùng thứ tự file.
    """
    if not list_of_probs:
        raise ValueError("list_of_probs rỗng")

    arrs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    mean_probs = np.mean(arrs, axis=0)
    # Chuẩn hoá tổng hàng = 1
    return mean_probs / np.sum(mean_probs, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 tối thiểu hoá Cross-Entropy (NLL) trên tập VAL:

        p = softmax(logit / T)  (slide trang 69).

    Lưu ý: T được khớp hoàn toàn trên VAL, không dùng tập test.
    """
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)

    def nll_obj(t_val: float) -> float:
        scaled = logits / max(t_val, 1e-4)
        probs = _softmax(scaled)
        correct_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.mean(np.log(correct_probs)))

    # Tìm kiếm T tối ưu trong khoảng [0.05, 10.0]
    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
    optimal_t = float(res.x)
    return max(optimal_t, 0.05)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về xác suất softmax(logits / T)."""
    T = max(float(T), 1e-4)
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước để tăng tốc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)
        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Với các kiến trúc dùng LayerNorm (ViT, Swin, ConvNeXt), hàm trả về bản sao model.
    """
    model_fused = copy.deepcopy(model)
    model_fused.eval()

    # Dùng torch.nn.utils.fusion nếu khả dụng
    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
    except ImportError:
        fuse_conv_bn_eval = None

    def _fuse_modules(module: nn.Module):
        children = list(module.named_children())
        for i in range(len(children) - 1):
            name1, m1 = children[i]
            name2, m2 = children[i + 1]

            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                if fuse_conv_bn_eval is not None:
                    fused_conv = fuse_conv_bn_eval(m1, m2)
                else:
                    # Tự tính trọng số gộp theo công thức
                    w = m1.weight.clone()
                    b = m1.bias.clone() if m1.bias is not None else torch.zeros(m1.out_channels, device=w.device)

                    gamma = m2.weight
                    beta = m2.bias
                    mean = m2.running_mean
                    var = m2.running_var
                    eps = m2.eps

                    invstd = gamma / torch.sqrt(var + eps)
                    w_fused = w * invstd.reshape(-1, 1, 1, 1)
                    b_fused = beta + (b - mean) * invstd

                    fused_conv = nn.Conv2d(
                        in_channels=m1.in_channels,
                        out_channels=m1.out_channels,
                        kernel_size=m1.kernel_size,
                        stride=m1.stride,
                        padding=m1.padding,
                        dilation=m1.dilation,
                        groups=m1.groups,
                        bias=True,
                    )
                    fused_conv.weight.data.copy_(w_fused)
                    fused_conv.bias.data.copy_(b_fused)

                setattr(module, name1, fused_conv)
                setattr(module, name2, nn.Identity())

        for _, child in module.named_children():
            _fuse_modules(child)

    _fuse_modules(model_fused)
    return model_fused
