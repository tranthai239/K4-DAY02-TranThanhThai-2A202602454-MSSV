"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2:
  - label smoothing (trang 56)
  - focal loss (trang 57)
  - Mixup / CutMix (trang 48)

Giao diện chuẩn:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: 'ce', 'ls' (label smoothing), 'focal', 'ce_weighted'.

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    kind = kind.lower()
    if kind == "ce":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    elif kind in ("ls", "label_smoothing"):
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        weight = kw.get("weight", None)
        return LabelSmoothingCE(smoothing=smoothing, weight=weight)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", kw.get("weight", None))
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind in ("ce_weighted", "weighted"):
        weight = kw.get("weight", None)
        if weight is None:
            raise ValueError("ce_weighted yêu cầu tham số 'weight'")
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loss kind: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Khi eps = 0, loss trở về đúng CrossEntropy chuẩn.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target, weight=self.weight, reduction=self.reduction)

        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # NLL loss
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        # Smooth loss: trung bình log_probs qua tất cả các lớp
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.weight is not None:
            weight_factor = self.weight[target]
            loss = loss * weight_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    - gamma: hệ số tập trung (gamma = 0 tương đương standard cross-entropy).
    - alpha: vector trọng số theo lớp (None hoặc tensor shape [num_classes]).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Lấy p_t và log(p_t) cho nhãn đúng
        log_pt = log_p.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        pt = p.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        # Trọng số focal (1 - p_t)^gamma
        if self.gamma > 0.0:
            focal_weight = torch.pow(1.0 - pt, self.gamma)
        else:
            focal_weight = torch.ones_like(pt)

        loss = -focal_weight * log_pt

        # Trọng số alpha
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (chỉ dùng train, không dùng val hay test).

    - beta = 0.0: trọng số nghịch đảo tần suất (1 / n_c), chuẩn hoá về trung bình bằng 1
    - beta > 0.0: class-balanced theo số mẫu hiệu dụng (Cui et al., CVPR 2019):
                  w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng bằng số lớp (mean = 1)
    """
    counts = np.asarray(counts, dtype=np.float64)
    num_classes = len(counts)

    if beta <= 0.0:
        # Nghịch đảo tần suất
        weights = 1.0 / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
    else:
        # Số mẫu hiệu dụng: (1 - beta) / (1 - beta^n)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode='mixup': x_mix = lam * x + (1 - lam) * x[perm]
    - mode='cutmix': cắt một hộp chữ nhật từ x[perm] dán vào x, sau đó điều chỉnh lam
      theo DIỆN TÍCH THỰC của hộp sau khi bị kẹp trong biên ảnh.
    - trả về (x_mix, (y_a, y_b, lam))
    """
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        _, _, h, w = x.shape
        # Tính kích thước hộp chữ nhật
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Tâm ngẫu nhiên của hộp
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))

        # Tọa độ hộp bị kẹp trong biên
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lại lambda theo diện tích thực tế
        actual_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = h * w
        lam = 1.0 - (float(actual_area) / float(total_area))

        return x_mixed, (y_a, y_b, lam)

    else:
        raise ValueError(f"Không hỗ trợ mix mode: {mode}")


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
