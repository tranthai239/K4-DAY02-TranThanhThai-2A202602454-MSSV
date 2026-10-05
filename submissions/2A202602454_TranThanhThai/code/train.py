"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chỉ số dùng để chọn checkpoint: macro-F1 val (tính qua eval.compute_metrics).
Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Đảm bảo import được các module cục bộ và eval.py từ repo root
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import dataset
import model as model_utils
import losses as loss_utils
import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | flip_vh
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py
    curves_dir: str = "curves"        # thư mục lưu ảnh biểu đồ huấn luyện
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(net: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (theo slide Day 2, trang 52)."""
    groups = model_utils.param_groups(net, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Linear Warmup rồi Cosine Annealing về gần 0 (slide trang 55)."""
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        # Cosine decay từ 1.0 về 0.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model_to_track: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model_to_track: nn.Module) -> None:
        for name, param in model_to_track.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model_target: nn.Module) -> None:
        """Nạp trọng số EMA vào model để đánh giá."""
        for name, param in model_target.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


def train_one_epoch(net: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện có hỗ trợ AMP, CutMix/Mixup và EMA."""
    net.train()
    # Nếu backbone bị freeze, bảo đảm BatchNorm giữ ở eval mode
    if getattr(net, "has_frozen_backbone", False) or cfg.init == "frozen":
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if any(not p.requires_grad for p in m.parameters()):
                    m.eval()

    total_loss = 0.0
    num_batches = len(loader)

    for batch in loader:
        images, labels, _ = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Trộn batch nếu có cấu hình Mixup / CutMix
        if cfg.mix is not None:
            mixed_images, targets = loss_utils.mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
            with autocast('cuda', enabled=cfg.amp):
                logits = net(mixed_images)
                loss = loss_utils.mixed_loss(criterion, logits, targets)
        else:
            with autocast('cuda', enabled=cfg.amp):
                logits = net(images)
                loss = criterion(logits, labels)

        optimizer.zero_grad()
        if cfg.amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(net)

        total_loss += float(loss.item())

    avg_loss = total_loss / max(1, num_batches)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(net: nn.Module, loader, criterion, device: torch.device):
    """Đánh giá model trên loader ở chế độ eval, KHÔNG tính gradient."""
    net.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []
    total_loss = 0.0

    with torch.inference_mode():
        for batch in loader:
            images, labels, filenames = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = net(images)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * len(labels)
            all_filenames.extend(filenames)
            all_y_true.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    total_samples = len(all_filenames)
    avg_loss = total_loss / max(1, total_samples)
    y_true = np.concatenate(all_y_true, axis=0) if all_y_true else np.array([], dtype=int)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, ev.NUM_CLASSES))

    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong huấn luyện chuẩn: Train Loss, Val Loss và Val Macro-F1 theo epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_macro_f1s = [h["val_macro_f1"] for h in history]
    val_top1s = [h["val_top1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Đồ thị 1: Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", color="royalblue", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="crimson", marker="s")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Đồ thị 2: Metrics (Macro-F1 & Top-1)
    axes[1].plot(epochs, val_macro_f1s, label="Val Macro-F1", color="forestgreen", marker="^", linewidth=2)
    axes[1].plot(epochs, val_top1s, label="Val Top-1 Acc", color="darkorange", linestyle="--", marker=".")
    axes[1].set_title(f"{title} - Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(z)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    return probs


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu toàn bộ kết quả, checkpoint, biểu đồ."""
    start_time = time.time()
    set_seed(cfg.seed)

    save_dir = run_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    # 1. Lưu cấu hình config.json
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Đọc và kiểm tra split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, images_dir=cfg.images_dir)

    # 3. Tạo transforms và loaders
    train_transform = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    eval_transform = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_transform,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, eval_transform,
        batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
    )

    # 4. Khởi tạo mô hình
    net = model_utils.build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    )
    net.to(device)

    num_params_m = model_utils.count_params(net)
    gmacs = model_utils.count_gmacs(net, img_size=cfg.img_size)

    # 5. Hàm loss
    loss_kwargs = {
        "smoothing": cfg.label_smoothing,
        "gamma": cfg.focal_gamma,
    }
    if cfg.loss in ("ce_weighted", "weighted") or cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().sort_index().to_numpy()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        w = loss_utils.class_weights(counts, beta=beta).to(device)
        loss_kwargs["weight"] = w

    criterion = loss_utils.build_criterion(cfg.loss, **loss_kwargs)
    val_criterion = nn.CrossEntropyLoss()

    # 6. Optimizer, Scheduler, Scaler, EMA
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler('cuda', enabled=cfg.amp)
    ema = EMA(net, decay=cfg.ema_decay) if cfg.ema_decay is not None else None
    # Tránh cảnh báo "lr_scheduler.step() before optimizer.step()" ở lần gọi đầu tiên
    optimizer._step_count = 1

    # 7. Vòng lặp epoch
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    epoch_durations = []

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        epoch_dur = time.time() - t0
        epoch_durations.append(epoch_dur)

        # Đánh giá trên Validation
        eval_net = net
        if ema is not None:
            eval_net = copy.deepcopy(net)
            ema.apply_shadow(eval_net)

        val_files, val_y, val_logits, val_loss = evaluate(eval_net, val_loader, val_criterion, device)
        val_probs = _softmax(val_logits)
        val_preds = val_probs.argmax(axis=-1)
        metrics = ev.compute_metrics(val_y, val_preds, val_probs)

        macro_f1 = float(metrics["macro_f1"])
        top1 = float(metrics["top1"])

        record = {
            "epoch": ep,
            "train_loss": round(train_res["train_loss"], 5),
            "val_loss": round(val_loss, 5),
            "val_macro_f1": round(macro_f1, 5),
            "val_top1": round(top1, 5),
            "lr": train_res["lr"],
            "time_sec": round(epoch_dur, 2),
        }
        history.append(record)

        # Lưu checkpoint tốt nhất theo Macro-F1 Val (hòa thì lấy epoch sớm hơn)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = ep
            best_state_dict = copy.deepcopy(eval_net.state_dict())

    # Ép FP32 trước khi lưu checkpoint (tránh lỗi dtype khi suy luận không dùng autocast)
    best_state_dict_fp32 = {k: v.float() for k, v in best_state_dict.items()}
    torch.save(best_state_dict_fp32, save_dir / "best_model.pth")

    # 8. Nạp lại checkpoint tốt nhất và lưu dự đoán trên tập Val
    net.load_state_dict(best_state_dict)
    val_files, val_y, best_val_logits, _ = evaluate(net, val_loader, val_criterion, device)
    best_val_probs = _softmax(best_val_logits)
    ev.save_predictions(pred_path(cfg, "val"), val_files, val_y, best_val_probs)

    # 9. Chỉ khi save_test_predictions = True (Bước 4): chạy test đúng MỘT lần duy nhất
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, eval_transform,
            batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
        )
        test_files, test_y, test_logits, _ = evaluate(net, test_loader, val_criterion, device)
        test_probs = _softmax(test_logits)
        ev.save_predictions(pred_path(cfg, "test"), test_files, test_y, test_probs)
        test_preds = test_probs.argmax(axis=-1)
        test_metrics = ev.compute_metrics(test_y, test_preds, test_probs)

    # 10. Lưu history và vẽ đường cong
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / "history.csv", index=False)

    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} ({cfg.backbone})")

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "val_top1": history[best_epoch - 1]["val_top1"] if best_epoch > 0 else 0.0,
        "params_m": num_params_m,
        "gmacs": gmacs,
        "mean_epoch_time_s": round(float(np.mean(epoch_durations)), 2),
        "train_sec_per_epoch": round(float(np.mean(epoch_durations)), 2),
        "total_time_s": round(time.time() - start_time, 2),
    }
    if test_metrics is not None:
        summary["test_macro_f1"] = float(test_metrics["macro_f1"])
        summary["test_top1"] = float(test_metrics["top1"])
        summary["test_ece"] = float(test_metrics["ece"])

    # Dọn sạch VRAM và RAM để tránh tràn bộ nhớ (OOM) làm ngắt kết nối Colab
    del net, eval_net, best_state_dict, optimizer, scheduler, scaler
    del train_loader, val_loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến các cặp KEY=VALUE thành dict, ép kiểu theo fields của Config."""
    field_types = {f.name: f.type for f in fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ: {pair}. Cần định dạng KEY=VALUE")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in field_types:
            raise KeyError(f"Trường '{key}' không tồn tại trong Config. Các trường: {list(field_types.keys())}")

        target_type = field_types[key]
        if val.lower() == "none":
            overrides[key] = None
        elif "bool" in str(target_type).lower():
            overrides[key] = val.lower() in ("true", "1", "yes")
        elif "int" in str(target_type).lower():
            overrides[key] = int(val)
        elif "float" in str(target_type).lower():
            overrides[key] = float(val)
        else:
            overrides[key] = val

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: py train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện một thí nghiệm DeepWeeds.")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè cấu hình, ví dụ: --set exp_id=B01 seed=1")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"=== Bắt đầu thí nghiệm {cfg.exp_id} (Backbone: {cfg.backbone}, Seed: {cfg.seed}) ===")
    res = run(cfg)
    print(f"=== Hoàn thành: {res} ===")


if __name__ == "__main__":
    main()
