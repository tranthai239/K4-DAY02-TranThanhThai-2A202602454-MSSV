"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau mỗi lần đo
  - >= 50 lần đo, báo cáo p50, p95, p99
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, gộp BN, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], None], warmup: int = 10, iters: int = 100, sync: Callable[[], None] | None = None) -> dict:
    """Đo thời gian thực thi hàm `fn()`, trả về kết quả bằng mili-giây (ms).

    `sync` là hàm đồng bộ (torch.cuda.synchronize hoặc None).
    """
    if sync is None:
        if torch.cuda.is_available():
            sync = torch.cuda.synchronize
        else:
            sync = lambda: None  # noqa: E731

    # 1. Warmup (bỏ qua cuBLAS init, shader compile)
    for _ in range(warmup):
        fn()
    sync()

    # 2. Đo thời gian lặp
    timings = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        timings[i] = (t1 - t0) * 1000.0  # chuyển về ms

    p50 = float(np.percentile(timings, 50))
    p95 = float(np.percentile(timings, 95))
    p99 = float(np.percentile(timings, 99))
    mean = float(np.mean(timings))
    std = float(np.std(timings, ddof=1)) if iters > 1 else 0.0

    return {
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi trực tiếp vào sheet `Latency` của results.xlsx.
    """
    model.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(dev)

    dtype = dtype.lower()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
    else:
        model = model.float()
        dummy_input = dummy_input.float()

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else lambda: None

    @torch.inference_mode()
    def forward_fn():
        if dtype == "amp" and dev.type == "cuda":
            with torch.cuda.amp.autocast():
                _ = model(dummy_input)
        else:
            _ = model(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    # Tính thông lượng (images/sec)
    p50_sec = res["p50"] / 1000.0
    images_per_s = round(batch_size / p50_sec, 2) if p50_sec > 0 else 0.0

    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ thực tế khi chạy TTA với k_views so với 1-view mốc (slide trang 63)."""
    base_res = latency_report(model, batch_size, img_size, dtype, device, warmup, iters)

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else lambda: None
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    @torch.inference_mode()
    def tta_fn():
        for _ in range(k_views):
            _ = model(dummy_input)

    tta_res = bench(tta_fn, warmup=warmup, iters=iters, sync=sync_fn)
    ratio = round(tta_res["p50"] / max(base_res["p50"], 1e-4), 2)

    return {
        "k_views": k_views,
        "single_view_p50": base_res["p50"],
        "tta_p50": tta_res["p50"],
        "tta_p95": tta_res["p95"],
        "tta_p99": tta_res["p99"],
        "relative_cost": ratio,
    }
