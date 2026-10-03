"""test_code.py - Kiểm tra toàn diện bộ mã nguồn trong submissions/2A202602454_TranThanhThai/code/.

Chạy lệnh:
    py -3.12 -m unittest tests/test_code.py -v
"""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "submissions" / "2A202602454_TranThanhThai" / "code"

import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(ROOT))
dataset = _load("code_dataset", CODE_DIR / "dataset.py")
model_utils = _load("code_model", CODE_DIR / "model.py")
loss_utils = _load("code_losses", CODE_DIR / "losses.py")
infer_utils = _load("code_infer", CODE_DIR / "inference.py")
bench_utils = _load("code_bench", CODE_DIR / "benchmark.py")
import eval as ev


class TestDatasetModule(unittest.TestCase):
    def test_load_and_check_split(self):
        labels_dir = ROOT / "data" / "labels"
        if (labels_dir / "train_subset0.csv").exists():
            train_df, val_df, test_df = dataset.load_split(labels_dir, fold=0)
            self.assertEqual(len(train_df), 10501)
            self.assertEqual(len(val_df), 3501)
            self.assertEqual(len(test_df), 3507)

            stats = dataset.check_split(train_df, val_df, test_df)
            self.assertEqual(stats["n"]["total"], 17509)
            self.assertEqual(stats["overlap"]["train_val"], 0)
            self.assertEqual(stats["overlap"]["train_test"], 0)
            self.assertEqual(stats["overlap"]["val_test"], 0)

    def test_transforms_shapes(self):
        t_train = dataset.build_transforms(train=True, img_size=224, aug="basic")
        t_val = dataset.build_transforms(train=False, img_size=224)
        from PIL import Image
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        out_tr = t_train(img)
        out_val = t_val(img)
        self.assertEqual(out_tr.shape, (3, 224, 224))
        self.assertEqual(out_val.shape, (3, 224, 224))


class TestModelModule(unittest.TestCase):
    def test_build_and_params(self):
        net = model_utils.build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
        params_m = model_utils.count_params(net)
        self.assertGreater(params_m, 20.0)
        self.assertLess(params_m, 35.0)

    def test_freeze_backbone(self):
        net = model_utils.build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
        trainable = [p for p in net.parameters() if p.requires_grad]
        head_classifier = net.get_classifier()
        head_params = list(head_classifier.parameters())
        self.assertEqual(len(trainable), len(head_params))

    def test_param_groups_three_groups(self):
        net = model_utils.build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
        groups = model_utils.param_groups(net, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0]["weight_decay"], 0.05)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[2]["lr"], 1e-3)


class TestLossesModule(unittest.TestCase):
    def test_focal_gamma_zero_matches_ce(self):
        logits = torch.randn(20, 9)
        targets = torch.randint(0, 9, (20,))
        focal_loss = loss_utils.FocalLoss(gamma=0.0)(logits, targets)
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)
        self.assertAlmostEqual(focal_loss.item(), ce_loss.item(), places=5)

    def test_label_smoothing_zero_matches_ce(self):
        logits = torch.randn(20, 9)
        targets = torch.randint(0, 9, (20,))
        ls_loss = loss_utils.LabelSmoothingCE(smoothing=0.0)(logits, targets)
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)
        self.assertAlmostEqual(ls_loss.item(), ce_loss.item(), places=5)

    def test_class_weights(self):
        counts = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 5000]
        w = loss_utils.class_weights(counts, beta=0.0)
        self.assertEqual(len(w), 9)
        self.assertGreater(w[0].item(), w[8].item())

    def test_cutmix_dimensions(self):
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        x_m, (y_a, y_b, lam) = loss_utils.mix_batch(x, y, alpha=1.0, mode="cutmix")
        self.assertEqual(x_m.shape, x.shape)
        self.assertEqual(len(y_a), 4)
        self.assertEqual(len(y_b), 4)
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)


class TestInferenceModule(unittest.TestCase):
    def test_temperature_scaling(self):
        rng = np.random.default_rng(0)
        n = 200
        labels = rng.integers(0, 9, n)
        logits = rng.normal(size=(n, 9))
        logits[np.arange(n), labels] += 2.0  # trained model signal
        T = infer_utils.fit_temperature(logits, labels)
        self.assertGreater(T, 0.2)
        self.assertLess(T, 3.0)

        probs_cal = infer_utils.apply_temperature(logits, T)
        self.assertEqual(probs_cal.shape, (n, 9))
        np.testing.assert_allclose(probs_cal.sum(axis=1), np.ones(n), atol=1e-5)

    def test_aggregate_views(self):
        logits1 = np.ones((10, 9))
        logits2 = np.zeros((10, 9))
        probs = infer_utils.aggregate_views([logits1, logits2], space="prob")
        self.assertEqual(probs.shape, (10, 9))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-5)


class TestBenchmarkModule(unittest.TestCase):
    def test_bench_cpu(self):
        res = bench_utils.bench(lambda: sum(range(1000)), warmup=2, iters=10)
        self.assertIn("p50", res)
        self.assertIn("p95", res)
        self.assertIn("p99", res)
        self.assertGreater(res["p50"], 0.0)


if __name__ == "__main__":
    unittest.main()
