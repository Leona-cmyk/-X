from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import WindowedNPPADDataset, build_class_maps, load_trainable_manifest, split_manifest_by_file
from src.models.deep_timeseries import FocalLoss, MultiTaskTemporalNet
from src.models.nucleoguard_mtsformer import NucleoGuardMTSFormer, count_parameters
from src.nppad_paths import ADVANCED_MODEL_PATH, REPORT_ROOT, SCALER_PATH
from src.preprocessing import TimeSeriesStandardScaler, write_sensor_metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", "--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", "--window_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--early-ratio", "--early_ratio", type=float, default=1.0)
    p.add_argument("--random-early-ratio", "--random_early_ratio", action="store_true")
    p.add_argument("--lambda-reg", "--lambda_reg", type=float, default=0.01)
    p.add_argument("--lambda-consistency", "--lambda_consistency", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files-per-class", "--max_files_per_class", type=int, default=30)
    p.add_argument("--max-windows-per-file", "--max_windows_per_file", type=int, default=4)
    p.add_argument("--hidden-dim", "--hidden_dim", type=int, default=96)
    p.add_argument("--sensor-dim", "--sensor_dim", type=int, default=32)
    p.add_argument("--patch-size", "--patch_size", type=int, default=8)
    p.add_argument("--patch-stride", "--patch_stride", type=int, default=4)
    p.add_argument("--num-layers", "--num_layers", type=int, default=2)
    p.add_argument("--num-heads", "--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--loss", choices=["ce", "focal"], default="ce")
    p.add_argument("--gaussian-noise-std", "--gaussian_noise_std", type=float, default=0.005)
    p.add_argument("--sensor-dropout", "--sensor_dropout", type=float, default=0.02)
    p.add_argument("--time-mask-ratio", "--time_mask_ratio", type=float, default=0.02)
    p.add_argument("--magnitude-scale-std", "--magnitude_scale_std", type=float, default=0.01)
    p.add_argument("--mixed-precision", "--mixed_precision", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume model weights from outputs/models/advanced_model.pt when compatible.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "x_full": torch.stack([b["x_full"] for b in batch]),
        "early_ratio": torch.stack([b["early_ratio"] for b in batch]),
        "y_class": torch.stack([b["y_class"] for b in batch]),
        "y_severity": torch.stack([b["y_severity"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def top2_accuracy(y_true, probabilities) -> float:
    if len(y_true) == 0:
        return 0.0
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    return float(np.mean([yt in top2[i] for i, yt in enumerate(y_true)]))


def evaluate(model, loader, device, lambda_reg: float = 0.01):
    model.eval()
    y_true, y_pred, sev_true, sev_pred, probs_all, confs = [], [], [], [], [], []
    losses = []
    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            yc = batch["y_class"].to(device)
            ys = batch["y_severity"].to(device)
            out = model(x)
            losses.append((ce(out["logits"], yc) + lambda_reg * l1(out["severity"], ys)).item())
            probs = out["probabilities"].cpu().numpy()
            y_true.extend(yc.cpu().numpy().tolist())
            y_pred.extend(out["logits"].argmax(dim=-1).cpu().numpy().tolist())
            probs_all.extend(probs.tolist())
            confs.extend(probs.max(axis=1).tolist())
            sev_true.extend(ys.cpu().numpy().tolist())
            sev_pred.extend(out["severity"].cpu().numpy().tolist())
    probs_all = np.asarray(probs_all) if probs_all else np.zeros((0, 0))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro") if y_true else 0.0,
        "severity_mae": mean_absolute_error(sev_true, sev_pred) if sev_true else 0.0,
        "top2_accuracy": top2_accuracy(y_true, probs_all) if len(probs_all) else 0.0,
        "average_confidence": float(np.mean(confs)) if confs else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "sev_true": sev_true,
        "sev_pred": sev_pred,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)

    manifest = load_trainable_manifest(max_files_per_class=args.max_files_per_class)
    class_to_idx, idx_to_class = build_class_maps(manifest)
    train_m, val_m, test_m = split_manifest_by_file(manifest, seed=args.seed)

    scaler = TimeSeriesStandardScaler.fit_from_files(train_m["path"].tolist())
    scaler.save(SCALER_PATH)
    write_sensor_metadata(scaler.sensor_cols)

    train_ds = WindowedNPPADDataset(
        train_m,
        class_to_idx,
        scaler,
        args.window_size,
        args.stride,
        args.early_ratio,
        random_early_ratio=args.random_early_ratio,
        augment=True,
        gaussian_noise_std=args.gaussian_noise_std,
        sensor_dropout=args.sensor_dropout,
        time_mask_ratio=args.time_mask_ratio,
        magnitude_scale_std=args.magnitude_scale_std,
        max_windows_per_file=args.max_windows_per_file,
    )
    val_ds = WindowedNPPADDataset(val_m, class_to_idx, scaler, args.window_size, args.stride, args.early_ratio, max_windows_per_file=args.max_windows_per_file)
    test_ds = WindowedNPPADDataset(test_m, class_to_idx, scaler, args.window_size, args.stride, args.early_ratio, max_windows_per_file=args.max_windows_per_file)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = NucleoGuardMTSFormer(
        len(scaler.sensor_cols),
        len(class_to_idx),
        sensor_dim=args.sensor_dim,
        model_dim=args.hidden_dim,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    if args.resume and ADVANCED_MODEL_PATH.exists():
        checkpoint = torch.load(ADVANCED_MODEL_PATH, map_location=device)
        if checkpoint.get("model_name") == "NucleoGuard-MTSFormer" and len(checkpoint.get("sensor_cols", [])) == len(scaler.sensor_cols):
            missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
            print(f"Resumed from {ADVANCED_MODEL_PATH}; missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print("Resume skipped: checkpoint is incompatible with current NucleoGuard-MTSFormer config.")
    counts = train_m["accident"].map(class_to_idx).value_counts().sort_index().to_numpy()
    weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    ce_loss = FocalLoss(weight=weights) if args.loss == "focal" else nn.CrossEntropyLoss(weight=weights)
    reg_loss = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    amp_enabled = args.mixed_precision and device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = -1.0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = batch["x"].to(device)
            yc = batch["y_class"].to(device)
            ys = batch["y_severity"].to(device)
            x_full = batch["x_full"].to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(x)
                loss = ce_loss(out["logits"], yc) + args.lambda_reg * reg_loss(out["severity"], ys)
                if args.lambda_consistency > 0:
                    full_out = model(x_full)
                    early_logp = F.log_softmax(out["logits"], dim=-1)
                    full_prob = F.softmax(full_out["logits"].detach(), dim=-1)
                    loss = loss + args.lambda_consistency * F.kl_div(early_logp, full_prob, reduction="batchmean")
            opt.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(opt)
            grad_scaler.update()
            train_losses.append(loss.item())
        scheduler.step()
        val = evaluate(model, val_loader, device, lambda_reg=args.lambda_reg)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{k: v for k, v in val.items() if not k.startswith("y_") and not k.startswith("sev_")}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_name": "NucleoGuard-MTSFormer",
                    "model_state": model.state_dict(),
                    "sensor_cols": scaler.sensor_cols,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "window_size": args.window_size,
                    "hidden_dim": args.hidden_dim,
                    "sensor_dim": args.sensor_dim,
                    "patch_size": args.patch_size,
                    "patch_stride": args.patch_stride,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "params_count": count_parameters(model),
                    "args": vars(args),
                },
                ADVANCED_MODEL_PATH,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best val macro_f1={best_f1:.4f}")
            break

    checkpoint = torch.load(ADVANCED_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_loader, device, lambda_reg=args.lambda_reg)
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(test["y_true"], test["y_pred"], target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(test["y_true"], test["y_pred"])

    metrics = {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "severity_mae": test["severity_mae"],
        "top2_accuracy": test["top2_accuracy"],
        "average_confidence": test["average_confidence"],
        "params_count": checkpoint.get("params_count", count_parameters(model)),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "classes": labels,
        "window_size": args.window_size,
        "early_ratio": args.early_ratio,
    }
    (REPORT_ROOT / "advanced_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_ROOT / "advanced_config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in vars(args).items()), encoding="utf-8")
    pd.DataFrame(report).transpose().to_csv(REPORT_ROOT / "advanced_classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(REPORT_ROOT / "advanced_confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(REPORT_ROOT / "advanced_training_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"early_ratio": r, "accuracy": np.nan, "macro_f1": np.nan, "severity_mae": np.nan, "note": "Run scripts/evaluate_advanced.py"} for r in [0.1, 0.2, 0.3, 0.5, 1.0]]
    ).to_csv(REPORT_ROOT / "advanced_early_diagnosis_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "model_name": "Baseline RandomForest",
                "accuracy": json.loads((REPORT_ROOT / "metrics.json").read_text(encoding="utf-8")).get("accuracy") if (REPORT_ROOT / "metrics.json").exists() else np.nan,
                "macro_f1": json.loads((REPORT_ROOT / "metrics.json").read_text(encoding="utf-8")).get("macro_f1") if (REPORT_ROOT / "metrics.json").exists() else np.nan,
                "top2_accuracy": np.nan,
                "severity_mae": json.loads((REPORT_ROOT / "metrics.json").read_text(encoding="utf-8")).get("severity_mae") if (REPORT_ROOT / "metrics.json").exists() else np.nan,
                "params_count": np.nan,
                "inference_ms": np.nan,
                "notes": "Existing statistical-feature baseline",
            },
            {"model_name": "CNN-only", "accuracy": np.nan, "macro_f1": np.nan, "top2_accuracy": np.nan, "severity_mae": np.nan, "params_count": np.nan, "inference_ms": np.nan, "notes": "Not trained in this run; reserved for dedicated ablation"},
            {"model_name": "LSTM/GRU", "accuracy": np.nan, "macro_f1": np.nan, "top2_accuracy": np.nan, "severity_mae": np.nan, "params_count": np.nan, "inference_ms": np.nan, "notes": "Not trained in this run; reserved for dedicated ablation"},
            {"model_name": "Transformer-only", "accuracy": np.nan, "macro_f1": np.nan, "top2_accuracy": np.nan, "severity_mae": np.nan, "params_count": np.nan, "inference_ms": np.nan, "notes": "Not trained in this run; reserved for dedicated ablation"},
            {"model_name": "CNN + Transformer", "accuracy": np.nan, "macro_f1": np.nan, "top2_accuracy": np.nan, "severity_mae": np.nan, "params_count": np.nan, "inference_ms": np.nan, "notes": "Predecessor architecture; see earlier run if needed"},
            {"model_name": "CNN + Transformer + Multi-task", "accuracy": np.nan, "macro_f1": np.nan, "top2_accuracy": np.nan, "severity_mae": np.nan, "params_count": np.nan, "inference_ms": np.nan, "notes": "Not trained separately in this run"},
            {"model_name": "NucleoGuard-MTSFormer + Early Consistency + Robust Augmentation", "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "top2_accuracy": metrics["top2_accuracy"], "severity_mae": metrics["severity_mae"], "params_count": metrics["params_count"], "inference_ms": np.nan, "notes": "Current trained advanced model"},
        ]
    ).to_csv(REPORT_ROOT / "advanced_ablation.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
