"""Train YOLOv8m on KITTI MOT for 5-class fine-tune (T4b).

Run:
    .venv/Scripts/python scripts/train_detector.py            # full 100-epoch training
    .venv/Scripts/python scripts/train_detector.py --dry-run  # 1-epoch validation

Default settings target ~5-7 hour overnight training on RTX 3070 Ti Laptop (8GB VRAM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m on KITTI MOT")
    parser.add_argument("--data", default="data/kitti_yolo/data.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", default="yolov8m_kitti", help="Run name")
    parser.add_argument("--model", default="yolov8m.pt", help="Pretrained weights")
    parser.add_argument("--dry-run", action="store_true", help="1-epoch sanity check only")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Aborting — CPU training is impractical.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run prepare_kitti_yolo_finetune first.")
        return 1

    epochs = 1 if args.dry_run else args.epochs
    name = f"{args.name}_dryrun" if args.dry_run else args.name
    print(f"Training: {epochs} epoch(s), batch={args.batch}, imgsz={args.imgsz}, name={name}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    project_dir = str(Path("runs/train").resolve())
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        name=name,
        project=project_dir,
        device=0,
        exist_ok=True,
        verbose=True,
    )

    # Find weights at the actual save location reported by Ultralytics
    save_dir = Path(model.trainer.save_dir)
    weights = save_dir / "weights" / "best.pt"
    if not weights.exists():
        # Fallback: check the expected path directly
        weights = Path("runs/train") / name / "weights" / "best.pt"
    if weights.exists():
        size_mb = weights.stat().st_size / 1e6
        print(f"DONE. Best weights at: {weights} ({size_mb:.1f} MB)")
        return 0
    print(f"WARNING: Expected weights at {save_dir / 'weights' / 'best.pt'} but not found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
