"""Start a new local training run; no original benchmark is reproduced implicitly."""
from pathlib import Path
import argparse
import tempfile
import yaml
from validate_dataset import validate_split


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8m-pose.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 1:
        parser.error("epochs and imgsz must be positive")
    for split in ("train", "val"):
        _, problems = validate_split(root / "data", split)
        if problems:
            parser.error("Dataset validation failed; run scripts/validate_dataset.py")
    from ultralytics import YOLO
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["path"] = str((root / "data").resolve())
    with tempfile.TemporaryDirectory(prefix="salmon-data-") as folder:
        data_path = Path(folder) / "data.yaml"
        data_path.write_text(yaml.safe_dump(config))
        YOLO(args.model).train(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz,
                              device=args.device, seed=42, fliplr=0.0, flipud=0.0,
                              project=str(root / "runs"), name="salmon-pose")


if __name__ == "__main__":
    main()
