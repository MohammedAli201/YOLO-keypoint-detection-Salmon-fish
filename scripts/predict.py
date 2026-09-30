"""Run inference with a locally trained 20-keypoint salmon model."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/predictions"))
    args = parser.parse_args()
    if not args.weights.is_file() or not args.source.exists():
        parser.error("weights and source must exist locally")
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    shape = getattr(model.model, "kpt_shape", None)
    if shape is None or list(shape) != [20, 3]:
        parser.error("Expected a salmon pose model with kpt_shape [20, 3]")
    model.predict(source=str(args.source), save=True, project=str(args.output), name="salmon")


if __name__ == "__main__":
    main()
