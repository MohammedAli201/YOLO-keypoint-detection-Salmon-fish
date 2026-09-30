"""Validate image/label pairing and YOLO pose rows without training a model."""
from pathlib import Path
import argparse
import math

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def validate_row(row, keypoints=20):
    fields = row.split()
    if len(fields) != 5 + 3 * keypoints:
        return f"expected {5 + 3 * keypoints} fields, got {len(fields)}"
    try:
        values = [float(field) for field in fields]
    except ValueError:
        return "non-numeric label field"
    if not all(math.isfinite(value) for value in values):
        return "nonfinite label field"
    if values[0] != 0:
        return "expected class 0 (fish)"
    if not all(0 <= value <= 1 for value in values[1:5]):
        return "bounding box coordinates must be normalised to [0, 1]"
    if values[3] <= 0 or values[4] <= 0:
        return "bounding box width and height must be positive"
    for index in range(5, len(values), 3):
        x, y, visible = values[index:index+3]
        if visible not in (0, 1, 2):
            return "keypoint visibility must be 0, 1 or 2"
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            return "keypoint coordinates must be normalised to [0, 1]"
    return None


def validate_split(root, split):
    root = Path(root)
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    errors = []
    if not image_dir.is_dir() or not label_dir.is_dir():
        return 0, [f"{split}: missing images or labels directory"]
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        errors.append(f"{split}: no images found")
    stems = {image.stem for image in images}
    if len(stems) != len(images):
        errors.append(f"{split}: duplicate image stems")
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            errors.append(f"{split}/{image.name}: missing label")
            continue
        rows = label.read_text().splitlines()
        if not rows:
            errors.append(f"{split}/{label.name}: empty annotation")
        for number, row in enumerate(rows, 1):
            problem = validate_row(row)
            if problem:
                errors.append(f"{split}/{label.name}:{number}: {problem}")
    for label in sorted(label_dir.glob("*.txt")):
        if label.stem not in stems:
            errors.append(f"{split}/{label.name}: label has no matching image")
    return len(images), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    errors = []
    for split in ("train", "val"):
        count, issues = validate_split(args.data, split)
        print(f"{split}: {count} images, {len(issues)} problems")
        errors.extend(issues)
    train = {p.stem for p in (args.data / "images/train").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    val = {p.stem for p in (args.data / "images/val").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    if train & val:
        errors.append("train and validation contain overlapping image stems")
    for issue in errors:
        print(issue)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
