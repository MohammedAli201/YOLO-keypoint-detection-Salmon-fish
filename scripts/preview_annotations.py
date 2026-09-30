"""Render existing dataset annotations; this does not run model inference."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--output", type=Path, default=root / "docs/annotation-example.png")
    args = parser.parse_args()
    image_path = args.image or sorted((root / "data/images/train").glob("*.png"))[0]
    label_path = args.label or root / "data/labels/train" / f"{image_path.stem}.txt"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    figure, axes = plt.subplots(1, 2, figsize=(13, 3.25), facecolor="#f5f7fb")
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    axes[0].set_title("Dataset image", loc="left", fontweight="bold", pad=14)
    axes[1].set_title("Existing annotation · 20 keypoints", loc="left", fontweight="bold", pad=14)
    for row in label_path.read_text().splitlines():
        data = np.array([float(value) for value in row.split()])
        if len(data) != 65:
            raise ValueError("Expected one box and 20 keypoints with visibility.")
        cx, cy, bw, bh = data[1:5]
        axes[1].add_patch(Rectangle(((cx-bw/2)*width, (cy-bh/2)*height), bw*width, bh*height, fill=False, edgecolor="#1ac6b4", linewidth=1.4))
        for index, (x, y, visible) in enumerate(data[5:].reshape(20, 3)):
            if visible:
                axes[1].scatter(x*width, y*height, s=26, c="#ffca45", edgecolors="#182235", linewidths=.7)
                axes[1].annotate(str(index), (x*width,y*height), xytext=(4,-11), textcoords="offset points", fontsize=8, color="#15213d", bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=1))
    figure.suptitle("Salmon keypoint detection", x=.045, ha="left", fontsize=21, fontweight="bold", color="#15213d")
    figure.text(.045,.04,"Annotation preview from the committed dataset — not a model prediction or an accuracy result.",fontsize=10,color="#475569")
    figure.subplots_adjust(top=.78,bottom=.18,left=.04,right=.98,wspace=.10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=160,facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
