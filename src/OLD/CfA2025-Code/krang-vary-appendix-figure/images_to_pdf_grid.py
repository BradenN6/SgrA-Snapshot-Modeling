"""
images_to_pdf_grid.py

Arranges 16 PNG images in a 4x4 grid and saves the result as a PDF.

Usage:
    python images_to_pdf_grid.py image1.png image2.png ... image16.png
    python images_to_pdf_grid.py *.png
    python images_to_pdf_grid.py --dir /path/to/images --output grid.pdf

Requirements:
    pip install matplotlib Pillow
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


def make_grid_pdf(image_paths: list[str], output_pdf: str = "grid.pdf",
                  title: str = "", dpi: int = 150) -> None:
    """
    Arrange up to 16 images in a 4x4 grid and save as a PDF.

    Parameters
    ----------
    image_paths : list of str
        Paths to the PNG images (exactly 16 expected).
    output_pdf  : str
        Output PDF file path.
    title       : str
        Optional title printed above the grid.
    dpi         : int
        Resolution used when rendering the figure.
    """
    if len(image_paths) != 16:
        raise ValueError(
            f"Expected exactly 16 images, got {len(image_paths)}."
        )

    rows, cols = 4, 4
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3, rows * 3 + (0.6 if title else 0)),
    )

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    for idx, (ax, path) in enumerate(zip(axes.flat, image_paths)):
        img = imread(path)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    fig.savefig(output_pdf, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def collect_images(args) -> list[str]:
    """Resolve the list of image paths from CLI arguments."""
    if args.dir:
        folder = Path(args.dir)
        paths = sorted(folder.glob("*.png"))
        if len(paths) < 16:
            sys.exit(
                f"Error: found only {len(paths)} PNG files in '{args.dir}'. Need 16."
            )
        if len(paths) > 16:
            print(
                f"Warning: found {len(paths)} PNG files; using the first 16 "
                f"(sorted by name)."
            )
        return [str(p) for p in paths[:16]]

    if args.images:
        if len(args.images) != 16:
            sys.exit(
                f"Error: expected 16 image paths, got {len(args.images)}."
            )
        return args.images

    sys.exit("Error: provide either 16 image paths or --dir <directory>.")


def main():
    parser = argparse.ArgumentParser(
        description="Arrange 16 PNG images in a 4x4 grid and export as PDF."
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="16 PNG image file paths (positional).",
    )
    parser.add_argument(
        "--dir",
        metavar="DIR",
        help="Directory containing PNG files (first 16, sorted by name).",
    )
    parser.add_argument(
        "--output", "-o",
        default="grid.pdf",
        metavar="OUTPUT",
        help="Output PDF file path (default: grid.pdf).",
    )
    parser.add_argument(
        "--title",
        default="",
        metavar="TITLE",
        help="Optional title to display above the grid.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering resolution in DPI (default: 150).",
    )

    args = parser.parse_args()
    image_paths = collect_images(args)
    make_grid_pdf(image_paths, output_pdf=args.output, title=args.title, dpi=args.dpi)


if __name__ == "__main__":
    main()
