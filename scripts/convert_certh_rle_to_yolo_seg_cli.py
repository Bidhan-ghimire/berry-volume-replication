"""
CERTH COCO-style RLE or polygon annotations to YOLO segmentation converter.

No dataset paths are hardcoded. Provide paths from the command line.

Example from the repository root:
    python scripts/convert_certh_rle_to_yolo_seg_cli.py --annotations-dir "data/raw/certh/annotations/annotations" --images-dir "data/raw/certh/images/images/multiple-instance-multiple-class" --output "data/processed/certh_yolo_seg" --overwrite

Output structure:
    output/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test
        dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ConverterConfig:
    annotations_dir: Path
    images_dir: Path
    output: Path
    split_files: dict[str, str]
    class_id: int
    class_name: str
    min_contour_area: float
    epsilon_ratio: float
    keep_largest_contour_only: bool
    overwrite: bool
    dataset_yaml_name: str


def parse_args() -> ConverterConfig:
    parser = argparse.ArgumentParser(
        description="Convert CERTH COCO-style RLE or polygon annotations to YOLO segmentation format."
    )

    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="Folder containing annotation JSON files, e.g. data/raw/certh/annotations/annotations",
    )

    parser.add_argument(
        "--images-dir",
        required=True,
        help="Folder containing raw images, e.g. data/raw/certh/images/images/multiple-instance-multiple-class",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for YOLO dataset, e.g. data/processed/certh_yolo_seg",
    )

    parser.add_argument("--train-json", default="mimc_train_images.json")
    parser.add_argument("--val-json", default="mimc_valid_images.json")
    parser.add_argument("--test-json", default="mimc_test_images.json")

    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", default="grape_cluster")

    parser.add_argument("--min-contour-area", type=float, default=25.0)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)

    contour_group = parser.add_mutually_exclusive_group()
    contour_group.add_argument(
        "--keep-largest-contour-only",
        dest="keep_largest_contour_only",
        action="store_true",
        help="Keep only the largest contour from each mask annotation.",
    )
    contour_group.add_argument(
        "--keep-all-contours",
        dest="keep_largest_contour_only",
        action="store_false",
        help="Keep all valid contours from each mask annotation.",
    )
    parser.set_defaults(keep_largest_contour_only=True)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder before writing new results.",
    )

    parser.add_argument(
        "--dataset-yaml-name",
        default="dataset.yaml",
        help="Name of the dataset YAML file written inside the output folder.",
    )

    args = parser.parse_args()

    split_files = {
        "train": args.train_json,
        "val": args.val_json,
        "test": args.test_json,
    }

    return ConverterConfig(
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
        output=Path(args.output),
        split_files=split_files,
        class_id=args.class_id,
        class_name=args.class_name,
        min_contour_area=args.min_contour_area,
        epsilon_ratio=args.epsilon_ratio,
        keep_largest_contour_only=args.keep_largest_contour_only,
        overwrite=args.overwrite,
        dataset_yaml_name=args.dataset_yaml_name,
    )


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)


def build_image_lookup(images_dir: Path) -> dict[str, Path]:
    """Find all images and store them by file name."""
    image_lookup = {}

    for image_path in images_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_lookup[image_path.name] = image_path

    return image_lookup


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def yolo_polygon_line(points: np.ndarray, width: int, height: int, class_id: int) -> str:
    """Convert polygon points from pixel coordinates to one YOLO label line."""
    values = [str(class_id)]

    for x, y in points:
        x_normalized = clamp01(float(x) / float(width))
        y_normalized = clamp01(float(y) / float(height))
        values.append(f"{x_normalized:.6f}")
        values.append(f"{y_normalized:.6f}")

    return " ".join(values)


def decode_coco_rle(segmentation: dict) -> np.ndarray:
    """Decode one COCO compressed RLE segmentation into a binary mask."""
    rle = dict(segmentation)

    # pycocotools often expects compressed RLE counts as bytes, not as a string.
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")

    mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return (mask > 0).astype(np.uint8)


def rle_to_yolo_lines(segmentation: dict, width: int, height: int, config: ConverterConfig) -> list[str]:
    """Convert one RLE mask annotation into one or more YOLO segmentation lines."""
    mask = decode_coco_rle(segmentation)

    # Prevent failure if the JSON size is inconsistent.
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_contour_area:
            continue

        epsilon = config.epsilon_ratio * cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        points = polygon.reshape(-1, 2)

        if len(points) >= 3:
            valid_contours.append((area, points))

    if not valid_contours:
        return []

    if config.keep_largest_contour_only:
        valid_contours = [max(valid_contours, key=lambda item: item[0])]

    return [
        yolo_polygon_line(points, width, height, config.class_id)
        for _, points in valid_contours
    ]


def polygon_to_yolo_lines(segmentation: list, width: int, height: int, config: ConverterConfig) -> list[str]:
    """Fallback for COCO polygon annotations, if any appear in the JSON."""
    if not segmentation:
        return []

    if all(isinstance(value, (int, float)) for value in segmentation):
        polygons = [segmentation]
    else:
        polygons = segmentation

    yolo_lines = []

    for polygon in polygons:
        if not isinstance(polygon, list):
            continue
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            continue

        points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        yolo_lines.append(yolo_polygon_line(points, width, height, config.class_id))

    return yolo_lines


def annotation_to_yolo_lines(annotation: dict, width: int, height: int, config: ConverterConfig) -> list[str]:
    segmentation = annotation.get("segmentation")

    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        return rle_to_yolo_lines(segmentation, width, height, config)

    if isinstance(segmentation, list):
        return polygon_to_yolo_lines(segmentation, width, height, config)

    return []


def get_image_size(image_record: dict, image_path: Path) -> tuple[int, int]:
    """Use width and height from JSON. If missing, read them from the image file."""
    width = int(image_record.get("width", 0) or 0)
    height = int(image_record.get("height", 0) or 0)

    if width > 0 and height > 0:
        return width, height

    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0

    height, width = image.shape[:2]
    return width, height


def convert_split(split: str, json_name: str, image_lookup: dict[str, Path], config: ConverterConfig) -> dict:
    json_path = config.annotations_dir / json_name

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    data = load_json(json_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    annotations_by_image_id = defaultdict(list)
    for annotation in annotations:
        image_id = str(annotation.get("image_id"))
        annotations_by_image_id[image_id].append(annotation)

    image_output_folder = config.output / "images" / split
    label_output_folder = config.output / "labels" / split
    make_folder(image_output_folder)
    make_folder(label_output_folder)

    stats = {
        "images_in_json": len(images),
        "annotations_in_json": len(annotations),
        "images_copied": 0,
        "missing_images": 0,
        "label_files_written": 0,
        "non_empty_label_files": 0,
        "objects_written": 0,
        "annotations_without_valid_polygon": 0,
    }

    for image_record in images:
        image_id = str(image_record.get("id"))
        file_name = image_record.get("file_name") or image_record.get("filename") or image_record.get("name")

        if not file_name:
            continue

        image_name = Path(file_name).name
        source_image_path = image_lookup.get(image_name)

        if source_image_path is None:
            stats["missing_images"] += 1
            continue

        width, height = get_image_size(image_record, source_image_path)
        if width <= 0 or height <= 0:
            stats["missing_images"] += 1
            continue

        destination_image_path = image_output_folder / image_name
        shutil.copy2(source_image_path, destination_image_path)
        stats["images_copied"] += 1

        yolo_lines = []
        for annotation in annotations_by_image_id.get(image_id, []):
            lines = annotation_to_yolo_lines(annotation, width, height, config)

            if not lines:
                stats["annotations_without_valid_polygon"] += 1

            yolo_lines.extend(lines)

        label_path = label_output_folder / f"{Path(image_name).stem}.txt"
        label_text = "\n".join(yolo_lines)

        if label_text:
            label_text += "\n"

        label_path.write_text(label_text, encoding="utf-8")

        stats["label_files_written"] += 1
        stats["objects_written"] += len(yolo_lines)

        if yolo_lines:
            stats["non_empty_label_files"] += 1

    return stats


def write_dataset_yaml(config: ConverterConfig) -> None:
    yaml_text = (
        f"path: {config.output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  {config.class_id}: {config.class_name}\n"
    )

    (config.output / config.dataset_yaml_name).write_text(yaml_text, encoding="utf-8")


def main() -> None:
    config = parse_args()

    if not config.annotations_dir.exists():
        raise FileNotFoundError(f"Annotation folder not found: {config.annotations_dir}")

    if not config.images_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {config.images_dir}")

    if config.overwrite and config.output.exists():
        shutil.rmtree(config.output)

    make_folder(config.output)

    image_lookup = build_image_lookup(config.images_dir)
    if not image_lookup:
        raise RuntimeError(f"No images found under {config.images_dir}")

    print(f"Images found: {len(image_lookup)}")
    print(f"Annotation folder: {config.annotations_dir}")
    print(f"Image folder: {config.images_dir}")
    print(f"Output folder: {config.output}\n")

    all_stats = {}
    for split, json_name in config.split_files.items():
        all_stats[split] = convert_split(split, json_name, image_lookup, config)

    write_dataset_yaml(config)

    total_objects = 0
    total_non_empty_label_files = 0

    print("CERTH RLE to YOLO segmentation conversion finished.\n")

    for split, stats in all_stats.items():
        print(split.upper())
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

        total_objects += stats["objects_written"]
        total_non_empty_label_files += stats["non_empty_label_files"]

    print(f"Total non-empty label files: {total_non_empty_label_files}")
    print(f"Total YOLO objects written: {total_objects}")
    print(f"Dataset YAML: {config.output / config.dataset_yaml_name}")


if __name__ == "__main__":
    main()
