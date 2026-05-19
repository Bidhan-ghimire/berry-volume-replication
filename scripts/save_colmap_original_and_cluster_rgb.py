from ultralytics import YOLO
from pathlib import Path
import argparse
import shutil
import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def numeric_key(path: Path):
    """
    Sort numeric filenames correctly.

    Example:
        1.jpg, 2.jpg, 10.jpg

    instead of:
        1.jpg, 10.jpg, 2.jpg
    """

    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem.lower())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create two matching folders: original images and RGB cluster-only JPEG images."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO segmentation model .pt file",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Folder containing original input images",
    )

    parser.add_argument(
        "--out-original",
        required=True,
        help="Output folder for selected original images",
    )

    parser.add_argument(
        "--out-mask",
        required=True,
        help="Output folder for selected RGB masked JPEG images",
    )

    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Use every Nth image",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of selected images. Use 0 for all selected images.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device like 0, or cpu",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Target YOLO class ID for grape cluster",
    )

    parser.add_argument(
        "--mask-mode",
        choices=["largest", "union"],
        default="largest",
        help="largest = keep biggest detected cluster, union = merge all detected clusters",
    )

    parser.add_argument(
        "--close-kernel",
        type=int,
        default=0,
        help="Optional morphological closing kernel size, e.g. 5 or 7",
    )

    parser.add_argument(
        "--dilate-pixels",
        type=int,
        default=0,
        help="Optional dilation size in pixels, e.g. 3 or 5",
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output folders before writing new results",
    )

    return parser.parse_args()


def get_selected_images(source_dir, every_n, max_images):
    if every_n < 1:
        raise ValueError("--every-n must be 1 or larger")

    images = sorted(
        [
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ],
        key=numeric_key,
    )

    images = images[::every_n]

    if max_images > 0:
        images = images[:max_images]

    if not images:
        raise RuntimeError(f"No images found in: {source_dir}")

    return images


def postprocess_mask(mask, close_kernel=0, dilate_pixels=0):
    mask = (mask > 0).astype(np.uint8) * 255

    if close_kernel and close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if dilate_pixels and dilate_pixels > 0:
        kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def build_mask(result, image_shape, class_id=0, mask_mode="largest"):
    h, w = image_shape[:2]

    if result.masks is None or result.boxes is None:
        return np.zeros((h, w), dtype=np.uint8), 0

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    selected = []

    for i, cls in enumerate(classes):
        if cls == class_id:
            m = masks[i]

            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

            m = (m > 0.5).astype(np.uint8)
            selected.append(m)

    if not selected:
        return np.zeros((h, w), dtype=np.uint8), 0

    if mask_mode == "largest":
        areas = [int(m.sum()) for m in selected]
        mask = selected[int(np.argmax(areas))]
        return mask, 1

    mask = np.zeros((h, w), dtype=np.uint8)

    for m in selected:
        mask = np.maximum(mask, m)

    return mask, len(selected)


def save_original_as_jpg(src_path, dst_path, image, jpg_quality):
    """
    Save the selected original image as JPEG.

    If the source is already JPG or JPEG, copy it directly.
    If the source is PNG, TIFF, BMP, or WEBP, convert it to JPEG.
    """

    if src_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(src_path, dst_path)
        return True

    return cv2.imwrite(
        str(dst_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
    )


def check_output_folders(source_dir, out_original_dir, out_mask_dir):
    source_resolved = source_dir.resolve()
    out_original_resolved = out_original_dir.resolve()
    out_mask_resolved = out_mask_dir.resolve()

    if out_original_resolved == source_resolved:
        raise ValueError("--out-original cannot be the same folder as --source")

    if out_mask_resolved == source_resolved:
        raise ValueError("--out-mask cannot be the same folder as --source")

    if out_original_resolved == out_mask_resolved:
        raise ValueError("--out-original and --out-mask must be different folders")


def main():
    args = parse_args()

    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 1 and 100")

    source_dir = Path(args.source)
    out_original_dir = Path(args.out_original)
    out_mask_dir = Path(args.out_mask)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    check_output_folders(source_dir, out_original_dir, out_mask_dir)

    if args.overwrite:
        if out_original_dir.exists():
            shutil.rmtree(out_original_dir)

        if out_mask_dir.exists():
            shutil.rmtree(out_mask_dir)

    out_original_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_selected_images(
        source_dir=source_dir,
        every_n=args.every_n,
        max_images=args.max_images,
    )

    model = YOLO(args.model)

    print(f"Selected images: {len(image_files)}")
    print(f"Original output folder: {out_original_dir}")
    print(f"RGB mask output folder: {out_mask_dir}")

    saved = 0
    skipped = 0

    for idx, img_path in enumerate(image_files, start=1):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"[SKIP] Could not read image: {img_path}")
            skipped += 1
            continue

        out_name = f"{idx:05d}.jpg"

        original_out_path = out_original_dir / out_name
        mask_out_path = out_mask_dir / out_name

        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        result = results[0]

        mask, n_used = build_mask(
            result=result,
            image_shape=image.shape,
            class_id=args.class_id,
            mask_mode=args.mask_mode,
        )

        mask = postprocess_mask(
            mask=mask,
            close_kernel=args.close_kernel,
            dilate_pixels=args.dilate_pixels,
        )

        masked_rgb = image.copy()
        masked_rgb[mask == 0] = 0

        ok_original = save_original_as_jpg(
            src_path=img_path,
            dst_path=original_out_path,
            image=image,
            jpg_quality=args.jpg_quality,
        )

        ok_mask = cv2.imwrite(
            str(mask_out_path),
            masked_rgb,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality],
        )

        if not ok_original:
            print(f"[FAILED] Could not save original image: {original_out_path}")
            skipped += 1
            continue

        if not ok_mask:
            print(f"[FAILED] Could not save RGB mask image: {mask_out_path}")
            skipped += 1
            continue

        saved += 1

        print(
            f"[{idx}/{len(image_files)}] saved: {out_name} | detections used: {n_used}"
        )

    print()
    print("Done.")
    print(f"Saved pairs: {saved}")
    print(f"Skipped images: {skipped}")
    print(f"Original images: {out_original_dir}")
    print(f"RGB masked images: {out_mask_dir}")


if __name__ == "__main__":
    main()