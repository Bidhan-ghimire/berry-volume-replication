"""
calibrate_camera.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from checkerboard images."
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Folder containing checkerboard calibration images.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where calibration outputs will be saved.",
    )

    parser.add_argument(
        "--cols",
        required=True,
        type=int,
        help="Number of inner checkerboard corners across columns.",
    )

    parser.add_argument(
        "--rows",
        required=True,
        type=int,
        help="Number of inner checkerboard corners across rows.",
    )

    parser.add_argument(
        "--square-size-mm",
        required=True,
        type=float,
        help="Checkerboard square size in millimeters.",
    )

    parser.add_argument(
        "--detector",
        choices=["classic", "sb"],
        default="sb",
        help="Corner detector. Use 'sb' first. Use 'classic' if sb fails.",
    )

    parser.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Minimum number of successful checkerboard detections required.",
    )

    parser.add_argument(
        "--save-corners",
        action="store_true",
        help="Save preview images with detected checkerboard corners drawn.",
    )

    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def make_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp


def detect_corners(gray: np.ndarray, board_size: tuple[int, int], detector: str):
    if detector == "sb":
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        flags += getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        flags += getattr(cv2, "CALIB_CB_ACCURACY", 0)

        found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
        return found, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    found, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

    return found, corners


def compute_reprojection_errors(
    objpoints,
    imgpoints,
    used_images,
    camera_matrix,
    dist_coeffs,
    rvecs,
    tvecs,
):
    rows = []

    for i, image_name in enumerate(used_images):
        projected_points, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs,
        )

        observed = imgpoints[i].reshape(-1, 2)
        projected = projected_points.reshape(-1, 2)

        error_px = float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1))))

        rows.append(
            {
                "image": image_name,
                "reprojection_error_px": error_px,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir
    board_size = (args.cols, args.rows)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)

    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    print(f"Image folder: {image_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Images found: {len(image_paths)}")
    print(f"Checkerboard inner corners: {args.cols} x {args.rows}")
    print(f"Square size: {args.square_size_mm} mm")
    print(f"Detector: {args.detector}")

    object_template = make_object_points(
        cols=args.cols,
        rows=args.rows,
        square_size_mm=args.square_size_mm,
    )

    objpoints = []
    imgpoints = []
    used_images = []
    failed_rows = []
    image_size = None

    corners_dir = output_dir / "detected_corners"
    if args.save_corners:
        corners_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if img is None:
            failed_rows.append({"image": image_path.name, "reason": "could_not_read"})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]  # width, height

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            failed_rows.append(
                {
                    "image": image_path.name,
                    "reason": f"wrong_size_{current_size}_expected_{image_size}",
                }
            )
            continue

        found, corners = detect_corners(gray, board_size, args.detector)

        if not found:
            failed_rows.append({"image": image_path.name, "reason": "checkerboard_not_detected"})
            continue

        objpoints.append(object_template.copy())
        imgpoints.append(corners)
        used_images.append(image_path.name)

        if args.save_corners:
            preview = img.copy()
            cv2.drawChessboardCorners(preview, board_size, corners, found)
            cv2.imwrite(str(corners_dir / f"detected_{image_path.stem}.jpg"), preview)

        print(f"[OK] {image_path.name}")

    print()
    print(f"Successful detections: {len(used_images)}")
    print(f"Failed detections: {len(failed_rows)}")

    if len(used_images) < args.min_images:
        raise RuntimeError(
            f"Only {len(used_images)} successful detections. "
            f"At least {args.min_images} are required."
        )

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    reprojection_rows = compute_reprojection_errors(
        objpoints=objpoints,
        imgpoints=imgpoints,
        used_images=used_images,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )

    mean_reprojection_error = float(
        np.mean([row["reprojection_error_px"] for row in reprojection_rows])
    )

    dist = dist_coeffs.ravel()

    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    p1 = float(dist[2]) if len(dist) > 2 else 0.0
    p2 = float(dist[3]) if len(dist) > 3 else 0.0
    k3 = float(dist[4]) if len(dist) > 4 else 0.0

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    colmap_opencv_params = f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"
    colmap_full_opencv_params = f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2},{k3},0,0,0"

    summary = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "checkerboard_cols_inner_corners": int(args.cols),
        "checkerboard_rows_inner_corners": int(args.rows),
        "square_size_mm": float(args.square_size_mm),
        "detector": args.detector,
        "successful_detections": len(used_images),
        "failed_detections": len(failed_rows),
        "rms_reprojection_error": float(rms_error),
        "mean_per_image_reprojection_error_px": mean_reprojection_error,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        "colmap_opencv_params": colmap_opencv_params,
        "colmap_full_opencv_params": colmap_full_opencv_params,
    }

    np.savez(
        output_dir / "camera_calibration_checkerboard.npz",
        image_size=np.array(image_size),
        checkerboard=np.array(board_size),
        square_size_mm=np.array(args.square_size_mm),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_error=np.array(rms_error),
        mean_per_image_reprojection_error_px=np.array(mean_reprojection_error),
        used_images=np.array(used_images),
    )

    summary_rows = [{"parameter": key, "value": value} for key, value in summary.items()]
    write_csv(output_dir / "calibration_summary.csv", summary_rows, ["parameter", "value"])

    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    write_csv(
        output_dir / "per_image_reprojection_errors.csv",
        reprojection_rows,
        ["image", "reprojection_error_px"],
    )

    write_csv(output_dir / "used_images.csv", [{"image": x} for x in used_images], ["image"])

    write_csv(
        output_dir / "failed_images.csv",
        failed_rows,
        ["image", "reason"],
    )

    (output_dir / "colmap_opencv_camera_params.txt").write_text(
        "# COLMAP OPENCV camera model\n"
        "# Order: fx,fy,cx,cy,k1,k2,p1,p2\n"
        f"{colmap_opencv_params}\n",
        encoding="utf-8",
    )

    (output_dir / "colmap_full_opencv_camera_params.txt").write_text(
        "# COLMAP FULL_OPENCV camera model\n"
        "# Order: fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6\n"
        f"{colmap_full_opencv_params}\n",
        encoding="utf-8",
    )

    print()
    print("Calibration complete.")
    print(f"Image size: {image_size}")
    print(f"RMS reprojection error: {rms_error}")
    print(f"Mean per-image reprojection error: {mean_reprojection_error}")
    print()
    print("COLMAP OPENCV camera parameters:")
    print(colmap_opencv_params)
    print()
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
