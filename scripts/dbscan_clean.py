"""
dbscan_clean.py
"""

from pathlib import Path
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Clean a point cloud using DBSCAN.")
    parser.add_argument("--input", required=True, help="Input point cloud .ply")
    parser.add_argument("--output", required=True, help="Output cleaned point cloud .ply")
    parser.add_argument("--eps", type=float, default=0.05, help="DBSCAN eps")
    parser.add_argument("--min-points", type=int, default=10, help="DBSCAN min_points")
    parser.add_argument(
        "--keep",
        choices=["non_noise", "largest"],
        default="non_noise",
        help="non_noise keeps all DBSCAN clusters. largest keeps only the largest cluster.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if len(pcd.points) == 0:
        raise RuntimeError("Input point cloud has zero points.")

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=args.eps,
            min_points=args.min_points,
            print_progress=False,
        )
    )

    valid = labels >= 0

    if not np.any(valid):
        raise RuntimeError("DBSCAN found no valid clusters. Increase eps or lower min-points.")

    if args.keep == "largest":
        cluster_ids, counts = np.unique(labels[valid], return_counts=True)
        largest_cluster = cluster_ids[np.argmax(counts)]
        keep_indices = np.where(labels == largest_cluster)[0]
    else:
        keep_indices = np.where(valid)[0]

    cleaned = pcd.select_by_index(keep_indices)

    if len(cleaned.points) == 0:
        raise RuntimeError("Cleaned point cloud has zero points.")

    o3d.io.write_point_cloud(str(output_path), cleaned)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
