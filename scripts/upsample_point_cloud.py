from pathlib import Path
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Simple KNN interpolation upsampling for point clouds.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=int, default=2, help="2 means roughly double the number of points.")
    parser.add_argument("--k", type=int, default=6, help="Number of nearest neighbors.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.factor < 1:
        raise ValueError("--factor must be 1 or larger.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if len(pcd.points) == 0:
        raise RuntimeError("Input point cloud has zero points.")

    points = np.asarray(pcd.points)
    has_colors = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_colors else None

    tree = o3d.geometry.KDTreeFlann(pcd)

    new_points = [points]
    new_colors = [colors] if has_colors else []

    if args.factor > 1:
        generated_points = []
        generated_colors = []

        for i, p in enumerate(points):
            _, idx, _ = tree.search_knn_vector_3d(p, args.k + 1)
            idx = [j for j in idx if j != i]

            if not idx:
                continue

            for _ in range(args.factor - 1):
                j = int(rng.choice(idx))
                t = float(rng.uniform(0.25, 0.75))

                q = points[j]
                new_p = (1.0 - t) * p + t * q
                generated_points.append(new_p)

                if has_colors:
                    c = colors[i]
                    d = colors[j]
                    new_c = (1.0 - t) * c + t * d
                    generated_colors.append(new_c)

        if generated_points:
            new_points.append(np.asarray(generated_points))

            if has_colors:
                new_colors.append(np.asarray(generated_colors))

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack(new_points))

    if has_colors:
        out.colors = o3d.utility.Vector3dVector(np.vstack(new_colors))

    distances = np.asarray(out.compute_nearest_neighbor_distance())
    median_distance = float(np.median(distances[distances > 0]))

    out.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=median_distance * 10.0,
            max_nn=30,
        )
    )

    o3d.io.write_point_cloud(str(output_path), out)

    print(f"Input points: {len(points)}")
    print(f"Output points: {len(out.points)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()