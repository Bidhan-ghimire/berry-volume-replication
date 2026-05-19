"""
alpha_shape_mesh_creation.py
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import open3d as o3d
import trimesh


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def mesh_stats(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.is_empty:
        return 0, 0, False, ""

    watertight = bool(mesh.is_watertight)
    volume = abs(float(mesh.volume)) if watertight else ""

    return len(mesh.vertices), len(mesh.faces), watertight, volume


def main():
    parser = argparse.ArgumentParser(description="Create alpha-shape meshes from a point cloud.")
    parser.add_argument("--input", required=True, help="Input point cloud .ply")
    parser.add_argument("--out-dir", required=True, help="Output folder for alpha meshes")
    parser.add_argument("--multipliers", default="4,6,8,10,12,16", help="Comma-separated alpha multipliers")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if len(pcd.points) == 0:
        raise RuntimeError("Input point cloud has zero points.")

    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    if len(distances) == 0:
        raise RuntimeError("Could not estimate point spacing.")

    median_nn = float(np.median(distances))
    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]

    rows = []

    for mult in multipliers:
        alpha = median_nn * mult
        mesh_path = out_dir / f"alpha_mult_{safe_name(mult)}.ply"

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        vertices, faces, watertight, volume = mesh_stats(mesh_path)

        rows.append({
            "input": str(input_path),
            "mesh": str(mesh_path),
            "median_nn_distance": median_nn,
            "alpha_multiplier": mult,
            "alpha": alpha,
            "vertices": vertices,
            "faces": faces,
            "watertight": watertight,
            "volume_units3": volume,
        })

    report_path = out_dir / "alpha_shape_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
