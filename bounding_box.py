import open3d as o3d
import pca
import numpy as np


# Parameters
mesh_path = "D00921.obj"
show_aabb = True
show_obb = True

# Load the mesh with open3d
mesh = o3d.io.read_triangle_mesh(mesh_path)

# Get axis-aligned bounding box (AABB)
aabb = mesh.get_axis_aligned_bounding_box()
aabb_points = aabb.get_box_points()
aabb_line_indices = [[0, 1], [1, 6], [6, 3], [3, 0], [0, 2], [2, 5], [5, 4], [4, 7], [7, 2], [6, 4], [1, 7], [3, 5]]
aabb_lineset = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(aabb_points),
    lines=o3d.utility.Vector2iVector(aabb_line_indices),
)
red = [255, 0, 0]
aabb_lineset.colors = o3d.utility.Vector3dVector(np.array([red] * 12))

# Get oriented bounding box (OBB)
obb = mesh.get_oriented_bounding_box()
obb_points = obb.get_box_points()
obb_line_indices = [[0, 1], [1, 6], [6, 3], [3, 0], [0, 2], [2, 5], [5, 4], [4, 7], [7, 2], [6, 4], [1, 7], [3, 5]]
obb_lineset = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(obb_points),
    lines=o3d.utility.Vector2iVector(obb_line_indices),
)
blue = [0, 0, 255]
obb_lineset.colors = o3d.utility.Vector3dVector(np.array([blue] * 12))

# Visualize the geometry
to_draw = [mesh]
to_draw.append(aabb_lineset) if show_aabb else print("Not showing Axis-Aligned Bounding Box")
to_draw.append(obb_lineset) if show_obb else print("Not showing Oriented Bounding Box")
o3d.visualization.draw_geometries(
    to_draw,
    width=1280,
    height=720,
    mesh_show_wireframe=True
)
