import open3d as o3d
import numpy as np
import pandas as pd


def do_pca(mesh_path, visualization_offset=np.array([0, 0, 0])):
    # Load the mesh with open3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi/4, np.pi/5)))

    # Obtain numpy array containing the mesh's vertex positions.
    # Note: modifying values in the array will also change the corresponding vertex
    # positions in the mesh object
    vertices = np.asarray(mesh.vertices)

    # Compute barycenter
    barycenter = np.mean(vertices, axis=0)

    # Align mesh and vertices to origin
    vertices -= barycenter

    # Compute covariance matrix on vertices array
    cov = np.cov(vertices.transpose())

    # Compute eigen vectors and eigen values of covariance matrix
    # The eigen vectors are the columns of the returned 2d-array
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Order the eigenvectors according to the magnitude of their corresponding eigen value
    # (from largest to smallest)
    eigencombined = [(eigenvalues[i], eigenvectors[:, i]) for i in range(3)]
    eigencombined.sort(key=lambda x:x[0], reverse=True)
    eigenvectors = [item[1] for item in eigencombined]
    eigenvalues = [item[0] for item in eigencombined]

    # Replace the minor eigen vector with cross product of major and medium (to enforce right-handed reference frame)
    eigenvectors.pop(2)
    eigenvectors.append(np.cross(eigenvectors[0], eigenvectors[1]))
    print(eigenvectors)

    # Visualize the mesh including principle axes
    # Apply visualization offset to mesh
    vertices += visualization_offset

    # Create lines representing the eigen vectors
    eigvec_startpoint = np.zeros(3) + visualization_offset
    eigvec_points = [eigvec_startpoint] + [vec + visualization_offset for vec in eigenvectors]
    eigvec_line_indices = [[0, 1], [0, 2], [0, 3]]
    eigvec_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(eigvec_points),
        lines=o3d.utility.Vector2iVector(eigvec_line_indices),
    )

    # Add colors to the lines
    # Red, green, blue for major, 2nd-, and 3rd-largest eigen vectors, respectively. 
    line_colors = np.array([[255,0,0], [0,255,0], [0,0,255]])
    eigvec_lineset.colors = o3d.utility.Vector3dVector(line_colors)

    return mesh, eigvec_lineset


def main():
    # Parameters
    mesh1, lineset1 = do_pca("./database/Jet\m1155.obj")
    print(lineset1)
    mesh2, lineset2 = do_pca("m514_REMESHED.obj", np.array([2, 0, 0]))

    o3d.visualization.draw_geometries(
        [mesh1, mesh2, lineset1, lineset2], width=1280, height=720, mesh_show_wireframe=True)

if __name__ == '__main__':
    main()
