import open3d as o3d
import numpy as np
import os
import glob
import pymeshlab as pml
import copy
from conversions import pml_to_o3d, o3d_to_pml

def translate_to_center(mesh):
    """
    Translate the mesh so that center is at the coordinate-frame origin.
    """
    mesh.translate(-mesh.get_center())
    return mesh


def scale_unitcube(mesh):
    """
    Scale mesh to fit in a unit-sized cube.
    """
    center = mesh.get_center()

    #test is mesh is centered at origin
    if center[0] > 0.001 or center[1] > 0.001 or center[2] > 0.001:
        raise ValueError(
            f'Mesh must be centered at origin'
        )
    factor = 1 / max(mesh.get_max_bound() - mesh.get_min_bound())
    print("Factor: ", factor)
    mesh.scale(factor, center)
    return mesh


def get_eigen_vectors(mesh):
    # Load the mesh with open3d
    

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

    eigvec_startpoint = np.zeros(3)
    eigvec_points = [eigvec_startpoint] + [vec for vec in eigenvectors]
    eigvec_line_indices = [[0, 1], [0, 2], [0, 3]]
    eigvec_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(eigvec_points),
        lines=o3d.utility.Vector2iVector(eigvec_line_indices),
    )

    # Add colors to the lines
    # Red, green, blue for major, 2nd-, and 3rd-largest eigen vectors, respectively. 
    line_colors = np.array([[255,0,0], [0,255,0], [0,0,255]])
    eigvec_lineset.colors = o3d.utility.Vector3dVector(line_colors)

    return eigenvectors, eigvec_lineset


#flip
def flip_mesh(mesh):
    mass = [0, 0, 0]
    for triangle in mesh.triangles:
        center = [0, 0, 0]
        for v in triangle:
            vertex = mesh.vertices[v]
            center += vertex
        center = center / 3
        
        for i, C in enumerate(center):
            mass[i] += np.sign(C) * (C * C)
    flip_matrix = np.asarray(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    for i, C in enumerate(mass):
        if C > 0:
            print("flipped")
            flip_matrix[i, i] = -1

    return mesh.transform(flip_matrix)

def transform(mesh):
    eigen_vectors,_ = get_eigen_vectors(mesh)
   
    mesh_r = copy.deepcopy(mesh)

    mesh_r.rotate(eigen_vectors, center=(0, 0, 0))

    """below this line is for visualisation"""
    # Display the mesh including a world axis system.

    # Create the endpoints of each line. Each line is unit-length.
    # For the world axes, the origin is shared by all lines. So we have 4 endpoints in total
    line_endpoints = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    

    # List of indices into the 'line_endpoints' list, which describes which endpoints form which line
    line_indices = [[0, 1], [0, 2], [0, 3]]

    # Create a line set from the endpoints and indices
    world_axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_endpoints),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    # world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame() 

    # Render the line set and the mesh
    # o3d.visualization.draw_geometries([ mesh_r, world_axes], width=1280, height=720, mesh_show_wireframe=True)
    return mesh_r

def copymesh(mesh):
    mesh.save_current_mesh("tmp.obj")
    newmesh = pml.MeshSet()
    newmesh.load_new_mesh("tmp.obj")
    return newmesh

def vertexcount(mesh):
    return len(mesh.vertices)
def trianglecount(mesh):
    return len(mesh.triangles)


def remesh_increase(mesh, targetlen=0.02, iterations=3):
    # print(f"Vertices before: {mesh.current_mesh().vertex_number()}, Faces before: {mesh.current_mesh().face_number()}")
    # print(f"Refining mesh ({iterations} iterations)... ")
    mesh.meshing_isotropic_explicit_remeshing(
        targetlen=pml.AbsoluteValue(targetlen), iterations=iterations)

    vAfter = mesh.current_mesh().vertex_number()
    fAfter = mesh.current_mesh().face_number()
    # print(f"Vertices after: {vAfter}, Faces after: {fAfter}")
    return mesh

def remesh(mesh, classname, filename, target_vertices=2000, factor=0.3):
    # mesh = pml.MeshSet()
    # mesh.load_new_mesh(mesh_path)
    print(f"Vertices before: {mesh.current_mesh().vertex_number()}, Faces before: {mesh.current_mesh().face_number()}")

    # if mesh.current_mesh().vertex_number() < target_vertices * (1-factor):
    count = 0
    while mesh.current_mesh().vertex_number() < target_vertices* (1-factor) and count < 5:#not in range(int(target_vertices*(1-factor)), int(target_vertices*(1+factor))):
        count += 1
        remesh_increase(mesh)

    if mesh.current_mesh().vertex_number() > target_vertices* (1+factor):
        mesh.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_vertices*1.4))

    newFilePath = f"./resampled/{classname}/{filename}"

    if not os.path.exists(f"./resampled/{classname}"):
        os.makedirs(f"./resampled/{classname}")

    mesh = o3d_to_pml(transform(flip_mesh(pml_to_o3d(mesh))))
    mesh.save_current_mesh(newFilePath)

    vAfter = mesh.current_mesh().vertex_number()
    fAfter = mesh.current_mesh().face_number()
    print(f"Vertices after: {vAfter}, Faces after: {fAfter}")
    return mesh, vAfter, fAfter, newFilePath
            


def remesh_o3d(mesh, classname, filename, target_vertices=7000, factor=0.2, newfolderpath = "./resampledo3d"):
    print(f"Vertices before: {vertexcount(mesh)}, Faces before: {trianglecount(mesh)}")

    count = 0
    targetlen = 0.02
    while vertexcount(mesh) < target_vertices* (1-factor) and count < 15:#not in range(int(target_vertices*(1-factor)), int(target_vertices*(1+factor))):
        count += 1
        mesh = mesh.subdivide_loop(number_of_iterations=1)
        
    meshcopy = copy.deepcopy(mesh)

    count = 0
    factor2 = 1
    while vertexcount(meshcopy) not in range(int(target_vertices* (1-factor)),int(target_vertices* (1+factor))) and count < 150:
        count += 1
        meshcopy = meshcopy.simplify_quadric_decimation(target_number_of_triangles=int(factor2 * int(target_vertices*(trianglecount(mesh)/vertexcount(mesh))))) #Uses the ratio of Triangles/Vertices to determine the target triangles

        if vertexcount(meshcopy) < target_vertices* (1-factor):
            factor2 *= 1.2
        elif vertexcount(meshcopy) > target_vertices* (1+factor):
            factor2 *= 0.8
        else:
            mesh = meshcopy
        meshcopy = copy.deepcopy(mesh)


def remesh(mesh, classname, filename, target_vertices=7000, factor=0.2, newfolderpath = "./resampled5"):
    print(f"Vertices before: {mesh.current_mesh().vertex_number()}, Faces before: {mesh.current_mesh().face_number()}")

    while mesh.current_mesh().vertex_number() not in range(int(target_vertices* (1-factor)),int(target_vertices* (1+factor))):
        count = 0

        targetlen = 0.02
        while mesh.current_mesh().vertex_number() < target_vertices* (1-factor) and count < 5:#not in range(int(target_vertices*(1-factor)), int(target_vertices*(1+factor))):
            count += 1
            remesh_increase(mesh, targetlen=targetlen)
            targetlen //= 2


        meshcopy = copymesh(mesh)

        count = 0
        factor2 = 1
        maxcount = 15
        while meshcopy.current_mesh().vertex_number() not in range(int(target_vertices* (1-factor)),int(target_vertices* (1+factor))) and count < maxcount:
            count += 1
            meshcopy.meshing_decimation_quadric_edge_collapse(targetfacenum=int(factor2 * int(target_vertices*(mesh.current_mesh().face_number()/mesh.current_mesh().vertex_number())))) #Uses the ratio of Triangles/Vertices to determine the target triangles
            print(f"Target: {int(factor2 * int(target_vertices*(mesh.current_mesh().face_number()/mesh.current_mesh().vertex_number())))}")
            print(f"Result: {meshcopy.current_mesh().vertex_number()}")
            if count == maxcount:
                mesh = meshcopy
            elif meshcopy.current_mesh().vertex_number() < target_vertices* (1-factor):
                factor2 *= 1.1
            elif meshcopy.current_mesh().vertex_number() > target_vertices* (1+factor):
                factor2 *= 0.9
            else:
                mesh = meshcopy

            meshcopy = copymesh(mesh)


def remesh_pml2(mesh, target_vertices=7000):
    print(f"Vertices before: {mesh.current_mesh().vertex_number()}, Faces before: {mesh.current_mesh().face_number()}")

    count = 0
    targetlen = 0.02        
    while mesh.current_mesh().vertex_number() < int(target_vertices) and count < 5:
        count += 1
        targetlen //= 2
        mesh.meshing_isotropic_explicit_remeshing(targetlen=pml.AbsoluteValue(targetlen), iterations=1)
        
    mesh = o3d_to_pml(scale_unitcube(flip_mesh(transform(pml_to_o3d(mesh)))))

    vAfter = mesh.current_mesh().vertex_number()
    fAfter = mesh.current_mesh().face_number()
    print(f"Vertices after: {vAfter}, Faces after: {fAfter}")
    return mesh, vAfter, fAfter


def resample_single_file(mesh, target_vertices=7000):
    if not isinstance(mesh, pml.MeshSet):
        mesh = o3d_to_pml(mesh)
    newmesh, _, _ = remesh_pml2(mesh, target_vertices=target_vertices)
    print("Test")
    return pml_to_o3d(newmesh)