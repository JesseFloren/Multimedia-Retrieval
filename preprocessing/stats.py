import open3d as o3d
import numpy as np
import os
import glob
import pandas as pd
import pymeshlab as pml
import time
import matplotlib as plt

def getstatistics_o3d(dbpath="../resampledo3d/", pkl_path="dataframe_resampledo3d.pkl"):
    mesh_data2 = []
    for class_folder in os.listdir(dbpath):
        # Construct the full path to the subfolder
        class_folder_path = os.path.join(dbpath, class_folder)

        # Check if the item in the directory is a directory (to skip files)
        if os.path.isdir(class_folder_path):
            # Assuming the class folder name is the class name
            class_name = class_folder

            # Iterate through .obj files in the subfolder
            for obj_file_path in glob.glob(os.path.join(class_folder_path, '*.obj')):
                
                mesh = o3d.io.read_triangle_mesh(obj_file_path)
                filename = obj_file_path.split("\\")[-1]
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                filepath = class_folder_path
                mesh_data2.append([class_name, filename, len(vertices), len(triangles), obj_file_path])

                print(f"Class: {class_name}, Vertices: {len(vertices)}, Triangles: {len(triangles)}")
    df2 = pd.DataFrame(mesh_data2, columns=["Class", "Filename", "Vertices", "Triangles", "Filepath"])
    df2.to_pickle(pkl_path)
    return df2


def getstatistics_pml(dbpath="../resampledPML/", pkl_path="dataframe_resampled_pml.pkl"):
    mesh_data2 = []
    for class_folder in os.listdir(dbpath):
        # Construct the full path to the subfolder
        class_folder_path = os.path.join(dbpath, class_folder)

        # Check if the item in the directory is a directory (to skip files)
        if os.path.isdir(class_folder_path):
            # Assuming the class folder name is the class name
            class_name = class_folder

            # Iterate through .obj files in the subfolder
            for obj_file_path in glob.glob(os.path.join(class_folder_path, '*.obj')):                
                # mesh = o3d.io.read_triangle_mesh(obj_file_path)
                mesh = pml.MeshSet()
                mesh.load_new_mesh(obj_file_path)
                filename = obj_file_path.split("\\")[-1]
                # vertices = np.asarray(mesh.vertices)
                # triangles = np.asarray(mesh.triangles)
                filepath = class_folder_path
                mesh_data2.append([class_name, filename, mesh.current_mesh().vertex_number(), mesh.current_mesh().face_number(), obj_file_path])

                print(f"Class: {class_name}, Vertices: {mesh.current_mesh().vertex_number()}, Triangles: {mesh.current_mesh().face_number()}")
    df2 = pd.DataFrame(mesh_data2, columns=["Class", "Filename", "Vertices", "Triangles", "Filepath"])
    df2.to_pickle(pkl_path)
    return df2