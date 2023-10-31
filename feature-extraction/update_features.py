import multiprocessing 
import sys

import angle_distance_distr as ad
import hole_stitching as hs
import open3d as o3d
import numpy as np
import time
import os
import glob

import volume as v
import surface as s
import simple as si

def get_feature_vector(mesh, prev, features):
    st = time.time()
    
    updated = ""

    if 0 in features:
        verticies, triangles = hs.stitch_mesh_holes(mesh)
        prev[0] = v.get_mesh_volume(verticies, triangles)
        prev[1] = s.get_surface(verticies, triangles)
        prev[2] = si.calc_compactness(prev[1], prev[0])
        prev[3] = si.calc_diameter(mesh)
        prev[4] = si.calc_rectangularity(mesh, prev[0])
        prev[5] = si.calc_eccentricity(mesh)
        prev[6] = si.calc_convexity(mesh, prev[0])
        updated += "Scalar, "
    
    if 1 in features:
        A3_data = ad.calc_mesh_a3(mesh, 1000000)
        prev[7] = ad.normalise_distribution(A3_data)
        updated += "A3, "
    if 2 in features:
        D1_data = ad.calc_mesh_d1(mesh, 10000)
        prev[8] = ad.normalise_distribution(D1_data)
        updated += "D1, "
    if 3 in features:
        D2_data = ad.calc_mesh_d2(mesh, 100000)
        prev[9] = ad.normalise_distribution(D2_data)
        updated += "D2, "
    if 4 in features:
        D3_data = ad.calc_mesh_d3(mesh, 1000000)
        prev[10] = ad.normalise_distribution(D3_data)
        updated += "D3, "
    if 5 in features:
        D4_data = ad.calc_mesh_d4(mesh, 1000000)
        prev[11] = ad.normalise_distribution(D4_data)
        updated += "D4, "

    et = time.time()
    elapsed_time = et - st
    print("Updated features:", updated, "in", elapsed_time, "seconds")

    return prev

def update_feature_file(obj_file_path):
    dbpath = r"./resampledPML/"
    file_path = obj_file_path.replace(dbpath, r"./features/").replace(".obj", "")
    prev = np.load(file_path, allow_pickle=True)
    vect = get_feature_vector(o3d.io.read_triangle_mesh(obj_file_path), prev, [0])
    data_file = open(file_path, "wb")
    np.save(data_file, np.asarray(vect, dtype="object"))
    data_file.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_paths = glob.glob(os.path.join(r"./resampledPML/{}".format(sys.argv[1]) , '*.obj'))
        with multiprocessing.Pool() as pool: 
            pool.map(update_feature_file, file_paths) 
    else:
        dbpath = r"./resampledPML/"
        for class_folder in os.listdir(dbpath):
            class_folder_path = os.path.join(dbpath, class_folder)
            if os.path.isdir(class_folder_path):
                class_name = class_folder
                file_paths = glob.glob(os.path.join(class_folder_path, '*.obj'))

                with multiprocessing.Pool() as pool: 
                    pool.map(update_feature_file, file_paths) 
