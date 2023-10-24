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

def get_feature_vector(mesh):
    st = time.time()
    verticies, triangles = hs.stitch_mesh_holes(mesh)
    V = v.get_mesh_volume(verticies, triangles)
    S = s.get_surface(verticies, triangles)
    c = si.calc_compactness(S, V)
    D = si.calc_diameter(mesh)
    R = si.calc_rectangularity(mesh, V)
    E = si.calc_eccentricity(mesh)
    C = si.calc_convexity(mesh, V)


    A3_data = ad.calc_mesh_a3(mesh, 1000000)
    A3 = ad.normalise_distribution(A3_data, 180)
    D1_data = ad.calc_mesh_d1(mesh, 10000)
    D1 = ad.normalise_distribution(D1_data, np.max(D1_data))
    D2_data = ad.calc_mesh_d2(mesh, 100000)
    D2 = ad.normalise_distribution(D2_data, np.max(D2_data))
    D3_data = ad.calc_mesh_d3(mesh, 1000000)
    D3 = ad.normalise_distribution(D3_data, np.max(D3_data))
    D4_data = ad.calc_mesh_d4(mesh, 1000000)
    D4 = ad.normalise_distribution(D4_data, np.max(D4_data))

    et = time.time()
    elapsed_time = et - st
    print("Elapsed time:", elapsed_time, "seconds")

    return [V, S, c, D, R, E, C, A3, D1, D2, D3, D4]

def generate_feature_file(obj_file_path):
    dbpath = r"./resampledPML/"
    file_path = obj_file_path.replace(dbpath, r"./features/").replace(".obj", "")
    data_file = open(file_path, "wb")

    vect = get_feature_vector(o3d.io.read_triangle_mesh(obj_file_path))
    np.save(data_file, np.asarray(vect, dtype="object"))
    data_file.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.mkdir("./features/" + sys.argv[1])
        file_paths = glob.glob(os.path.join(r"./resampledPML/{}".format(sys.argv[1]) , '*.obj'))
        # with multiprocessing.Pool() as pool: 
        #     pool.map(generate_feature_file, file_paths) 
        
        for file in file_paths:
            generate_feature_file(file)
    else:
        dbpath = r"./resampledPML/"
        for class_folder in os.listdir(dbpath):
            class_folder_path = os.path.join(dbpath, class_folder)
            if os.path.isdir(class_folder_path):
                class_name = class_folder
                try:
                    os.mkdir("./features/" + class_name)
                except:
                    continue
                file_paths = glob.glob(os.path.join(class_folder_path, '*.obj'))

                with multiprocessing.Pool() as pool: 
                    pool.map(generate_feature_file, file_paths) 
