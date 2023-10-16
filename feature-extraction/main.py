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
    A3 = ad.normalise_distribution(ad.calc_mesh_a3(mesh, 1000000), 1000, 180)
    D1 = ad.normalise_distribution(ad.calc_mesh_d1(mesh, 1000000), 1000, 1)
    D2 = ad.normalise_distribution(ad.calc_mesh_d2(mesh, 1000000), 1000, 1)
    D3 = ad.normalise_distribution(ad.calc_mesh_d3(mesh, 1000000), 1000, 1)
    D4 = ad.normalise_distribution(ad.calc_mesh_d4(mesh, 1000000), 1000, 1)

    et = time.time()
    elapsed_time = et - st
    print("Elapsed time:", elapsed_time, "seconds")

    return [V, S, c, D, R, E, C, A3, D1, D2, D3, D4]


if __name__ == "__main__":
    dbpath = r"./resampled3/"
    for class_folder in os.listdir(dbpath):
        class_folder_path = os.path.join(dbpath, class_folder)
        if os.path.isdir(class_folder_path):
            class_name = class_folder
            try:
                os.mkdir("./features/" + class_name)
            except:
                continue
            for obj_file_path in glob.glob(os.path.join(class_folder_path, '*.obj')):
                try:
                    vect = get_feature_vector(o3d.io.read_triangle_mesh(obj_file_path))
                    data_file = open(obj_file_path.replace(dbpath, r"./features/").replace(".obj", ""), "wb")
                    np.save(data_file, np.asarray(vect, dtype="object"))
                    data_file.close()
                except:
                    continue