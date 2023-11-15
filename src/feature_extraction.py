import angle_distance_distr as ad
import numpy as np
import time

import surface as s
import simple as si
import silhouette as sil

def get_feature_vector(mesh):
    st = time.time()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    S = s.get_surface(vertices, triangles)
    D = si.calc_diameter(mesh)
    E = si.calc_eccentricity(mesh)

    # NEW
    Vobb = si.calc_OBB_volume(mesh)
    Vch = si.calc_convex_hull_volume(mesh)

    A3_data = ad.calc_mesh_a3(mesh, 1000000)
    A3 = ad.normalise_distribution(A3_data)
    D1_data = ad.calc_mesh_d1(mesh)
    D1 = ad.normalise_distribution(D1_data)
    D2_data = ad.calc_mesh_d2(mesh, 100000)
    D2 = ad.normalise_distribution(D2_data)
    D3_data = ad.calc_mesh_d3(mesh, 1000000)
    D3 = ad.normalise_distribution(D3_data)
    D4_data = ad.calc_mesh_d4(mesh, 1000000)
    D4 = ad.normalise_distribution(D4_data)
    SIL1, SIL2, SIL3, R02, R12, RE1, RE2, RE3, CI1, CI2, CI3 = sil.get_silhouette_data(mesh)

    et = time.time()
    elapsed_time = et - st
    print("Elapsed time:", elapsed_time, "seconds")

    return [S, D, Vobb, E, Vch, A3, D1, D2, D3, D4, SIL1, SIL2, SIL3, R02, R12, RE1, RE2, RE3, CI1, CI2, CI3]
