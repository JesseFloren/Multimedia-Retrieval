import math
import numpy as np
import open3d as o3d

def calc_compactness(S, V):
    return S**3 / (36 * math.pi * V**2 )
    
def calc_rectangularity(mesh, V):
    obb = mesh.get_oriented_bounding_box()
    Vobb = obb.volume()
    return V/Vobb

def calc_diameter(mesh):
    D = 0
    verts = np.asarray(mesh.vertices)

    for x1, y1, z1 in verts:
        for x2, y2, z2 in verts:
            currD = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
            if D < currD:
                D = currD
    
    return D

def calc_convexity(mesh, V):
    Hull, _ = mesh.compute_convex_hull()
    Vch = Hull.get_volume()
    C = V / Vch
    return C


def calc_eccentricity(mesh):
    verts = np.asarray(mesh.vertices)
    cov = np.cov(verts.transpose())
    eigenvalues, _ = np.linalg.eig(cov)
    E =  min(eigenvalues) / max(eigenvalues)
    return E