import numpy as np

def get_mesh_volume(verticies, triangles):
    V = 0
    for t in triangles:
        o = np.mean(verticies, axis=0)
        V += get_tetahedron_volume(verticies[t], o)
    return abs(V)

def get_tetahedron_volume(triangle, barycenter):
    a, b, c = triangle
    ab = a - barycenter
    ac = b - barycenter
    ad = c - barycenter
    return determinant_3x3((ab,ac,ad)) / 6

def determinant_3x3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))