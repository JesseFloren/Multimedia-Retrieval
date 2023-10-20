import numpy as np
import math
import volume as v

def normalise_distribution(data, bins, max):
    step = max / math.sqrt(len(data))
    curr_step = 0
    norm_hist = []

    for _ in range(bins):
        count = 0
        for d in data:
            if d > curr_step and d < (curr_step + step):
                count += 1
        norm_hist.append(count)
        curr_step += step

    norm_hist = np.asarray(norm_hist) / sum(norm_hist)
    return norm_hist

def calc_mesh_a3(mesh, n):
    vertices = np.asarray(mesh.vertices)
    a3_sample = vertices[np.random.randint(len(vertices), size=(n, 3))]
    a3_dist = []

    for a, b, c in a3_sample:

        if sum(a - b) == 0 or sum(b - c) == 0 or sum(c - a) == 0: 
            continue

        ab = a - b; bc = c - b

        abVec = math.sqrt(sum(ab * ab))
        bcVec = math.sqrt(sum(bc * bc))

        try:        
            res = sum((ab / abVec) * (bc / bcVec))
            a3_dist.append(math.acos(res)*180.0/ math.pi)
        except:
            continue
    return a3_dist

def calc_mesh_d1(mesh, n):
    vertices = np.asarray(mesh.vertices)
    xb, yb, zb = np.mean(vertices, axis=0)
    d1_sample = vertices[np.random.randint(len(vertices), size=(n))]

    d1_dist = []

    for xs, ys, zs in d1_sample:
        d = ((xs - xb)**2 + (ys - yb)**2 + (zs - zb)**2)**0.5
        d1_dist.append(d)

    return d1_dist

def calc_mesh_d2(mesh, n):
    vertices = np.asarray(mesh.vertices)
    d2_sample1 = vertices[np.random.randint(len(vertices), size=(n))]
    d2_sample2 = vertices[np.random.randint(len(vertices), size=(n))]

    d2_dist = []

    for i in range(n):
        xb, yb, zb = d2_sample1[i]
        xs, ys, zs = d2_sample2[i]
        d = ((xs - xb)**2 + (ys - yb)**2 + (zs - zb)**2)**0.5
        d2_dist.append(d)

    return d2_dist

def calc_mesh_d3(mesh, n):
    vertices = np.asarray(mesh.vertices)
    d3_sample1 = vertices[np.random.randint(len(vertices), size=(n))]
    d3_sample2 = vertices[np.random.randint(len(vertices), size=(n))]
    d3_sample3 = vertices[np.random.randint(len(vertices), size=(n))]


    d3_dist = []

    for i in range(n):
        A = d3_sample1[i]
        B = d3_sample2[i]
        Cp = d3_sample3[i]

        Xab, Yab, Zab = A - B
        Xac, Yac, Zac = A - Cp

        d = ((Yab * Zac - Zab * Yac)**2 + (Zab * Xac - Xab * Zac)**2 + (Xab * Yac - Yab * Xac)**2)**0.5
        d3_dist.append(d)

    return np.sqrt(d3_dist)

def calc_mesh_d4(mesh, n):
    vertices = np.asarray(mesh.vertices)
    d4_sample1 = vertices[np.random.randint(len(vertices), size=n)]
    d4_sample2 = vertices[np.random.randint(len(vertices), size=(n))]
    d4_sample3 = vertices[np.random.randint(len(vertices), size=(n))]
    d4_sample4 = vertices[np.random.randint(len(vertices), size=(n))]

    d4_dist = []

    for i in range(n):
        A = d4_sample1[i]
        B = d4_sample2[i]
        Cp = d4_sample3[i]
        Dp = d4_sample4[i]

        ab = A - B
        ac = A - Cp
        ad = A - Dp

        d = abs(v.determinant_3x3((ab,ac,ad)) / 6)
        d4_dist.append(d)

    return np.cbrt(d4_dist)
