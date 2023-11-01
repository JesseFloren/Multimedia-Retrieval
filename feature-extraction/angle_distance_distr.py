import numpy as np
import math
import volume as v
import random

def normalise_distribution(data):
    data_max = max(data)
    step = data_max / 100
    bins = [i * step for i in range(101)]
    hist,_ = np.histogram(data,bins=bins)
    return hist / sum(hist)

def calc_mesh_a3(mesh, n):
    vertices = np.asarray(mesh.vertices)
    a3_sample = vertices[unique_random_index3(n, range(len(vertices)))]
    a3_dist = []

    for a, b, c in a3_sample:
        ab = a - b; bc = c - b

        abVec = math.sqrt(sum(ab * ab))
        bcVec = math.sqrt(sum(bc * bc))

        try:
            res = sum((ab / abVec) * (bc / bcVec))
            a3_dist.append(math.acos(res)*180.0/ math.pi)
        except:
            continue

    return a3_dist

def calc_mesh_d1(mesh):
    vertices = np.asarray(mesh.vertices)
    xb, yb, zb = np.mean(vertices, axis=0)

    d1_dist = []

    for xs, ys, zs in vertices:
        d = ((xs - xb)**2 + (ys - yb)**2 + (zs - zb)**2)**0.5
        d1_dist.append(d)

    return d1_dist

def calc_mesh_d2(mesh, n):
    vertices = np.asarray(mesh.vertices)
    d2_sample = vertices[unique_random_index2(n, range(len(vertices)))]

    d2_dist = []

    for i in range(n):

        xb, yb, zb = d2_sample[i][0]
        xs, ys, zs = d2_sample[i][1]

        d = ((xs - xb)**2 + (ys - yb)**2 + (zs - zb)**2)**0.5
        d2_dist.append(d)

    return d2_dist

def calc_mesh_d3(mesh, n):
    vertices = np.asarray(mesh.vertices)
    d3_sample = vertices[unique_random_index3(n, range(len(vertices)))]

    d3_dist = []

    for i in range(n):
        A, B, Cp = d3_sample[i]

        Xab, Yab, Zab = A - B
        Xac, Yac, Zac = A - Cp

        d = ((Yab * Zac - Zab * Yac)**2 + (Zab * Xac - Xab * Zac)**2 + (Xab * Yac - Yab * Xac)**2)**0.5
        d3_dist.append(d)

    return np.sqrt(d3_dist)

def calc_mesh_d4(mesh, n):
    vertices = np.asarray(mesh.vertices)
    d4_sample = vertices[unique_random_index4(n, range(len(vertices)))]

    d4_dist = []

    for i in range(n):
        A, B, Cp, Dp = d4_sample[i]

        ab = A - B
        ac = A - Cp
        ad = A - Dp

        d = abs(v.determinant_3x3((ab,ac,ad)) / 6)
        d4_dist.append(d)

    return np.cbrt(d4_dist)

def give_random_triplets(n, m, input):
    return np.random.choice(input, size=(n, m), replace=True)

def unique_random_index2(n, input):
    random_triplets = give_random_triplets(n, 2, input)
    equal_indices = np.where((random_triplets[:, 0] == random_triplets[:, 1]))
    while equal_indices[0].size > 0:
        random_triplets[equal_indices] = give_random_triplets(random_triplets[equal_indices].shape[0], 2, input)
        equal_indices = np.where((random_triplets[:, 0] == random_triplets[:, 1]))
    return np.asarray(random_triplets)

def unique_random_index3(n, input):
    random_triplets = give_random_triplets(n, 3, input)
    equal_indices = np.where((random_triplets[:, 0] == random_triplets[:, 1]) | (random_triplets[:, 1] == random_triplets[:, 2]) | (random_triplets[:, 0] == random_triplets[:, 2]))
    while equal_indices[0].size > 0:
        random_triplets[equal_indices] = give_random_triplets(random_triplets[equal_indices].shape[0], 3, input)
        equal_indices = np.where((random_triplets[:, 0] == random_triplets[:, 1]) | (random_triplets[:, 1] == random_triplets[:, 2]) | (random_triplets[:, 0] == random_triplets[:, 2]))
    return np.asarray(random_triplets)

def unique_random_index4(n, input):
    random_triplets = give_random_triplets(n, 4, input)
    equal_indices = np.where((random_triplets[:, 0] == random_triplets[:, 1]) | (random_triplets[:, 1] == random_triplets[:, 2]) | (random_triplets[:, 0] == random_triplets[:, 2]) | (random_triplets[:, 0] == random_triplets[:, 3]) | (random_triplets[:, 1] == random_triplets[:, 3]) | (random_triplets[:, 2] == random_triplets[:, 3]))
    while equal_indices[0].size > 0:
        random_triplets[equal_indices] = give_random_triplets(random_triplets[equal_indices].shape[0], 4, input)
        equal_indices = np.where((random_triplets[:, 0] == random_triplets[:, 1]) | (random_triplets[:, 1] == random_triplets[:, 2]) | (random_triplets[:, 0] == random_triplets[:, 2]) | (random_triplets[:, 0] == random_triplets[:, 3]) | (random_triplets[:, 1] == random_triplets[:, 3]) | (random_triplets[:, 2] == random_triplets[:, 3]))
    return np.asarray(random_triplets)