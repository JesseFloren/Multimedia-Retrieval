import numpy as np
import conversions as pp
import pymeshlab as pml
import math

def prepare_mesh_sil(mesh):
    mesh = pp.o3d_to_pml(mesh)
    mesh.meshing_isotropic_explicit_remeshing(targetlen=pml.AbsoluteValue(0.01), iterations=5)
    mesh = pp.pml_to_o3d(mesh)
    return mesh

def get_object_image(mesh):
    vertices = np.asarray(mesh.vertices)

    grid1 = np.zeros((150, 150))
    grid2 = np.zeros((150, 150))
    grid3 = np.zeros((150, 150))

    for x1, y1, z1 in vertices:
        x = round(75 + x1 * 100)
        y = round(75 + y1 * 100)
        z = round(75 + z1 * 100)
        
        if x < 150 and y < 150:
            grid1[x][y] = 1
        if y < 150 and z < 150:
            grid2[y][z] = 1
        if x < 150 and z < 150:
            grid3[x][z] = 1

    return grid1, grid2, grid3


def compress_vector(vector, count):
    for _ in range(count):
        new_vector = []
        for i in range(0, len(vector) - 1, 2):
            new_vector.append((vector[i] + vector[i + 1]) / 2)
        vector = new_vector
    return vector

def Area(matrix):
    x_count, y_count = np.shape(matrix)
    area = 0
    for x in range(x_count):
        for y in range(y_count):
            if matrix[x][y] == 1:
                area += 1
    return area

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def CircularArea(corners):
    max_diam = 0
    for x1, y1 in corners:
        for x2, y2 in corners:
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist > max_diam:
                max_diam = dist
    return math.pi * ((max_diam / 2 + 0.5)**2)  

def GetMinMax(grid):
    x_min = None; x_max = None; y_min = None; y_max = None
    x_of_y_max = None; x_of_y_min = None
    y_of_x_min = None; y_of_x_max = None
    
    x_count, y_count = np.shape(grid)
    for x in range(x_count):
        for y in range(y_count):
            if grid[x][y] == 1:
                if x_min == None:
                    x_min = x; x_max = x; y_min = y; y_max = y
                    x_of_y_max = x; x_of_y_min = x
                    y_of_x_min = y; y_of_x_max = y
                if x_min > x:
                    x_min = x; y_of_x_min = y
                if y_min > y:
                    y_min = y; x_of_y_min = x
                if x_max < x:
                    x_max = x; y_of_x_max = y
                if y_max < y:
                    y_max = y; x_of_y_max = x
    poly_corner = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    circ_corner = [(x_min, y_of_x_min), (x_of_y_max, y_max), (x_max, y_of_x_max), (x_of_y_min, y_min)]
    return poly_corner, circ_corner

def CalcRectCirc(grids, areas):
    rect_circ = []
    for i in range(len(grids)):
        poly_corner, circ_corner = GetMinMax(grids[i])
        poly_area = PolygonArea(poly_corner)
        circ_area = CircularArea(circ_corner)
        rect_circ.append((areas[i] / poly_area, areas[i] / circ_area))
    return rect_circ

def get_silhouette_data(mesh):
    mesh = prepare_mesh_sil(mesh)
    imgs = get_object_image(mesh)
    grid1, grid2, grid3 = [compress_vector(img.flatten(), 7) for img in imgs]
    areas = [Area(i) for i in imgs]
    rc1, rc2, rc3 = CalcRectCirc(imgs, areas)
    areas.sort()
    return grid1, grid2, grid3, areas[0] / areas[2], areas[1] / areas[2], rc1[0], rc2[0], rc3[0], rc1[1], rc2[1], rc3[1]