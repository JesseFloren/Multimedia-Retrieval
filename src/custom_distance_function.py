import numpy as np
import os
import glob
import standerdization as st
import math


from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean


def load_data(dbpath):
    data = []
    data_path = []
    for class_folder in os.listdir(dbpath):
        class_folder_path = os.path.join(dbpath, class_folder)
        for obj_file_path in glob.glob(os.path.join(class_folder_path, '*')):
            imp = np.load(obj_file_path, allow_pickle=True)
            data.append(imp)
            data_path.append(obj_file_path)
    return data, data_path

def compute_emd_cols(target, matrix, cols, out):
    rows = len(matrix)
    for col in cols:
        for row in range(rows):
            out[row][col] = wasserstein_distance(target[col], matrix[row][col])
    return out

def compute_distance_cols(target, matrix, cols, out):
    rows = len(matrix)
    for col in cols:
        for row in range(rows):
            out[row][col] = abs(target[col] - matrix[row][col])
    return out

def euclidean_average(vector, weights):
    return math.sqrt(sum(weights * vector * vector))

def calculate_distances(data, target, scalar_indexes, hist_indexes, weights):
    res = np.zeros((len(data), len(scalar_indexes + hist_indexes)))
    res = compute_emd_cols(target, data, hist_indexes, res)
    res = compute_distance_cols(target, data, scalar_indexes, res)
    distances = [euclidean_average(row, weights) for row in res]
    return distances

def top_n_closest_objects(data, paths, target, w, n=10):
    scalar_indexes = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20]
    hist_indexes = [5, 6, 7, 8, 9, 10, 11, 12]
    data.append(target)
    data = st.standardize_cols(data, scalar_indexes)
    target = data.pop()
    distances = calculate_distances(data, target, scalar_indexes, hist_indexes, w)
    res = [(distances[i], paths[i]) for i in range(len(distances))]
    res.sort(key=lambda x: x[0])
    if n == None:
        return res
    else:
        return res[:n]

def query_feature_file(target_vector):
    data, paths = load_data(r"./featuresPML2/")
    w = [0.04133703, 0.0244479, 0.03354204, 0.03047129, 0.04051029, 
          0.06507629, 0.04133703, 0.0448802, 0.04051029, 0.03436878, 
          0.12936596, 0.12486482, 0.10507516, 0.03850249, 0.03593279, 
          0.04139966, 0.03251526, 0.0326816, 0.03799905, 0.04065344, 0.03452864]
    return top_n_closest_objects(data, paths, target_vector, w)