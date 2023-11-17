import numpy as np
import os
import glob
import multiprocessing
from tqdm import tqdm
import standerdization as st
import math
from collections import Counter
import cv2 as cv
from scipy.spatial import distance

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

def compute_scalar_cols(target, matrix, cols, w1, out):
    weights = [w1[c] for c in cols]
    rows = len(matrix)
    for row in range(rows):
        target_vec = []
        row_vec = []
        for col in cols:
            target_vec.append(target[col])
            row_vec.append(matrix[row][col])
        out[row][0] = euclidean(target_vec, row_vec, weights)
    return out

def compute_euc_cols(target, matrix, cols, w1, out):
    weights = np.asarray([w1[c] for c in cols])
    rows = len(matrix)
    for row in range(rows):
        euc_out = np.zeros((len(cols)))
        for i in range(len(cols)):
            euc_out[i] = euclidean(target[cols[i]], matrix[row][cols[i]])
        out[row][1] = sum(euc_out * weights)
    return out

def compute_cos_cols(target, matrix, cols, w1, out):
    weights = np.asarray([w1[c] for c in cols])
    rows = len(matrix)
    for row in range(rows):
        euc_out = np.zeros((len(cols)))
        for i in range(len(cols)):
            euc_out[i] = distance.cosine(target[cols[i]], matrix[row][cols[i]])
        out[row][2] = sum(euc_out * weights)
    return out

def compute_emd_cols(target, matrix, cols, w1, out):
    weights = np.asarray([w1[c] for c in cols])
    rows = len(matrix)
    for row in range(rows):
        euc_out = np.zeros((len(cols)))
        for i in range(len(cols)):
            euc_out[i] = wasserstein_distance(target[cols[i]], matrix[row][cols[i]])
        out[row][3] = sum(euc_out * weights)
    return out

def normalize_distance(res):
    (_, cols) = np.shape(res)
    max_vals = np.asarray([np.max(res[:, i]) for i in range(cols)])
    max_vals[max_vals == 0] = 1
    return np.divide(res, max_vals)


def calculate_distances(data, target, scalar_indexes, euc_indexes, cos_indexes, emd_indexes, w1, w2):
    res = np.zeros((len(data), 4))
    try:
        res = compute_scalar_cols(target, data, scalar_indexes, w1, res)
        res = compute_euc_cols(target, data, euc_indexes, w1, res)
        res = compute_cos_cols(target, data, cos_indexes, w1, res)
        res = compute_emd_cols(target, data, emd_indexes, w1, res)
        res = normalize_distance(res)
        distances = [sum(row * w2) for row in res]
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    return distances

def top_n_closest_objects(data, paths, target, w1, w2, n=10):
    scalar_indexes = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20]
    euc_indexes = []
    cos_indexes = []
    emd_indexes = [5, 6, 7, 8, 9, 10, 11, 12]
    data.append(target)
    data = st.standardize_cols(data, scalar_indexes)
    target = data.pop()
    distances = calculate_distances(data, target, scalar_indexes, euc_indexes, cos_indexes, emd_indexes, w1, w2)
    res = [(distances[i], paths[i]) for i in range(len(distances))]
    res.sort(key=lambda x: x[0])
    if n == None:
        return res
    else:
        return res[:n]


def query_feature_file(target_vector):
    data, paths = load_data(r"./featuresPML2/")

    w1 = np.asarray([0.0507, 0.0300, 0.0412, 0.0374, 0.0497,
                     0.1308, 0.0750, 0.1326, 0.1038, 0.0752,
                     0.1676, 0.1687, 0.1578, 0.0472, 0.0441,
                     0.0508, 0.0399, 0.0401, 0.0466, 0.0499, 0.0424]) - 0.025
    w2 = np.asarray([0.3621, 0.1992, 0, 0.3095]) - 0.025

    return top_n_closest_objects(data, paths, target_vector, w1, w2)