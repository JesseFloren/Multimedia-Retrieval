import numpy as np 
import os
import glob
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
 
dbpath = r"./featuresPML2/"
 
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
  
def standardize_cols(matrix, cols):
    rows = len(matrix)
    for col in cols:
        col_total = np.zeros((rows))
        for row in range(rows):
            try:
                col_total[row] = matrix[row][col]
            except:
                print(row)
        std = np.std(col_total)
        mean = np.mean(col_total)
        for row in range(rows):
            matrix[row][col] = (matrix[row][col] - mean) / std
    return matrix
 
def flatten(data):
    new_db = []
    for i in data:
        new = i[0:5]
        for j in i[5:13]:
            new = np.append(new, j)
        new = np.append(new,i[13:])
        new_db.append(new)
    return new_db

def compute_tsne(data):
    scalar_indexes = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20]
    
    data = standardize_cols(data, scalar_indexes)
    db = flatten(data)
  
    test_db = pd.DataFrame(db)
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=450)
    tsne_results = tsne.fit_transform(test_db)
    return tsne_results[:,:2]

def compute_closest(feature_vector):
    data, paths = load_data(dbpath)
    data.append(feature_vector)
    tsne_results = compute_tsne(data)
    target = tsne_results[len(data) - 1]
    tsne_results = tsne_results[:-1]
    distances = [euclidean(target, res) for res in tsne_results]
    result = list(zip(distances, paths))
    result.sort(key=lambda x: x[0])
    return result[0:10]
    