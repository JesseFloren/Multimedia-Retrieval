import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance


def w_dist(a,b):
    return wasserstein_distance(a,b)*1000

#computes euclidean distance between 2 vectors
def euclidean_distance(vector1, vector2):
    euclidean = np.linalg.norm(np.asarray(vector1[2:9]) - np.asarray(vector2[2:9]))
    return euclidean

#computes cosine distance between 2 vectors
def cosine_distance(vector1, vector2):
    one_arr = np.asarray(vector1[2:9])
    two_arr = np.asarray(vector2[2:9])
    cosine = 1 - ((one_arr @ two_arr) / (np.linalg.norm(one_arr) * np.linalg.norm(two_arr)))
    return cosine

def get_random_indices(df, num_perclass=2,random_seed=False, print_indices=False):
# Create an empty dictionary to store the random indices
    random_indices_dict = {}
    indices_arr = np.array([],dtype=int)
    # Group the DataFrame by the "class" column
    # df = pd.read_pickle("../dataframe.pkl")
    grouped = df.groupby('class')

    # Iterate over each group and select two different random indices for each
    if random_seed:
        random.seed(random_seed)
    for name, group in grouped:
        indices = group.index.tolist()
        newvals = random.sample(indices, min(num_perclass, len(indices)),)
        random_indices_dict[name] = newvals
        indices_arr = np.append(indices_arr,newvals)
    if print_indices:
        # Display the random indices for each distinct class
        print(random_indices_dict)
        print(indices_arr)
    return indices_arr, random_indices_dict#


def get_top_k_results_allfeatures(newobj, df, k=10, verbose=0):
    w0=5;w1=1;w2=1;w3=1;w4=1;w5=1;w6=5;w7=1;w8=1;w9=1;w10=1;w11=1;w12=1
    weights = np.array([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
    weighted_sums = np.array([])
    for index, row in df.iterrows():
        # print(valdf.columns[0])
        values = np.array([])
        dist = euclidean_distance(newobj,row)
        values = np.append(values, dist)
        values = np.append(values, newobj[2:9]-row[2:9])
        # dist2 = cosine_distance(newobj,row)
        # values = np.append(values, dist2)
            
        for col in df.columns[9:]:
            dist3 = w_dist(newobj[col],row[col])
            # dist = get_emd(newobj[col],row[col])

            values = np.append(values, dist3)
        weighted_sum = np.dot(values, weights)
        weighted_sums = np.append(weighted_sums, weighted_sum)
        # print(weighted_sum)
        lowest_indices = np.argsort(weighted_sums)[:k+1]

        lowest_values = weighted_sums[lowest_indices]
    if verbose==1:
        print("5 Lowest Values:", lowest_values)
        print("Indices of the 5 Lowest Values:", lowest_indices)
    return lowest_indices, lowest_values


def get_top_k_results(newobj, df, k=10):
    valdf = df[['A3', 'D1', 'D2', 'D3', 'D4']]
    w1=1;w2=1;w3=1;w4=1;w5=1
    weights = np.array([w1,w2,w3,w4,w5])
    weighted_sums = np.array([])
    for index, row in valdf.iterrows():
        # print(valdf.columns[0])
        values = np.array([])
        for col in valdf.columns:
            dist = w_dist(newobj[col],row[col])
            # dist = get_emd(newobj[col],row[col])

            values = np.append(values, dist)
        weighted_sum = np.dot(values, weights)
        weighted_sums = np.append(weighted_sums, weighted_sum)
        # print(weighted_sum)
        lowest_indices = np.argsort(weighted_sums)[:k+1]

        lowest_values = weighted_sums[lowest_indices]

    # print("5 Lowest Values:", lowest_values)
    # print("Indices of the 5 Lowest Values:", lowest_indices)
    return lowest_indices, lowest_values




def run_evaluation(df, indices_arr, features="all"):
    totalcorrect = 0
    total = 0
    for i in tqdm(indices_arr):
        newobj = df.iloc()[i]
        if features=="all":
            lowest_indices, lowest_values = get_top_k_results_allfeatures(newobj, df, k = 10, verbose=0)

        else:
            lowest_indices, lowest_values = get_top_k_results(newobj, k = 10)

        y = df.iloc[lowest_indices[0]]["class"]
        correct = 0
        for ind in lowest_indices[1:]:
            if df.iloc[ind]["class"] == y:
                correct += 1 
        totalcorrect += correct
        total += len(lowest_indices[1:])
    print(f"{totalcorrect} out of {total} correct, accuracy: {totalcorrect/total}")
    return totalcorrect, total
try:
    df = pd.read_pickle("../querying/normalized_features_final.pkl")
except:
    df = pd.read_pickle("./querying/normalized_features_final.pkl")
df.reset_index(drop=True, inplace=True)
run_evaluation(df, get_random_indices(df, num_perclass=2,random_seed=7, print_indices=False)[0], features="all")