import numpy as np
import os
import glob
import multiprocessing
from tqdm import tqdm

from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance

data = []
data_path = []

dbpath = r"./features/"
for class_folder in os.listdir(dbpath):
    class_folder_path = os.path.join(dbpath, class_folder)
    for obj_file_path in glob.glob(os.path.join(class_folder_path, '*')):
        data.append(np.load(obj_file_path, allow_pickle=True))
        data_path.append(obj_file_path)

def flatten(ndarray):
    new = ndarray[0:7]
    for i in ndarray[7:12]:
        new = np.append(new, i)
    return new

def calculate_distances(target_index, weights1, weights2):
    weights1 = np.asarray(weights1) / sum(weights1)
    weights2 = np.asarray(weights2) / sum(weights2)
    target = data[target_index]
    target_path = data_path[target_index]

    distances = []
    for i in range(len(data)):
        compare = np.array([target[:7], data[i][:7]])
        cosine_distance = pdist(compare, metric='cosine', w=weights1)[0]
        a3_distance = wasserstein_distance(target[7], data[i][7])
        d1_distance = wasserstein_distance(target[8], data[i][8])
        d2_distance = wasserstein_distance(target[9], data[i][9])
        d3_distance = wasserstein_distance(target[10], data[i][10])
        d4_distance = wasserstein_distance(target[11], data[i][11])
        data_tuple = [data_path[i], np.array([cosine_distance, a3_distance, d1_distance, d2_distance, d3_distance, d4_distance])]
        distances.append(data_tuple)

    norm = [max([x[i] for _, x in distances]) for i in range(6)]
    res = []
    for name, x in distances:
        res.append([name, sum((x / norm) * weights2)])
    res.sort(key=lambda a: a[1])
    return res

def speed_calculate_distances(target_index):
    target = data[target_index]

    distances = []
    for i in range(len(data)):
        compare = np.array([target[:7], data[i][:7]])
        a3_distance = wasserstein_distance(target[7], data[i][7])
        d1_distance = wasserstein_distance(target[8], data[i][8])
        d2_distance = wasserstein_distance(target[9], data[i][9])
        d3_distance = wasserstein_distance(target[10], data[i][10])
        d4_distance = wasserstein_distance(target[11], data[i][11])
        data_tuple = [data_path[i], compare, [a3_distance, d1_distance, d2_distance, d3_distance, d4_distance]]
        distances.append(data_tuple)

    return distances

def weight_distances(distances, weights1, weights2):
    new_distances = [[name, np.insert(np.asarray(arr), 0, pdist(comp, metric='cosine', w=weights1)[0])] for name, comp, arr in distances]
    norm = [max([x[i] for _, x in new_distances]) for i in range(6)]
    res = [[name, sum((x / norm) * weights2)] for name, x in new_distances]
    res.sort(key=lambda a: a[1])
    return res

def get_class_val(res, class_name):
    weighting = np.arange(len(res))
    has_jet = np.array([(1 if class_name in n else 0) for n, _ in res])
    jet_val = sum(has_jet * weighting) / len(weighting)
    return jet_val

def get_class_val_top(res, class_name):
    weighting = np.arange(10, 0, -1)
    has_jet = np.array([(1 if class_name in n else 0) for n, _ in res[:10]])
    jet_val = sum(has_jet * weighting) / len(weighting)
    return jet_val

def get_class_indexes(class_name, paths): 
    index = []
    for i in range(len(paths)):
        if class_name in paths[i]:
            index.append(i)
    return (min(index), max(index))

def get_distances(class_name):
    minI, maxI = get_class_indexes(class_name, data_path)
    return [speed_calculate_distances(i) for i in np.random.choice(range(minI, maxI), 3)]

def run_planes(class_name, ds, w1, w2):
    vals = np.asarray([get_class_val(weight_distances(d, w1, w2), class_name) for d in ds])
    return sum(vals) / len(vals)

classes = ["Cup", "Jet", "Bed", "House", "City", "DeskLamp", "Guitar", "Knife", "Wheel", "Rocket", "Sign", "Ship", "Motorcycle", "Humanoid", "Car", "Gun",  "Glasses", "MultiSeat", "AircraftBuoyant",  "Bottle"]

def update_weighting(class_name, n, start_weight1, start_weight2):
    ds = get_distances(class_name)

    prev = run_planes(class_name, ds, start_weight1, start_weight2)
    for i in range(n):
        for i in range(len(start_weight1)):
            start_weight1[i] += 1
            curr = run_planes(class_name, ds, start_weight1, start_weight2)
            if curr < prev:
                prev = curr
            else: 
                start_weight1[i] -= 1
                
        for i in range(len(start_weight2)):
            start_weight2[i] += 1
            curr = run_planes(class_name, ds, start_weight1, start_weight2)
            if curr < prev:
                prev = curr
            else: 
                start_weight2[i] -= 1

    return (start_weight1, start_weight2)


def do_class_weighting(i):
    try:
        return update_weighting(classes[i], 50, [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])
    except:
        print(i)
        exit()


def train_weights():
    res = []
    with multiprocessing.Pool(12) as pool:    
        res = list(tqdm(pool.imap(do_class_weighting, range(len(classes))), total=len(classes)))

    res1 = []
    res2 = []

    for x, y in res:
        res1.append(x)
        res2.append(y)

    res1 = (np.sum(res1, axis=0) / len(res1)).astype(int)
    res2 = (np.sum(res2, axis=0) / len(res2)).astype(int)
    print(res1,res2)

def test_class(c):
    minI, maxI  = get_class_indexes(c, data_path)
    indexes = np.random.choice(range(minI, maxI), 10)
    avgs = []
    for i in indexes:
        dsx = calculate_distances(i, [ 30, 155,  61 ,177 ,113 , 66, 143], [153, 102 , 48 ,106 , 90  ,88])[:10]
        avgs.append(sum([1 if c in i[0] else 0 for i in dsx]) / 10)
    class_avg = sum(avgs) / len(avgs)
    print(c, "avg:", class_avg)
    return class_avg

def test_classes():
    acc = []
    # with multiprocessing.Pool(12) as pool:    
    #     acc = list(pool.imap(test_class, classes))

    for c in classes:
        acc.append(test_class(c))
    print(sum(acc) / len(acc))




if __name__ == '__main__':
    train_weights()
