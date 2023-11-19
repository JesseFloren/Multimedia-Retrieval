import csv
import statistics as stats
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import os
import glob
from sklearn.metrics import auc

def get_database_properties(features_path):
    classes = []
    class_sizes = {}
    filepaths = []
    with open(features_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            filepath = row[0]
            filepaths.append(filepath)
            label = row.pop(1)
            if label not in classes:
                classes.append(label)
            row.pop(0)
            no_features = len(row)
            if label in class_sizes.keys():
                class_sizes[label] += 1
            else:
                class_sizes[label] = 1

    return no_features, len(classes), classes, filepaths, class_sizes


def get_query_results(results_path, class_sizes):
    results = {}
    ground_truth = {}
    with open(results_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for i, row in enumerate(csv_reader):
            row_items = row[0].split(",")
            query_path = row_items.pop(0)
            query_class = row_items.pop(0)
            k = len(row_items)
            results[query_path] = [result_class for result_class in row_items]
            ground_truth[query_path] = [query_class for _ in range(k)]
    
    return results, ground_truth


def append_to_dict(_dict, label, value):
    if not (label in _dict.keys()):
        _dict[label] = [value]
    else:
        _dict[label].append(value)


def plot_perclass_metrics(data_dict, no_classes, title):
    # Plot histogram
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    items = [(label, value) for label, value in zip(labels, values)]
    items.sort(key=lambda x: x[1])
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    plt.bar(labels, values, color ='maroon', 
        width = 0.7)
    plt.xlabel("Classes")
    plt.xticks(rotation=90)
    plt.ylabel(title)
    plt.title(f"{title} per class")
    plt.show()

def most_common_class(result):
    classes = []
    for _, p in result:
        classes.append(p.replace("\\", "/").split("/")[2])
    count = Counter(classes)
    return count.most_common()[0][0]

def load_data(path):
    data = []
    for file_path in glob.glob(os.path.join(path, '*.npy')):
        data.append(np.load(file_path, allow_pickle=True))
    return np.asarray(data)

def eval_for_path(input, treshold):
    ground_truth = [d[0][1].replace("\\", "/").split("/")[2] for d in input]
    
    data = []
    for query in input:
        query_dists = []
        for distance in query:
            if distance[0] < treshold:
                query_dists.append(distance)
        data.append(query_dists)

    query_results = []
    for d in data:
        query = []
        for _, p in d:
            query.append(p.replace("\\", "/").split("/")[2])
        query_results.append(query)

    sensitivities = {}
    specificities = {}
    database_size = len(ground_truth)
    for i, [y_pred, y_true] in enumerate(zip(query_results, ground_truth)):
        query_class = y_true
        objects_in_class = ground_truth.count(query_class)

        TP = y_pred.count(query_class)
        FP = len(y_pred) - TP
        FN = objects_in_class - TP
        TN = database_size - len(y_pred) - FN

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)


        append_to_dict(sensitivities, query_class, sensitivity)
        append_to_dict(specificities, query_class, specificity)
    

    perclass_sensitivities = {label : stats.mean(class_sensitivities) for label, class_sensitivities in sensitivities.items()}
    perclass_specificities = {label : stats.mean(class_specificities) for label, class_specificities in specificities.items()}

  
    overall_sensitivity = stats.mean([sensitivity for sensitivity in perclass_sensitivities.values()])
    overall_specificity = stats.mean([specificity for specificity in perclass_specificities.values()])
    return overall_sensitivity, overall_specificity
    

def main():
    data = load_data(r"./extra-code/evaluation/output") 
    print("Loaded data")
    steps = 150
    max = 0
    for query in data:
        for distance in query:
            if distance[0] > max:
                max = distance[0]
    step = 0
    roc = []
    treshold = 0
    while treshold <= max:
        step += 1
        print(step)
        roc.append(eval_for_path(data, treshold))
        treshold += (max / steps)

    roc = np.asarray(roc)

    x_ = roc[:, 0]
    y_ = roc[:, 1]
    print("AUROC:", auc(x_, y_))
    plt.plot(x_, y_)
    plt.xlabel("Sensitivity")
    plt.ylabel("Specificity")
    plt.show()


if __name__ == "__main__":
    main()
