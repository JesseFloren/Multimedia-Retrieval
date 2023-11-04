import sklearn.metrics as metrics
import numpy as np
import csv
import statistics as stats
from matplotlib import pyplot as plt


def get_query_results(results_path):
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


def main():
    query_results_path = "./Evaluation/data/query_results.csv"
    no_classes = 47
    query_results, ground_truth = get_query_results(query_results_path)

    # Compute metrics
    precisions = {}
    recalls = {}
    f1_scores = {}
    accuracies = {}
    sensitivities = {}
    specificities = {}
    database_size = len(query_results)
    for i, [y_pred, y_true] in enumerate(zip(list(query_results.values()), list(ground_truth.values()))):
        query_class = y_true[0]
        query_size = len(y_true)

        # Get true positives/negatives and false positives/negatives
        TP = y_pred.count(query_class)          # Correctly labelled as 'member of query class' 
        FP = query_size - TP                    # Incorrectly labelled as 'member of query class' (i.e. all returned shapes that are not part of the query class)
        TN = database_size - query_size - FP    # Correctly labelled as 'NOT a member of query class' (i.e. all shapes in the database not part of query class that were not returned)
        FN = query_size - TP                    # Incorrectly labelled as 'NOT a member of query class' (i.e. all shapes in the database that are
                                                # a part of the query class but were not returned)

        # Compute performance metrics
        precision = TP / query_size
        recall = TP / FN
        accuracy = (TP + TN) / database_size
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * recall) / (precision + recall))
        
        # Store performance metric results
        append_to_dict(precisions, query_class, precision)
        append_to_dict(recalls, query_class, recall)
        append_to_dict(f1_scores, query_class, f1_score)
        append_to_dict(accuracies, query_class, accuracy)
        append_to_dict(sensitivities, query_class, sensitivity)
        append_to_dict(specificities, query_class, specificity)
    

    # Aggregate performance metrics for each class
    perclass_precisions = {label : stats.mean(class_precisions) for label, class_precisions in precisions.items()}
    perclass_recalls = {label : stats.mean(class_recalls) for label, class_recalls in recalls.items()}
    perclass_f1_scores = {label : stats.mean(class_f1_scores) for label, class_f1_scores in f1_scores.items()}
    perclass_accuracies = {label : stats.mean(class_accuracies) for label, class_accuracies in accuracies.items()}
    perclass_sensitivities = {label : stats.mean(class_sensitivities) for label, class_sensitivities in sensitivities.items()}
    perclass_specificities = {label : stats.mean(class_specificities) for label, class_specificities in specificities.items()}
    # print("\nper-class mean precisions: ", perclass_precisions)
    # print("\nper-class mean recalls: ", perclass_recalls)
    # print("\nper-class mean f1 scores: ", perclass_f1_scores)
    # print("\nper-class mean accuracies: ", perclass_accuracies)
    # print("\nper-class mean sensitivities: ", perclass_sensitivities)
    # print("\nper-class mean specificities: ", perclass_specificities)

    # Aggregate performance metrics across entire database
    overall_precision = stats.mean(list(perclass_precisions.values()))
    overall_recall = stats.mean(list(perclass_recalls.values()))
    overall_f1_score = stats.mean(list(perclass_f1_scores.values()))
    overall_accuracy = stats.mean(list(perclass_accuracies.values()))
    overall_sensitivity = stats.mean(list(perclass_sensitivities.values()))
    overall_specificity = stats.mean(list(perclass_specificities.values()))
    print("\n" + "-"*30 + "\nOVERALL PERFORMANCE\n" + "-"*30)
    print("Overall precision: ", overall_precision)
    print("Overall recall: ", overall_recall)
    print("Overall F1 score: ", overall_f1_score)
    print("Overall accuracy: ", overall_accuracy)
    print("Overall sensitivity: ", overall_sensitivity)
    print("Overall specificity: ", overall_specificity)

    # Plot per-class performance metrics
    plot_type = "precision"
    plot_perclass_metrics(perclass_precisions, no_classes, "precision") if plot_type == "precision" else None
    plot_perclass_metrics(perclass_recalls, no_classes, "recall") if plot_type == "recall" else None
    plot_perclass_metrics(perclass_f1_scores, no_classes, "F1 score") if plot_type == "F1 score" else None
    plot_perclass_metrics(perclass_accuracies, no_classes, "accuracy") if plot_type == "accuracy" else None
    plot_perclass_metrics(perclass_sensitivities, no_classes, "sensitivity") if plot_type == "sensitivity" else None
    plot_perclass_metrics(perclass_specificities, no_classes, "specificity") if plot_type == "specificity" else None


if __name__ == "__main__":
    main()
