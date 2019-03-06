import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    print('predictions', prediction)
    print('      truth', ground_truth)
    print('         cw',np.bitwise_xor(prediction, ground_truth))
    tp = np.sum(np.bitwise_and(prediction, ground_truth))
    print('TP=', tp)
    fp = np.sum(np.bitwise_xor(prediction, ground_truth))

    print('FP=', fp)
    precision = tp/(tp + fp)
    print('precision=', precision)
    print('preciiosnn_real', precision_score(ground_truth, prediction))
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    precision = precision_score(ground_truth, prediction)
    recall = recall_score(ground_truth, prediction)
    f1 = f1_score(ground_truth, prediction)
    accuracy = accuracy_score(ground_truth, prediction)

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    return accuracy_score(ground_truth, prediction)
