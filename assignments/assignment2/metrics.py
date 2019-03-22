import numpy as np
def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    return np.sum(prediction == ground_truth)/prediction.shape[0]

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = np.sum((prediction==ground_truth) & (prediction==True))
    FP = np.sum((prediction!=ground_truth) & (prediction==True))
    FN = np.sum((prediction!=ground_truth) & (prediction==False))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = np.sum(prediction==ground_truth)/prediction.shape[0]
    f1 = 2*(precision*recall)/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy
