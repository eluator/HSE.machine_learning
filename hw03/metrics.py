import numpy as np

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    loss = np.sum((y_predicted - y_true)**2)
    grad = 2*(y_predicted - y_true)/y_predicted.shape[0];
    
    return loss, grad

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    result = 1- mse(y_true, y_predicted)[0]/np.mean(y_true)
    return result

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
    return sum([int(prediction[i] == ground_truth[i]) for i in range(len(prediction))])/len(prediction);

def r2_accuracy(prediction, ground_truth):
    return sum([r2(prediction[i], ground_truth[i]) for i in range(len(prediction))])/len(prediction);
