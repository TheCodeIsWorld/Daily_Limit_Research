import numpy as np
import math
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error

def evaluate(prediction, ground_truth):
    assert ground_truth.shape == prediction.shape, 'shape mismatch'
    performace = {}
    performace['mse'] = mean_squared_error(np.squeeze(ground_truth), np.squeeze(prediction))

    pred = np.round(prediction)

    try:
        performace['acc'] = accuracy_score(ground_truth, pred)
    except Exception:
        np.savetxt('prediction', pred, delimiter=',')
        exit(0)
    # performace['mcc'] = matthews_corrcoef(ground_truth, pred)
    return performace
