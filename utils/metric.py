import numpy as np


def rmspe(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
