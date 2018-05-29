# -*- coding: utf-8 -*-
"""score.py

- AP        # Average Precision of each class
- mean AP   # Mean Average Precision
"""
import numpy as np
from sklearn.metrics import average_precision_score


def AP(scores, labels):
    """Calculate the Average Precision of each class

    Arguments:
        scores {np.array} -- Score matrix (n_samples * n_classes)
        labels {np.array} -- Label vector (n_samples)

    Returns:
        [float] -- AP of each sample (0 ≤ AP ≤ 1)

    Example:
    >>> scores = np.array([             # Score matrix
            [0.2, 0.4, 0.8],            # Scores of the first sample over three classes
            [0.9, 0.2, 0.4],            # Scores of the second sample over three classes
            [0.5, 0.4, 0.3]             # Scores of the third sample over three classes
        ])
    >>> labels = np.array([2, 0, 1])    # Corresponding labels
    >>> AP(scores, labels)
    [1.0, 0.5, 1.0]
    """
    n_samples, n_classes = scores.shape
    APscores = [0] * n_classes
    for i in range(n_classes):
        y_test = np.zeros(shape=n_samples, dtype=np.int)
        y_score = scores[:, i]
        for j in range(n_samples):
            y_test[j] = 1 if labels[j] == i else 0
        APscores[i] = average_precision_score(y_test, y_score)
    return APscores


def meanAP(scores, labels):
    """Calculate the Mean Average Precision of each class

    Arguments:
        scores {np.array} -- Score matrix (n_samples * n_classes)
        labels {np.array} -- Label vector (n_samples)

    Returns:
        float -- Mean Average Precision (0 ≤ AP ≤ 1)

    Example:
    >>> scores = np.array([             # Score matrix
            [0.2, 0.4, 0.8],            # Scores of the first sample over three classes
            [0.9, 0.2, 0.4],            # Scores of the second sample over three classes
            [0.5, 0.4, 0.3]             # Scores of the third sample over three classes
        ])
    >>> labels = np.array([2, 0, 1])    # Corresponding labels
    >>> meanAP(scores, labels)
    0.8333333333333333
    """
    n_samples, n_classes = scores.shape
    y_test = np.zeros(shape=scores.shape, dtype=np.int)
    for i in range(n_samples):
        y_test[i, int(labels[i])] = 1
    return average_precision_score(y_test, scores, average="micro")
