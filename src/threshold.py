from sklearn.metrics import precision_recall_curve
import numpy as np

def calculate_threshold(model, X, y):
    probabilities = model.predict_proba(X)[:,1]

    precision, recall, thresholds = precision_recall_curve(
            y,
            probabilities
        )

    f1 = (
        (2 *  precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )

    best_index = np.argmax(f1)
    best_threshold = thresholds[best_index]

    return best_threshold