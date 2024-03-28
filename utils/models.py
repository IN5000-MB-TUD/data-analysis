from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def train_knn_classifier(X, y, model_path="", max_k=10, cross_validation_size=5):
    """
    Train KNN classifier given.

    :param X: The input data.
    :param y: The related labels.
    :param max_k: The maximum number of neighbors to evaluate.
    :param cross_validation_size: The size of each cross validation element.
    :param model_path: The path where the model needs to be saved.

    :return: The KNN classifier model. None if no path is specified.
    """
    if not model_path:
        return None

    # Check if model exists
    if not Path(model_path).exists():
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Initialize cross validation training
        k_values = [i for i in range(1, max_k)]
        scores = []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, X, y, cv=cross_validation_size)
            scores.append(np.mean(score))

        # Find the best k
        best_k = np.argmax(scores) + 1

        # Save the model with the best k
        knn_model = KNeighborsClassifier(n_neighbors=best_k)
        knn_model.fit(X, y)
        joblib.dump(
            knn_model,
            model_path,
        )
    else:
        knn_model = joblib.load(model_path)

    return knn_model
