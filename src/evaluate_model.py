from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

def evaluate_model(model, X, y, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1] # Fraud proability

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(
        "Accuracy:",
        accuracy_score(y, predictions)
    )

    print(
            "Precision:",
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        )
    
    print(
        "Recall:",
        recall_score(
            y,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            y,
            predictions,
            zero_division=0
        )
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y,
            probabilities
        )
    )

    print(
        "PR-AUC:",
        average_precision_score(
            y,
            probabilities
        )
    )
    
    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y,
            predictions
        )
    )