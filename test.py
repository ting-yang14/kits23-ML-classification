from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def test_model(X_test, y_test, case_id_test, classifier, scaler=None):
    """
    Test a trained model on the test dataset.

    Args:
        X_test: Test features
        y_test: Test labels
        case_id_test: Test case IDs
        classifier: Trained classifier
        scaler: Trained scaler (for SVM)

    Returns:
        Dictionary with test results
    """
    # Apply scaler if provided (for SVM)
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    # Make predictions
    y_pred = classifier.predict(X_test_scaled)
    y_prob = classifier.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_prob)

    # Return results dictionary
    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
        },
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "case_id": case_id_test,
    }
