import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import SVC


def train_and_evaluate_model(X, y, case_id, classifier, n_splits=5):
    """
    Train and evaluate a classifier using stratified k-fold cross-validation.

    Parameters:
    -----------
    X : pandas DataFrame
        Input features.
    y : pandas Series
        Target variable.
    case_id : pandas Series
        Case IDs corresponding to the data.
    classifier : sklearn-compatible classifier
        Machine learning classifier to use.
    n_splits : int, optional
        Number of cross-validation splits (default: 5).

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and case IDs.
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Prepare storage for metrics
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "auc": [],
    }

    # Store predictions, true labels, probabilities, and case IDs
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_case_id = []

    # Prepare DataFrame to store fold-level metrics
    fold_metrics_data = []

    # Scaler for preprocessing
    requires_scaling = isinstance(classifier, SVC)
    scaler = StandardScaler() if requires_scaling else None

    # Cross-validation loop
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        case_id_val = case_id.iloc[val_index]

        if requires_scaling:
            # Scale features if required
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Train classifier
        clf = classifier
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        # Calculate metrics
        accuracy = round(accuracy_score(y_val, y_pred), 3)
        precision = round(precision_score(y_val, y_pred), 3)
        recall = round(recall_score(y_val, y_pred), 3)
        f1 = round(f1_score(y_val, y_pred, average="macro"), 3)
        auc = round(roc_auc_score(y_val, y_prob), 3)

        # Store metrics
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
        metrics["auc"].append(auc)

        # Store fold-level metrics
        fold_metrics_data.append(
            {
                "Fold": fold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "AUC": auc,
            }
        )

        # Store for final plots
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_case_id.extend(case_id_val)

    # Calculate mean metrics
    metrics_mean = {k: np.mean(v) for k, v in metrics.items()}

    return {
        "metrics": metrics_mean,
        "metrics_by_fold": fold_metrics_data,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
        "case_id": all_case_id,
    }
