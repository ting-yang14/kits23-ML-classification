import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


def train_and_save_model(X, y, classifier, save_path, classifier_name):
    """
    Train the final model using all available data and save it to disk.

    Parameters:
    -----------
    X : pandas DataFrame
        Input features.
    y : pandas Series
        Target variable.
    classifier : sklearn-compatible classifier
        The selected classifier to train.
    save_path : str
        Path to save the trained model.
    classifier_name : str
        Name of the model file to be saved.

    Returns:
    --------
    None
    """
    requires_scaling = isinstance(classifier, SVC)
    scaler = StandardScaler() if requires_scaling else None

    if requires_scaling:
        X = scaler.fit_transform(X)

    classifier.fit(X, y)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    model_path = os.path.join(save_path, f"{classifier_name}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(classifier, file)

    if requires_scaling:
        scaler_path = os.path.join(save_path, f"{classifier_name}_scaler.pkl")
        with open(scaler_path, "wb") as file:
            pickle.dump(scaler, file)

    print(f"Model saved to: {model_path}")


def save_evaluation_metrics(results, save_path, classifier_name):
    """
    Save evaluation metrics to multiple CSV files.

    Parameters:
    -----------
    results : dict
        Dictionary containing evaluation metrics and case IDs.
    save_path : str
        Base path for saving CSV files.
    classifier_name : str
        Name of the classifier for file naming.
    """
    metrics_file = os.path.join(save_path, f"{classifier_name}_overall_metrics.csv")
    metrics_df = pd.DataFrame.from_dict(
        results["metrics"], orient="index", columns=["Value"]
    )
    metrics_df.index.name = "Metric"
    metrics_df.to_csv(metrics_file)
    if results.get("metrics_by_fold"):
        fold_metrics_file = os.path.join(
            save_path, f"{classifier_name}_fold_metrics.csv"
        )
        fold_metrics_df = pd.DataFrame(results["metrics_by_fold"])
        fold_metrics_df.to_csv(fold_metrics_file, index=False)

    roc_data_file = os.path.join(save_path, f"{classifier_name}_roc_data.csv")
    roc_data = pd.DataFrame(
        {
            "case_id": results["case_id"],
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
            "y_prob": results["y_prob"],
        }
    )
    roc_data.to_csv(roc_data_file, index=False)

    print(f"Metrics saved:")
    print(f"- Overall metrics: {metrics_file}")
    if results.get("metrics_by_fold"):
        print(f"- Fold-level metrics: {fold_metrics_file}")
    print(f"- ROC data: {roc_data_file}")


def save_roc_details(results, save_path, classifier_name, mean_auc, fold_aucs=None):
    """
    Save ROC AUC details to a CSV with case IDs.

    Parameters:
    -----------
    fold_aucs : list
        AUC scores for each fold.
    mean_auc : float
        Mean AUC score.
    results : dict
        Dictionary containing case IDs and other results.
    save_path : str
        Directory to save the CSV.
    classifier_name : str
        Name of the classifier for file naming.
    """
    if fold_aucs is None:
        roc_df = pd.DataFrame(
            {
                "AUC": mean_auc,
                "Case_IDs": [",".join(results["case_id"])],
            }
        )
    else:
        fold_size = len(results["y_true"]) // 5
        fold_case_ids = [
            results["case_id"][i * fold_size : (i + 1) * fold_size] for i in range(5)
        ]

        roc_df = pd.DataFrame(
            {
                "Fold": [f"Fold {i+1}" for i in range(len(fold_aucs))],
                "AUC": fold_aucs,
                "Case_IDs": [",".join(map(str, ids)) for ids in fold_case_ids],
            }
        )

        roc_df = pd.concat(
            [
                roc_df,
                pd.DataFrame([{"Fold": "Mean", "AUC": mean_auc, "Case_IDs": "N/A"}]),
            ],
            ignore_index=True,
        )

    roc_file = os.path.join(save_path, f"{classifier_name}_roc_details.csv")
    roc_df.to_csv(roc_file, index=False)
    print(f"ROC AUC details saved to: {roc_file}")


def get_classifier(classifier_name, params=None):
    """
    Return the specified classifier with optional parameters.

    Parameters:
    -----------
    classifier_name : str
        Name of the classifier to use.
    params : dict, optional
        Parameters for the classifier.

    Returns:
    --------
    sklearn classifier
        Initialized classifier.
    """
    if params is None:
        params = {}

    if classifier_name == "RF":
        return RandomForestClassifier(random_state=42, **params)
    elif classifier_name == "SVM":
        default_params = {"kernel": "rbf", "probability": True, "random_state": 42}

        # Only use defaults for keys not in params
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

        return SVC(**params)
    elif classifier_name == "XGB":
        return xgb.XGBClassifier(random_state=42, **params)
    else:
        raise ValueError(f"Invalid classifier. Choose from RF, SVM, XGB")
