import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import os


def plot_roc_curve(results, save_path, classifier_name):
    """
    Plot ROC curves and save the figure.

    Parameters:
    -----------
    results : dict
        Dictionary containing results from model evaluation.
    save_path : str
        Directory to save the plot.
    classifier_name : str
        Name of the classifier for file naming.

    Returns:
    --------
    fold_aucs : list
        AUC scores for each fold.
    mean_auc : float
        Mean AUC score.
    """
    plt.figure(figsize=(10, 8))

    # Color palette for fold lines
    fold_colors = ["#B0C4DE", "#E6E6FA", "#D3D3D3", "#A9A9A9", "#778899"]

    # Compute and plot individual fold ROC curves
    fold_size = len(results["y_true"]) // 5
    fold_aucs = []

    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size

        fold_y_true = results["y_true"][start:end]
        fold_y_prob = results["y_prob"][start:end]

        fpr, tpr, _ = roc_curve(fold_y_true, fold_y_prob)
        fold_auc = roc_auc_score(fold_y_true, fold_y_prob)
        fold_aucs.append(fold_auc)

        plt.plot(
            fpr,
            tpr,
            color=fold_colors[i],
            label=f"Fold {i+1} (AUC = {fold_auc:.3f})",
            alpha=0.6,
        )

    # Compute and plot mean ROC curve
    mean_fpr = np.arange(0, 1.01, 0.01)
    mean_tpr = np.zeros_like(mean_fpr)

    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size

        fold_y_true = results["y_true"][start:end]
        fold_y_prob = results["y_prob"][start:end]

        fpr, tpr, _ = roc_curve(fold_y_true, fold_y_prob)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    mean_tpr /= 5
    mean_auc = np.mean(fold_aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="#1E90FF",
        label=f"Mean ROC (AUC = {mean_auc:.3f})",
        linewidth=2.5,
    )

    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plot_file = os.path.join(save_path, f"{classifier_name}_roc_curve.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"ROC Curve saved to: {plot_file}")

    return fold_aucs, mean_auc


def plot_test_roc_curve(results, save_path, classifier_name):

    plt.figure(figsize=(10, 8))

    y_true = results["y_true"]
    y_prob = results["y_prob"]

    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.plot(
        fpr,
        tpr,
        color="#1E90FF",
        label=f"ROC (AUC = {auc:.3f})",
        linewidth=2.5,
    )

    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plot_file = os.path.join(save_path, f"{classifier_name}_roc_curve.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"ROC Curve saved to: {plot_file}")

    return None, auc


def plot_confusion_matrix(results, save_path, classifier_name):
    """
    Plot confusion matrix and save the figure.

    Parameters:
    -----------
    results : dict
        Dictionary containing results from model evaluation.
    save_path : str
        Directory to save the plot.
    classifier_name : str
        Name of the classifier for file naming.
    """
    plt.figure(figsize=(8, 6))

    cm = confusion_matrix(results["y_true"], results["y_pred"])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plot_file = os.path.join(save_path, f"{classifier_name}_confusion_matrix.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"Confusion Matrix plot saved to: {plot_file}")

    return cm
