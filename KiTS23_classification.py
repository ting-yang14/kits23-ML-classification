import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    roc_auc_score,
    confusion_matrix
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

def preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Preprocess the input CSV file for machine learning classification.
    
    Parameters:
    -----------
    file_path : str
        Path to the input CSV file.
    test_size : float, optional, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    
    Returns:
    --------
    X_train : numpy array
        Training features.
    X_test : numpy array
        Testing features.
    y_train : numpy array
        Training labels.
    y_test : numpy array
        Testing labels.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Drop specified columns
    df = df.drop(columns=['case_id', 'tumor_histologic_subtype'])
    
    # Drop NaN values
    df = df.dropna()
    
    # Convert gender to binary
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    
    # Convert malignant to binary target
    df['malignant'] = df['malignant'].map({True: 1, False: 0})
    
    # Separate features and target
    X = df.drop(columns=['malignant'])
    y = df['malignant']
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X, y, classifier, n_splits=5):
    """
    Train and evaluate a classifier using stratified k-fold cross-validation
    
    Parameters:
    -----------
    X : pandas DataFrame
        Input features
    y : pandas Series
        Target variable
    classifier : sklearn-compatible classifier
        Machine learning classifier to use
    n_splits : int, optional
        Number of cross-validation splits (default: 5)
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Prepare storage for metrics
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc_roc': []
    }
    
    # Store predictions and true labels for final plots
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
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
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        # Store metrics
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['auc_roc'].append(auc)
        
        # Store fold-level metrics
        fold_metrics_data.append({
            'Fold': fold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'AUC_ROC': auc
        })
        
        # Store for final plots
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
    
    # Calculate mean metrics
    metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
    
    return {
        'metrics': metrics_mean,
        'metrics_by_fold': fold_metrics_data,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'y_prob': all_y_prob
    }

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
        # Scale features if required
        X = scaler.fit_transform(X)
    # Train the model on the entire dataset
    classifier.fit(X, y)

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    # Save the trained model using pickle
    model_path = os.path.join(save_path, f'{save_path}\{classifier_name}.pkl')
    with open(model_path, "wb") as file:
        pickle.dump(classifier, file)
    
    
    if requires_scaling:
        scaler_path = os.path.join(save_path, f'{save_path}\{classifier_name}_scaler.pkl')
        with open(scaler_path, "wb") as file:
            pickle.dump(scaler, file)

    print(f"Model saved to: {model_path}")

def save_evaluation_metrics(results, filename):
    """
    Save evaluation metrics to multiple CSV files
    
    Parameters:
    -----------
    results : dict
        Dictionary containing evaluation metrics
    filename : str
        Base filename for saving CSV files
    """
    # 1. Save overall metrics
    metrics_df = pd.DataFrame.from_dict(results['metrics'], orient='index', columns=['Value'])
    metrics_df.index.name = 'Metric'
    metrics_df.to_csv(f'{filename}_overall_metrics.csv')
    
    # 2. Save fold-level metrics
    fold_metrics_df = pd.DataFrame(results['metrics_by_fold'])
    fold_metrics_df.to_csv(f'{filename}_fold_metrics.csv', index=False)
    
    # 3. Save ROC curve data
    roc_data = pd.DataFrame({
        'y_true': results['y_true'],
        'y_pred': results['y_pred'],
        'y_prob': results['y_prob']
    })
    roc_data.to_csv(f'{filename}_roc_data.csv', index=False)
    
    print(f"Metrics saved:")
    print(f"- Overall metrics: {filename}_overall_metrics.csv")
    print(f"- Fold-level metrics: {filename}_fold_metrics.csv")
    print(f"- ROC data: {filename}_roc_data.csv")

def plot_roc_curve(results, filename):
    """
    Plot ROC curve with individual fold results and mean ROC curve
    
    Parameters:
    -----------
    results : dict
        Dictionary containing true labels, predictions, and probabilities
    filename : str
        Base filename for saving ROC plot
    """
    plt.figure(figsize=(10, 8))
    
    # Color palette for fold lines (less bright)
    fold_colors = [
        '#B0C4DE',  # Light Steel Blue
        '#E6E6FA',  # Lavender
        '#D3D3D3',  # Light Gray
        '#A9A9A9',  # Dark Gray
        '#778899'   # Slate Gray
    ]
    
    # Compute and plot individual fold ROC curves
    fold_size = len(results['y_true']) // 5
    fold_aucs = []
    
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size
        
        # Get fold-specific data
        fold_y_true = results['y_true'][start:end]
        fold_y_prob = results['y_prob'][start:end]
        
        # Compute ROC for this fold
        fpr, tpr, _ = roc_curve(fold_y_true, fold_y_prob)
        fold_auc = roc_auc_score(fold_y_true, fold_y_prob)
        fold_aucs.append(fold_auc)
        
        # Plot fold ROC with less bright color
        plt.plot(fpr, tpr, color=fold_colors[i], 
                 label=f'Fold {i+1} (AUC = {fold_auc:.3f})', 
                 alpha=0.6)
    
    # Compute and plot mean ROC curve
    mean_fpr = np.arange(0, 1.01, 0.01)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size
        
        fold_y_true = results['y_true'][start:end]
        fold_y_prob = results['y_prob'][start:end]
        
        fpr, tpr, _ = roc_curve(fold_y_true, fold_y_prob)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    # Average the interpolated TPR
    mean_tpr /= 5
    
    # Compute mean AUC
    mean_auc = np.mean(fold_aucs)
    
    # Plot mean ROC curve with darker, more prominent color
    plt.plot(mean_fpr, mean_tpr, color='#1E90FF', 
             label=f'Mean ROC (AUC = {mean_auc:.3f})', 
             linewidth=2.5)
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', 
             label='Random Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f'{filename}_roc_curve.png')
    plt.close()
    
    print(f"ROC Curve saved to {filename}_roc_curve.png")
    
    return fold_aucs, mean_auc

def plot_confusion_matrix(results, filename):
    """
    Plot Confusion Matrix
    
    Parameters:
    -----------
    results : dict
        Dictionary containing true labels and predictions
    filename : str
        Base filename for saving Confusion Matrix plot
    """
    plt.figure(figsize=(8, 6))
    
    # Compute confusion matrix
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    
    # Plot confusion matrix using seaborn
    sns.heatmap(cm, 
                annot=True,      # Show numeric values
                fmt='d',          # Integer formatting
                cmap='Blues',     # Color scheme
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_confusion_matrix.png')
    plt.close()
    
    print(f"Confusion Matrix saved to {filename}_confusion_matrix.png")
    
    # Return the confusion matrix for potential further analysis
    return cm

# Optional: If you want to add these to your existing workflow
def save_roc_details(fold_aucs, mean_auc, filename):
    """
    Save ROC AUC details to a CSV
    
    Parameters:
    -----------
    fold_aucs : list
        AUC scores for each fold
    mean_auc : float
        Mean AUC score
    filename : str
        Filename to save the CSV
    """
    import pandas as pd
    
    # Create DataFrame
    roc_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(fold_aucs))],
        'AUC': fold_aucs
    })
    
    # Add mean AUC
    roc_df = roc_df.append({
        'Fold': 'Mean',
        'AUC': mean_auc
    }, ignore_index=True)
    
    # Save to CSV
    roc_df.to_csv(f'{filename}_roc_details.csv', index=False)
    print(f"ROC AUC details saved to {filename}_roc_details.csv")

def get_classifier(classifier_name):
    """
    Return the specified classifier.
    
    Parameters:
    -----------
    classifier_name : str
        Name of the classifier to use
    
    Returns:
    --------
    sklearn classifier
        Initialized classifier
    """
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        # 'DT': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGB': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    if classifier_name not in classifiers:
        raise ValueError(f"Invalid classifier. Choose from {', '.join(classifiers.keys())}")
    
    return classifiers[classifier_name]

def classification(X, y, save_path, classifier_name):
    classifier = get_classifier(classifier_name)
    results = train_and_evaluate_model(X, y, classifier)
    train_and_save_model(X, y, classifier, save_path, classifier_name)
    # Print metrics
    for metric, value in results['metrics'].items():
        print(f"{metric.capitalize()}: {value:.4f}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    # Save metrics and plots
    save_evaluation_metrics(results, f'{save_path}\{classifier_name}')
    plot_confusion_matrix(results, f'{save_path}\{classifier_name}')
    plot_roc_curve(results, f'{save_path}\{classifier_name}')
    
### Example usage
save_path = r"D:\Kits23_result\Kidney"
data_path = r"D:\Kits23_data\3D_Kidney_Radiomics.csv"
X_train, X_test, y_train, y_test = preprocess_data(data_path)
for classifier in ['XGB', 'RF', 'SVM']:
    classification(X_train, y_train, save_path , classifier)