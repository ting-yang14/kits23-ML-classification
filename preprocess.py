import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import h5py


def preprocess_radiomics_data(file_path, test_size=0.2, random_state=42):
    """
    Preprocess the input CSV file for machine learning classification, with optional manual train/test case split.

    Parameters:
    -----------
    file_path : str
        Path to the input CSV file.
    test_size : float, optional
        Proportion of the dataset to include in the test split (ignored if case IDs are provided).
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    X_train, X_test, y_train, y_test, case_id_train, case_id_test
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Drop NaN values
    df = df.dropna()
    # Store case_id separately
    case_id = df["case_id"]

    # Drop specified columns
    df = df.drop(columns=["case_id", "tumor_histologic_subtype"])

    # Convert gender to binary
    df["gender"] = df["gender"].map({"male": 0, "female": 1})

    # Convert malignant to binary target
    df["malignant"] = df["malignant"].map({True: 1, False: 0})

    # Separate features and target
    X = df.drop(columns=["malignant"])
    y = df["malignant"]

    # Align case_id with filtered data
    case_id = case_id.loc[X.index]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test, case_id_train, case_id_test = train_test_split(
        X, y, case_id, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, case_id_train, case_id_test


def read_case_embeddings(case_id, h5_dir, csv_path, feature_layer="bottleneck"):
    """
    Read embeddings from H5 file for a specific case and match with labels from CSV.
    (This is a simplified version of the previous function for context.)
    """
    try:
        h5_file = os.path.join(h5_dir, f"{case_id}_{feature_layer}.h5")
        if not os.path.exists(h5_file):
            print(f"❌ H5 file not found for case {case_id}: {h5_file}")
            return {"success": False}

        with h5py.File(h5_file, "r") as f:
            embeddings = f["features"][:]
            slice_nums = f["slice_nums"][:].tolist()

        df = pd.read_csv(csv_path)
        case_labels = df[df["case_id"] == case_id]
        if case_labels.empty:
            print(f"❌ No labels found for case {case_id} in {csv_path}")
            return {"success": False}

        labels = [case_labels["malignant"].iloc[0]] * len(slice_nums)
        return {
            "embeddings": embeddings,
            "slice_nums": slice_nums,
            "labels": labels,
            "case_id": case_id,
            "success": True,
        }
    except Exception as e:
        print(f"❌ Error reading embeddings for case {case_id}: {e}")
        return {"success": False}


def preprocess_embedding_data(
    h5_dir, csv_path, test_size=0.2, random_state=42, feature_layer="bottleneck"
):
    """
    Preprocess embeddings from H5 files and CSV labels for machine learning classification.

    Parameters:
    -----------
    h5_dir : str
        Directory containing H5 files with embeddings.
    csv_path : str
        Path to the CSV file containing ground truth labels ('2D_Embedding_GT.csv').
    test_size : float, optional
        Proportion of cases to include in the test split (default: 0.2).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    feature_layer : str, optional
        Feature layer to extract from H5 files (default: 'bottleneck').

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, case_id_train, case_id_test)
        - X_train, X_test: numpy arrays of concatenated embeddings
        - y_train, y_test: numpy arrays of binary labels (0/1)
        - case_id_train, case_id_test: lists of case IDs for train and test sets
    """
    # Read CSV to get all case IDs and labels
    df = pd.read_csv(csv_path)
    df = df.dropna()
    unique_case_ids = df["case_id"].unique()

    # Initialize lists to store all data
    all_embeddings = []
    all_labels = []
    all_case_ids = []
    all_slice_nums = []

    # Load embeddings for each case
    for case_id in unique_case_ids:
        result = read_case_embeddings(case_id, h5_dir, csv_path, feature_layer)
        if result["success"]:
            embeddings = result["embeddings"]
            labels = result["labels"]
            slice_nums = result["slice_nums"]

            # Append data
            all_embeddings.append(embeddings)
            all_labels.extend(
                [1 if label else 0 for label in labels]
            )  # Convert True/False to 1/0
            all_case_ids.extend([case_id] * len(slice_nums))
            all_slice_nums.extend(slice_nums)

    if not all_embeddings:
        raise ValueError("No valid embeddings loaded from H5 files.")

    # Concatenate all embeddings
    X = np.concatenate(all_embeddings, axis=0)
    y = np.array(all_labels)
    case_ids = np.array(all_case_ids)

    # Perform train-test split on unique case IDs to keep slices of a case together
    unique_case_ids = np.unique(case_ids)
    case_train, case_test = train_test_split(
        unique_case_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=df.set_index("case_id")
        .loc[unique_case_ids]["malignant"]
        .map({True: 1, False: 0}),
    )

    # Split data based on case IDs
    train_mask = np.isin(case_ids, case_train)
    test_mask = np.isin(case_ids, case_test)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    case_id_train = case_ids[train_mask].tolist()
    case_id_test = case_ids[test_mask].tolist()

    print(
        f"✓ Preprocessed data: {len(X_train)} training samples, {len(X_test)} test samples"
    )
    return X_train, X_test, y_train, y_test, case_id_train, case_id_test
