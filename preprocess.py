import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(file_path, test_size=0.2, random_state=42):
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
