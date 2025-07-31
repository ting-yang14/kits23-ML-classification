import argparse
import json
import os
import yaml
import pickle
from itertools import product
from preprocess import preprocess_data
from train_evaluate import train_and_evaluate_model
from visualize import plot_roc_curve, plot_test_roc_curve, plot_confusion_matrix
from test import test_model
from utils import (
    train_and_save_model,
    save_evaluation_metrics,
    save_roc_details,
    get_classifier,
)


def generate_param_combinations(params):
    """Generate all combinations of parameters for grid search."""
    param_names = list(params.keys())
    param_values = list(params.values())

    # Convert single values to lists for consistent processing
    param_values = [v if isinstance(v, list) else [v] for v in param_values]

    # Generate all combinations
    combinations = []
    for combo in product(*param_values):
        combinations.append(dict(zip(param_names, combo)))

    return combinations


def train_single_model(
    X_train, y_train, case_id_train, base_name, params, train_save_path
):
    """Train a single model with given parameters."""
    # Generate readable classifier name with parameters
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    full_classifier_name = f"{base_name}_{param_str}" if param_str else base_name

    print(f"\nTraining {full_classifier_name} with parameters: {params}")
    classifier = get_classifier(base_name, params)
    results = train_and_evaluate_model(X_train, y_train, case_id_train, classifier)

    train_folder = os.path.join(train_save_path, base_name)
    os.makedirs(train_folder, exist_ok=True)

    train_and_save_model(
        X_train, y_train, classifier, train_folder, full_classifier_name
    )
    save_evaluation_metrics(results, train_folder, full_classifier_name)
    fold_aucs, mean_auc = plot_roc_curve(results, train_folder, full_classifier_name)
    plot_confusion_matrix(results, train_folder, full_classifier_name)
    save_roc_details(results, train_folder, full_classifier_name, mean_auc, fold_aucs)

    # Print metrics
    for metric, value in results["metrics"].items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return results, full_classifier_name


def test_single_model(
    X_test,
    y_test,
    case_id_test,
    model_path,
    scaler_path,
    test_save_path,
    full_classifier_name,
    base_name,
):
    """Test a single model with given paths."""
    # Validate model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found for {full_classifier_name}. Skipping...")
        return None

    # Load model
    try:
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        print(f"Successfully loaded model: {full_classifier_name}")
    except Exception as e:
        print(f"Error loading model {full_classifier_name}: {e}")
        return None

    # Load scaler if needed and exists
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"Successfully loaded scaler for {full_classifier_name}")
        except Exception as e:
            print(f"Error loading scaler for {full_classifier_name}: {e}")
            return None
    elif base_name == "SVM" and not scaler_path:
        print(
            f"Warning: SVM model {full_classifier_name} may need a scaler but none provided"
        )

    # Create test folder
    test_folder = os.path.join(test_save_path, base_name)
    os.makedirs(test_folder, exist_ok=True)

    # Test the model
    try:
        test_results = test_model(X_test, y_test, case_id_test, classifier, scaler)

        # Save results
        save_evaluation_metrics(test_results, test_folder, full_classifier_name)
        _, test_auc = plot_test_roc_curve(
            test_results, test_folder, full_classifier_name
        )
        plot_confusion_matrix(test_results, test_folder, full_classifier_name)
        save_roc_details(
            test_results, test_folder, full_classifier_name, test_auc, None
        )

        # Print metrics
        print(f"\nTest results for {full_classifier_name}:")
        for metric, value in test_results["metrics"].items():
            print(f"{metric.capitalize()}: {value:.4f}")

        return test_results

    except Exception as e:
        print(f"Error testing model {full_classifier_name}: {e}")
        return None


def get_model_paths(train_save_path, base_name, full_classifier_name):
    """Get model and scaler paths for a given classifier."""
    train_folder = os.path.join(train_save_path, base_name)
    model_path = os.path.join(train_folder, f"{full_classifier_name}.pkl")

    # Check for scaler (not just for SVM, but for any model that might have one)
    scaler_path = os.path.join(train_folder, f"{full_classifier_name}_scaler.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = None

    return model_path, scaler_path


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers for KiTS23 dataset."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--train_save_path",
        type=str,
        help="Path to save trained models and results.",
    )
    parser.add_argument("--test_save_path", type=str, help="Path to save test results.")
    parser.add_argument(
        "--classifier_name", type=str, help="Classifier to use (RF, SVM, XGB)."
    )
    parser.add_argument(
        "--params", type=str, help="JSON string of classifier parameters."
    )
    parser.add_argument(
        "--config", type=str, help="Path to YAML config file for multiple models."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        default="train",
        help="Mode to run the script in (train or test).",
    )
    parser.add_argument("--testing_model", type=str, help="Path to the model to test.")
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Enable grid search mode for parameter combinations.",
    )

    args = parser.parse_args()

    # Load data
    X_train, X_test, y_train, y_test, case_id_train, case_id_test = preprocess_data(
        args.data_path
    )

    if args.mode == "train":
        if args.config:
            # Load YAML configuration
            with open(args.config, "r") as file:
                config = yaml.safe_load(file)

            # Track all results for comparison
            all_results = []

            for model_config in config["models"]:
                base_name = model_config["name"]
                params = model_config.get("params", {})

                if args.grid_search:
                    # Grid search mode: generate all parameter combinations
                    print(f"\n=== Grid Search for {base_name} ===")
                    param_combinations = generate_param_combinations(params)
                    print(f"Total combinations to test: {len(param_combinations)}")

                    model_results = []
                    for i, param_combo in enumerate(param_combinations, 1):
                        print(f"\n--- Combination {i}/{len(param_combinations)} ---")
                        results, full_name = train_single_model(
                            X_train,
                            y_train,
                            case_id_train,
                            base_name,
                            param_combo,
                            args.train_save_path,
                        )
                        model_results.append(
                            {
                                "name": full_name,
                                "params": param_combo,
                                "results": results,
                            }
                        )
                        all_results.append(model_results[-1])

                    # Find best model for this classifier type
                    best_model = max(
                        model_results, key=lambda x: x["results"]["metrics"]["auc"]
                    )
                    print(f"\n=== Best {base_name} Model ===")
                    print(f"Best model: {best_model['name']}")
                    print(f"Best parameters: {best_model['params']}")
                    print(f"Best AUC: {best_model['results']['metrics']['auc']:.4f}")
                    print(
                        f"Best F1 score: {best_model['results']['metrics']['f1_score']:.4f}"
                    )

                else:
                    # Single condition mode: train with specified parameters
                    results, full_name = train_single_model(
                        X_train,
                        y_train,
                        case_id_train,
                        base_name,
                        params,
                        args.train_save_path,
                    )
                    all_results.append(
                        {"name": full_name, "params": params, "results": results}
                    )

            # Print summary of all models
            if args.grid_search:
                print(f"\n=== Grid Search Summary ===")
                print(f"Total models trained: {len(all_results)}")

                # Group by classifier type
                classifier_types = {}
                for result in all_results:
                    classifier_type = result["name"].split("_")[0]
                    if classifier_type not in classifier_types:
                        classifier_types[classifier_type] = []
                    classifier_types[classifier_type].append(result)

                # Show best model for each type
                for classifier_type, models in classifier_types.items():
                    best_model = max(
                        models, key=lambda x: x["results"]["metrics"]["auc"]
                    )
                    print(f"\nBest {classifier_type}: {best_model['name']}")
                    print(f"  AUC: {best_model['results']['metrics']['auc']:.4f}")
                    print(
                        f"  F1 score: {best_model['results']['metrics']['f1_score']:.4f}"
                    )
                    print(f"  Parameters: {best_model['params']}")

                # Overall best model
                overall_best = max(
                    all_results, key=lambda x: x["results"]["metrics"]["auc"]
                )
                print(f"\nOverall Best Model: {overall_best['name']}")
                print(f"  AUC: {overall_best['results']['metrics']['auc']:.4f}")
                print(f"  Parameters: {overall_best['params']}")

        else:
            # Single classifier mode
            if not args.classifier_name:
                raise ValueError(
                    "Classifier name must be provided if no config file is specified."
                )

            params = json.loads(args.params) if args.params else {}

            if args.grid_search:
                # Grid search for single classifier
                print(f"\n=== Grid Search for {args.classifier_name} ===")
                param_combinations = generate_param_combinations(params)
                print(f"Total combinations to test: {len(param_combinations)}")

                all_results = []
                for i, param_combo in enumerate(param_combinations, 1):
                    print(f"\n--- Combination {i}/{len(param_combinations)} ---")
                    results, full_name = train_single_model(
                        X_train,
                        y_train,
                        case_id_train,
                        args.classifier_name,
                        param_combo,
                        args.train_save_path,
                    )
                    all_results.append(
                        {"name": full_name, "params": param_combo, "results": results}
                    )

                # Find and display best model
                best_model = max(
                    all_results, key=lambda x: x["results"]["metrics"]["auc"]
                )
                print(f"\n=== Best Model ===")
                print(f"Best model: {best_model['name']}")
                print(f"Best parameters: {best_model['params']}")
                print(f"Best AUC: {best_model['results']['metrics']['auc']:.4f}")
                print(
                    f"Best F1 score: {best_model['results']['metrics']['f1_score']:.4f}"
                )

            else:
                # Single condition mode
                results, full_name = train_single_model(
                    X_train,
                    y_train,
                    case_id_train,
                    args.classifier_name,
                    params,
                    args.train_save_path,
                )

    elif args.mode == "test":
        if not args.test_save_path:
            raise ValueError("Test save path must be provided for testing.")

        # Validate test data
        if X_test is None or y_test is None or case_id_test is None:
            raise ValueError("Test data is not properly loaded.")

        print(f"Testing on {len(X_test)} samples...")

        if args.config:
            # Load YAML configuration for testing multiple models
            with open(args.config, "r") as file:
                config = yaml.safe_load(file)

            all_test_results = []

            for model_config in config["models"]:
                base_name = model_config["name"]
                params = model_config.get("params", {})

                if args.grid_search:
                    # Test all parameter combinations from grid search
                    print(f"\n=== Testing Grid Search Results for {base_name} ===")
                    param_combinations = generate_param_combinations(params)

                    model_test_results = []
                    for param_combo in param_combinations:
                        param_str = "_".join(
                            f"{k}={v}" for k, v in sorted(param_combo.items())
                        )
                        full_classifier_name = (
                            f"{base_name}_{param_str}" if param_str else base_name
                        )

                        model_path, scaler_path = get_model_paths(
                            args.train_save_path, base_name, full_classifier_name
                        )

                        test_result = test_single_model(
                            X_test,
                            y_test,
                            case_id_test,
                            model_path,
                            scaler_path,
                            args.test_save_path,
                            full_classifier_name,
                            base_name,
                        )

                        if test_result:
                            model_test_results.append(
                                {
                                    "name": full_classifier_name,
                                    "params": param_combo,
                                    "results": test_result,
                                }
                            )

                    # Find best performing model on test set
                    if model_test_results:
                        best_test_model = max(
                            model_test_results,
                            key=lambda x: x["results"]["metrics"]["auc"],
                        )
                        print(f"\n=== Best {base_name} Model on Test Set ===")
                        print(f"Best model: {best_test_model['name']}")
                        print(f"Best parameters: {best_test_model['params']}")
                        print(
                            f"Test AUC: {best_test_model['results']['metrics']['auc']:.4f}"
                        )
                        print(
                            f"Test F1 score: {best_test_model['results']['metrics']['f1_score']:.4f}"
                        )

                        all_test_results.extend(model_test_results)
                    else:
                        print(f"No valid models found for {base_name}")

                else:
                    # Single condition testing
                    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
                    full_classifier_name = (
                        f"{base_name}_{param_str}" if param_str else base_name
                    )

                    model_path, scaler_path = get_model_paths(
                        args.train_save_path, base_name, full_classifier_name
                    )

                    test_result = test_single_model(
                        X_test,
                        y_test,
                        case_id_test,
                        model_path,
                        scaler_path,
                        args.test_save_path,
                        full_classifier_name,
                        base_name,
                    )

                    if test_result:
                        all_test_results.append(
                            {
                                "name": full_classifier_name,
                                "params": params,
                                "results": test_result,
                            }
                        )

            # Print overall test summary
            if all_test_results:
                print(f"\n=== Overall Test Summary ===")
                print(f"Total models tested: {len(all_test_results)}")

                # Overall best model on test set
                overall_best_test = max(
                    all_test_results, key=lambda x: x["results"]["metrics"]["auc"]
                )
                print(f"\nOverall Best Model on Test Set: {overall_best_test['name']}")
                print(
                    f"  Test AUC: {overall_best_test['results']['metrics']['auc']:.4f}"
                )
                print(
                    f"  Test F1 score: {overall_best_test['results']['metrics']['f1_score']:.4f}"
                )
                print(f"  Parameters: {overall_best_test['params']}")
            else:
                print("No models were successfully tested.")

        else:
            # Single classifier test mode
            if not args.testing_model:
                raise ValueError("Test model path must be provided for testing.")

            print(f"\nTesting model: {args.testing_model}")

            # Extract information from model path
            model_filename = os.path.basename(args.testing_model)
            base_name = model_filename.split("_")[0]
            full_classifier_name = os.path.splitext(model_filename)[0]

            # Determine scaler path
            scaler_path = args.testing_model.replace(".pkl", "_scaler.pkl")
            if not os.path.exists(scaler_path):
                scaler_path = None

            test_result = test_single_model(
                X_test,
                y_test,
                case_id_test,
                args.testing_model,
                scaler_path,
                args.test_save_path,
                full_classifier_name,
                base_name,
            )

            if not test_result:
                print("Testing failed.")
            else:
                print("Testing completed successfully.")


if __name__ == "__main__":
    main()
