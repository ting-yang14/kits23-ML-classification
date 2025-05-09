import argparse
import json
import os
import yaml
import pickle
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

            for model_config in config["models"]:
                base_name = model_config["name"]
                params = model_config.get("params", {})

                # Generate readable classifier name with parameters
                param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
                full_classifier_name = (
                    f"{base_name}_{param_str}" if param_str else base_name
                )

                print(f"\nTraining {full_classifier_name} with parameters: {params}")
                classifier = get_classifier(base_name, params)
                results = train_and_evaluate_model(
                    X_train, y_train, case_id_train, classifier
                )

                train_folder = os.path.join(args.train_save_path, base_name)
                os.makedirs(train_folder, exist_ok=True)

                train_and_save_model(
                    X_train,
                    y_train,
                    classifier,
                    train_folder,
                    full_classifier_name,
                )
                save_evaluation_metrics(results, train_folder, full_classifier_name)
                fold_aucs, mean_auc = plot_roc_curve(
                    results, train_folder, full_classifier_name
                )
                plot_confusion_matrix(results, train_folder, full_classifier_name)
                save_roc_details(
                    results,
                    train_folder,
                    full_classifier_name,
                    mean_auc,
                    fold_aucs,
                )

                for metric, value in results["metrics"].items():
                    print(f"{metric.capitalize()}: {value:.4f}")

        else:
            # Single classifier mode
            if not args.classifier_name:
                raise ValueError(
                    "Classifier name must be provided if no config file is specified."
                )

            params = json.loads(args.params) if args.params else {}
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            full_classifier_name = (
                f"{args.classifier_name}_{param_str}"
                if param_str
                else args.classifier_name
            )

            print(f"\nTraining {full_classifier_name} with parameters: {params}")
            classifier = get_classifier(args.classifier_name, params)
            results = train_and_evaluate_model(
                X_train, y_train, case_id_train, classifier
            )

            train_folder = os.path.join(args.train_save_path, args.classifier_name)
            os.makedirs(train_folder, exist_ok=True)

            train_and_save_model(
                X_train, y_train, classifier, train_folder, full_classifier_name
            )
            save_evaluation_metrics(results, train_folder, full_classifier_name)
            fold_aucs, mean_auc = plot_roc_curve(
                results, train_folder, full_classifier_name
            )
            plot_confusion_matrix(results, train_folder, full_classifier_name)
            save_roc_details(
                results, train_folder, full_classifier_name, mean_auc, fold_aucs
            )

            for metric, value in results["metrics"].items():
                print(f"{metric.capitalize()}: {value:.4f}")

    elif args.mode == "test":
        # Testing mode
        if args.config:
            # Load YAML configuration for testing multiple models
            with open(args.config, "r") as file:
                config = yaml.safe_load(file)

            for model_config in config["models"]:
                base_name = model_config["name"]
                params = model_config.get("params", {})

                # Generate model name with parameters
                param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
                full_classifier_name = (
                    f"{base_name}_{param_str}" if param_str else base_name
                )

                train_folder = os.path.join(args.train_save_path, base_name)
                model_path = os.path.join(train_folder, f"{full_classifier_name}.pkl")
                scaler_path = os.path.join(
                    train_folder, f"{full_classifier_name}_scaler.pkl"
                )

                # Check if model file exists
                if not os.path.exists(model_path):
                    print(
                        f"Model file not found for {full_classifier_name}. Skipping..."
                    )
                    continue

                # Check if scaler is needed and exists (for SVM)
                if base_name == "SVM" and not os.path.exists(scaler_path):
                    print(
                        f"Scaler file not found for {full_classifier_name}. Skipping..."
                    )
                    continue

                print(f"\nTesting {full_classifier_name}")

                # Load the model
                with open(model_path, "rb") as f:
                    classifier = pickle.load(f)

                # Load scaler if needed
                scaler = None
                if base_name == "SVM" and os.path.exists(scaler_path):
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)

                test_folder = os.path.join(args.test_save_path, base_name)
                os.makedirs(test_folder, exist_ok=True)

                # Test the model
                test_results = test_model(
                    X_test, y_test, case_id_test, classifier, scaler
                )
                # Save test results
                save_evaluation_metrics(test_results, test_folder, full_classifier_name)

                # Create test visualizations
                _, test_auc = plot_test_roc_curve(
                    test_results, test_folder, full_classifier_name
                )
                plot_confusion_matrix(test_results, test_folder, full_classifier_name)
                save_roc_details(
                    test_results, test_folder, full_classifier_name, test_auc, None
                )
                # Display metrics
                print(f"Test results for {full_classifier_name}:")
                for metric, value in test_results["metrics"].items():
                    print(f"{metric.capitalize()}: {value:.4f}")

        else:
            # Single classifier test mode
            # 檢查必要參數
            if not args.testing_model:
                raise ValueError("Test model path must be provided for testing.")
            if not args.test_save_path:
                raise ValueError("Test save path must be provided for testing.")

            print(f"\nTesting model: {args.testing_model}")

            # 取得模型檔名與 base model 名稱
            model_filename = os.path.basename(args.testing_model)
            base_name = model_filename.split("_")[0]

            # 建立測試輸出資料夾
            test_folder = os.path.join(args.test_save_path, base_name)
            os.makedirs(test_folder, exist_ok=True)

            # 處理 SVM 專屬的 Scaler
            scaler = None
            if base_name == "SVM":
                scaler_path = args.testing_model.replace(".pkl", "_scaler.pkl")
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
                print(f"Scaler found: {scaler_path}")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

            # 載入模型
            with open(args.testing_model, "rb") as f:
                classifier = pickle.load(f)

            # 執行模型測試
            test_results = test_model(X_test, y_test, case_id_test, classifier, scaler)

            full_classifier_name = os.path.splitext(model_filename)[0]
            # 儲存測試結果與視覺化
            save_evaluation_metrics(test_results, test_folder, full_classifier_name)
            _, test_auc = plot_test_roc_curve(
                test_results, test_folder, full_classifier_name
            )
            plot_confusion_matrix(test_results, test_folder, full_classifier_name)
            save_roc_details(
                test_results, test_folder, full_classifier_name, test_auc, None
            )

            # 印出測試指標
            print(f"\nTest results for {full_classifier_name}:")
            for metric, value in test_results["metrics"].items():
                print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
