#!/bin/bash

CONFIG_FILE="config.yaml"
DATA_DIR="./data"
TRAIN_DIR="./training_results"
TEST_DIR="./testing_results"

for data_path in "$DATA_DIR"/*.csv; do
    dataset_name=$(basename "$data_path" .csv)

    echo "🔧 Processing $dataset_name ..."

    # Train
    python main.py \
        --data_path "$data_path" \
        --train_save_path "$TRAIN_DIR/$dataset_name" \
        --config "$CONFIG_FILE" \
        --mode train

    # Test
    python main.py \
        --data_path "$data_path" \
        --train_save_path "$TRAIN_DIR/$dataset_name" \
        --test_save_path "$TEST_DIR/$dataset_name" \
        --config "$CONFIG_FILE" \
        --mode test

    echo "✅ Done with $dataset_name"
    echo "----------------------------------"
done