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

# Function to display summary statistics
display_summary() {
    local total_datasets=$1
    local successful_datasets=$2
    local failed_datasets=$3
    
    echo ""
    echo "=================================="
    echo "📊 PROCESSING SUMMARY"
    echo "=================================="
    echo "Total datasets: $total_datasets"
    echo "Successful: $successful_datasets"
    echo "Failed: $failed_datasets"
    echo ""
    
    if [ $failed_datasets -gt 0 ]; then
        print_warning "Some datasets failed to process. Check the logs above."
    else
        print_success "All datasets processed successfully!"
    fi
    
    # Show directory structure
    echo "📁 Output structure:"
    echo "Training results: $TRAIN_DIR"
    echo "Testing results: $TEST_DIR"
    echo ""
    
    # Show file counts
    if [ -d "$TRAIN_DIR" ]; then
        local model_count=$(find "$TRAIN_DIR" -name "*.pkl" 2>/dev/null | wc -l)
        echo "Total trained models: $model_count"
    fi
    
    if [ -d "$TEST_DIR" ]; then
        local result_count=$(find "$TEST_DIR" -name "*_overall_metrics.csv" 2>/dev/null | wc -l)
        echo "Total test results: $result_count"
    fi
}

# Function to handle script interruption
cleanup() {
    echo ""
    print_warning "Script interrupted by user"
    echo "Partial results may be available in:"
    echo "  - Training: $TRAIN_DIR"
    echo "  - Testing: $TEST_DIR"
    exit 1
}

# Main execution
main() {
    # Set up signal handlers
    trap cleanup INT TERM
    
    print_status "Starting batch processing of ML models"
    print_status "Config file: $CONFIG_FILE"
    print_status "Data directory: $DATA_DIR"
    print_status "Training output: $TRAIN_DIR"
    print_status "Testing output: $TEST_DIR"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Create output directories
    create_directories
    
    # Initialize counters
    local total_datasets=0
    local successful_datasets=0
    local failed_datasets=0
    
    # Process each CSV file
    # NOTE: The CSV file should be modified as the input feature table for training
    for data_path in "$DATA_DIR"/3D_Tumor_Radiomics.csv; do
        # Skip if no files match (in case of empty directory)
        [ -f "$data_path" ] || continue
        
        ((total_datasets++))
        
        echo ""
        echo "=================================="
        echo "📊 DATASET $total_datasets"
        echo "=================================="
        
        if process_dataset "$data_path"; then
            ((successful_datasets++))
            print_success "Completed dataset: $(basename "$data_path" .csv)"
        else
            ((failed_datasets++))
            print_error "Failed dataset: $(basename "$data_path" .csv)"
        fi
        
        echo "----------------------------------"
    done
    
    # Display final summary
    display_summary $total_datasets $successful_datasets $failed_datasets
    
    # Exit with appropriate code
    if [ $failed_datasets -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"
