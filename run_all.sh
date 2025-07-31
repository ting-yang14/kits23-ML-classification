#!/bin/bash

# Configuration
CONFIG_FILE="config.yaml"
DATA_DIR="./data"
TRAIN_DIR="./training_results"
TEST_DIR="./testing_results"

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Function to check if required files exist
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file '$CONFIG_FILE' not found!"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory '$DATA_DIR' not found!"
        exit 1
    fi
    
    # Check if there are any CSV files
    if ! ls "$DATA_DIR"/*.csv 1> /dev/null 2>&1; then
        print_error "No CSV files found in '$DATA_DIR'!"
        exit 1
    fi
    
    # Check if main.py exists
    if [ ! -f "main.py" ]; then
        print_error "main.py not found in current directory!"
        exit 1
    fi
    
    print_success "All prerequisites checked"
}

# Function to create directories
create_directories() {
    print_status "Creating output directories..."
    mkdir -p "$TRAIN_DIR"
    mkdir -p "$TEST_DIR"
    print_success "Directories created"
}

# Function to process a single dataset
process_dataset() {
    local data_path="$1"
    local dataset_name=$(basename "$data_path" .csv)
    
    print_info "Processing dataset: $dataset_name"
    print_status "Data path: $data_path"
    
    # Create dataset-specific directories
    local train_output="$TRAIN_DIR/$dataset_name"
    local test_output="$TEST_DIR/$dataset_name"
    
    mkdir -p "$train_output"
    mkdir -p "$test_output"
    
    # Training Phase
    print_status "🏋️ Training models for $dataset_name..."
    
    if python main.py \
        --data_path "$data_path" \
        --train_save_path "$train_output" \
        --config "$CONFIG_FILE" \
        --mode train \
        --grid_search; then
        print_success "Training completed for $dataset_name"
    else
        print_error "Training failed for $dataset_name"
        return 1
    fi
    
    # Check if any models were actually created
    if [ ! -d "$train_output" ] || [ -z "$(find "$train_output" -name "*.pkl" 2>/dev/null)" ]; then
        print_error "No trained models found for $dataset_name"
        return 1
    fi
    
    # Testing Phase
    print_status "🧪 Testing models for $dataset_name..."
    
    if python main.py \
        --data_path "$data_path" \
        --train_save_path "$train_output" \
        --test_save_path "$test_output" \
        --config "$CONFIG_FILE" \
        --mode test \
        --grid_search; then
        print_success "Testing completed for $dataset_name"
    else
        print_error "Testing failed for $dataset_name"
        return 1
    fi
    
    # Verify test results were generated
    if [ ! -d "$test_output" ] || [ -z "$(find "$test_output" -name "*_overall_metrics.csv" 2>/dev/null)" ]; then
        print_warning "No test results found for $dataset_name"
    fi
    
    return 0
}

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
    for data_path in "$DATA_DIR"/2D_*.csv; do
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