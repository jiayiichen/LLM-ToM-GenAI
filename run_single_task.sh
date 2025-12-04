#!/bin/bash

PROJECT_ID="genai-personal-478819"
CONFIG_FILE="model_gardem_configs/gemini.yaml"
CSV_FILE="data_processing/sample_test.csv"
OUTPUT_DIR="outputs/gemini_sample_test"

echo "Starting inference..."
echo "Project: $PROJECT_ID"
echo "Config: $CONFIG_FILE"
echo "CSV: $CSV_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
    echo "Output directory already exists. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi


python3 inference_submission/inference.py \
    -d "$CSV_FILE" \
    -o "$OUTPUT_DIR" \
    -p "$PROJECT_ID" \
    -c "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "Inference completed successfully!"
    echo ""
    echo "Parsing inference results..."
    PARSED_OUTPUT="${OUTPUT_DIR}/parsed_results.json"
    
    python3 parse_inference/parse_inference.py \
        -i "$OUTPUT_DIR" \
        -o "$PARSED_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "All tasks completed!"
        echo "Results saved to: $PARSED_OUTPUT"
    else
        echo "Parsing failed"
        exit 1
    fi
else
    echo "Inference failed"
    exit 1
fi
