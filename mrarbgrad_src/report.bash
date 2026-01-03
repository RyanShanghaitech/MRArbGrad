#!/bin/bash

# Define the output file
OUTPUT_FILE="report.txt"

# Clear the output file if it already exists
> "$OUTPUT_FILE"

# Find files with .cpp, .h, or .py extensions and concatenate them
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" \) -exec cat {} + >> "$OUTPUT_FILE"

echo "All .cpp, .h, and .py files have been concatenated into $OUTPUT_FILE"