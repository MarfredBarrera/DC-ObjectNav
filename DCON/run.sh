#!/bin/bash

# Default values
CORES="112-127"
GPU="0"
QUERY="a pillow"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--cores) CORES="$2"; shift ;;
        -g|--gpu) GPU="$2"; shift ;;
        -q|--query) QUERY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting perception.py with:"
echo "  CPU Cores: $CORES"
echo "  GPU Device: $GPU"
echo "  Target Query: $QUERY"

taskset -c $CORES python perception.py --gpu $GPU --query "$QUERY"