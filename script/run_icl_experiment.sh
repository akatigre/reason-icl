#!/bin/bash

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0  # Change this to the GPU you want to use

# Define the model name and split
MODEL_NAME="qwen2_5"
SPLIT="val"  # Use "val" for validation set, "test" for test set
VISUALIZE_ERRORS="--visualize_errors"  # Add this flag to visualize wrong predictions

# Create logs directory if it doesn't exist
mkdir -p logs/AOKVQA

# Run individual methods
echo "Running tag_trace experiment..."
python icl_experiment.py --model_name $MODEL_NAME --split $SPLIT --methods tag_trace --examples 8 $VISUALIZE_ERRORS

echo "Running embedding similarity experiment..."
python icl_experiment.py --model_name $MODEL_NAME --split $SPLIT --methods embedding --examples 8 $VISUALIZE_ERRORS

echo "Running VAE-trained embedding experiment..."
python icl_experiment.py --model_name $MODEL_NAME --split $SPLIT --methods vae --examples 8 $VISUALIZE_ERRORS

echo "Running cluster-based similarity experiment..."
python icl_experiment.py --model_name $MODEL_NAME --split $SPLIT --methods cluster --examples 8 $VISUALIZE_ERRORS

echo "Running random baseline for comparison..."
python icl_experiment.py --model_name $MODEL_NAME --split $SPLIT --methods random --examples 8 $VISUALIZE_ERRORS

# Run all methods for comparison (optional)
echo "Running all methods for comparison..."
python icl_experiment.py --model_name $MODEL_NAME --split $SPLIT --methods random tag_trace embedding vae cluster --examples 8 $VISUALIZE_ERRORS

# Visualize and analyze results
echo "Analyzing and visualizing results..."
python visualize_results.py --split $SPLIT --model $MODEL_NAME --methods random tag_trace embedding vae cluster --output ./results

echo "Experiment completed! Results saved to logs/AOKVQA/ and visualizations in results/" 