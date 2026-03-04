#!/bin/bash

echo "Starting HPC Storage Cleanup..."

echo "1/4: Emptying JupyterLab hidden trash..."
rm -rf ~/.local/share/Trash/files/*
rm -rf ~/.local/share/Trash/info/*

echo "2/4: Purging pip download cache..."
rm -rf ~/.cache/pip/*

echo "3/4: Clearing Hugging Face and matplotlib caches..."
rm -rf ~/.cache/huggingface/*
rm -rf ~/.cache/matplotlib/*

echo "4/4: Cleaning temporary VSC scratch/cache (if any exist)..."
# Using -f so it doesn't throw errors if the directory is already empty
rm -f /tmp/cache.* 2>/dev/null
