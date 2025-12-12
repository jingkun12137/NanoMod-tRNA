#!/usr/bin/env python3
"""
Evaluate NanoMod-tRNA prediction accuracy by comparing predictions with ground truth modification sites.
"""

import argparse
import pandas as pd
import sys
import os

def evaluate_predictions(mod_type, prediction_file, modification_sites_file):
    """
    Evaluate prediction accuracy by comparing with ground truth modification sites.
    
    Args:
        mod_type: Modification type (e.g., 'D')
        prediction_file: Path to prediction results TSV file
        modification_sites_file: Path to ground truth modification sites TSV file
    
    Returns:
        accuracy: Accuracy as a float between 0 and 1
    """
    
    # Check if files exist
    if not os.path.exists(prediction_file):
        print(f"Error: Prediction file not found: {prediction_file}")
        sys.exit(1)
    
    if not os.path.exists(modification_sites_file):
        print(f"Error: Modification sites file not found: {modification_sites_file}")
        sys.exit(1)
    
    # Load prediction results
    try:
        predictions = pd.read_csv(prediction_file, sep='\t')
        print(f"Loaded {len(predictions)} predictions from {prediction_file}")
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        sys.exit(1)
    
    # Load ground truth modification sites
    try:
        mod_sites = pd.read_csv(modification_sites_file, sep='\t')
        print(f"Loaded {len(mod_sites)} modification sites from {modification_sites_file}")
    except Exception as e:
        print(f"Error loading modification sites file: {e}")
        sys.exit(1)
    
    # Filter modification sites by modification type
    # Handle both string and numeric modification types
    try:
        # Try numeric comparison first
        numeric_mod_type = int(mod_type)
        mod_sites_filtered = mod_sites[mod_sites['modified_base'] == numeric_mod_type]
    except ValueError:
        # If conversion fails, use string comparison
        mod_sites_filtered = mod_sites[mod_sites['modified_base'] == mod_type]
    
    print(f"Found {len(mod_sites_filtered)} {mod_type} modification sites")
    
    # Create ground truth set: trna_combined + "_" + position
    ground_truth_sites = set()
    for _, row in mod_sites_filtered.iterrows():
        site_id = f"{row['trna_combined']}_{row['position']}"
        ground_truth_sites.add(site_id)
    
    print(f"Ground truth sites: {len(ground_truth_sites)}")
    
    # Evaluate predictions
    correct_predictions = 0
    total_predictions = len(predictions)
    
    for _, row in predictions.iterrows():
        site_id = row['site_id']
        predicted_modified = row['modified']
        
        # Check if this site should be modified according to ground truth
        should_be_modified = site_id in ground_truth_sites
        
        # Check if prediction is correct
        if (predicted_modified == 1 and should_be_modified) or (predicted_modified == 0 and not should_be_modified):
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Print detailed results
    print(f"\n=== Evaluation Results ===")
    print(f"Modification type: {mod_type}")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate NanoMod-tRNA prediction accuracy')
    parser.add_argument('--mod-type', type=str, required=True,
                        help='Modification type (e.g., D)')
    parser.add_argument('--prediction-file', type=str, required=True,
                        help='Path to prediction results TSV file')
    parser.add_argument('--modification-sites-file', type=str, 
                        default=None,
                        help='Path to ground truth modification sites TSV file (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Auto-detect modification sites file if not provided
    if args.modification_sites_file is None:
        args.modification_sites_file = f'NanoMod_data/tRNA_{args.mod_type}_modification_sites.tsv'
        print(f"Auto-detected modification sites file: {args.modification_sites_file}")
    
    # Evaluate predictions
    accuracy = evaluate_predictions(args.mod_type, args.prediction_file, args.modification_sites_file)
    
    return accuracy

if __name__ == '__main__':
    main()
