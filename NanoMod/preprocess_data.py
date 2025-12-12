#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NanoMod-tRNA Data Preprocessing Module

This module preprocesses the tRNA features data to separate modified and unmodified sites,
which greatly accelerates the training process.
"""

import os
import pandas as pd
import numpy as np
import datatable as dt
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NanoMod')

def preprocess_features(features_file, mod_sites_file, output_dir="NanoMod_tmp", min_reads=20, prefix="", mod_type='D'):
    """
    Preprocess features data to separate D-modified and unmodified sites
    
    Args:
        features_file: Path to features TSV file
        mod_sites_file: Path to known modification sites TSV file
        output_dir: Directory to save preprocessed files (default: NanoMod_tmp)
        min_reads: Minimum number of reads per site
        prefix: Prefix for output files (e.g., "train_" or "val_")
        mod_type: Modification type (default: D)
        
    Returns:
        Tuple of (modified_sites_file, unmodified_sites_file, stats)
    """
    start_time = time.time()
    logger.info(f"Starting preprocessing of {features_file}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths - include modification type to avoid conflicts
    modified_sites_file = os.path.join(output_dir, f"{prefix}{mod_type.lower()}_modified_sites.tsv")
    unmodified_sites_file = os.path.join(output_dir, f"{prefix}{mod_type.lower()}_unmodified_sites.tsv")
    
    # Load modification sites
    logger.info(f"Loading modification sites from {mod_sites_file}...")
    mod_sites = pd.read_csv(mod_sites_file, sep='\t')
    
    # Replace 'T' with 'U' in tRNA names for consistency
    mod_sites['trna_combined'] = mod_sites['trna_combined'].str.replace('T', 'U')
    
    # Create modification site positions mapping
    # For D modification, only mark the exact modified positions
    logger.info("Creating modification site ranges...")
    mod_positions = {}
    
    for _, row in mod_sites.iterrows():
        trna = row['trna_combined']
        pos = row['position']  # position column is already 0-based index
        mod_base = row['modified_base']
        
        if str(mod_base) == str(mod_type):  # support multiple modification types; compare as string to avoid type mismatch
            if trna not in mod_positions:
                mod_positions[trna] = {}
            
            # Only mark the exact modification position, do not include flanking bases
            mod_positions[trna][pos] = [pos]  # only the exact position
    
    # Count modification sites (support multiple modification types)
    mod_count = sum(1 for _, row in mod_sites.iterrows() if str(row['modified_base']) == str(mod_type))
    logger.info(f"Identified {mod_count} {mod_type} modification sites")
    
    # Load features data with datatable for speed
    logger.info(f"Loading features from {features_file}...")
    dt_data = dt.fread(features_file, sep='\t')
    features_df = dt_data.to_pandas()
    
    # Keep only required feature columns
    feature_cols = [
        'dtw.current', 'dtw.current_sd', 'dtw.length', 
        'dtw.amplitude', 'dtw.skewness', 'dtw.kurtosis', 'mistake'
    ]
    
    # Replace '*' with NaN and convert to numeric
    for col in feature_cols:
        features_df[col] = features_df[col].replace('*', np.nan)
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Drop rows with NaN in required feature columns
    features_df = features_df.dropna(subset=feature_cols)
    
    # Replace 'T' with 'U' in tRNA names for consistency
    features_df['seq.name'] = features_df['seq.name'].str.replace('T', 'U')
    
    # Prepare per-site data
    logger.info("Preparing site data with labels...")
    
    # Create modified and unmodified datasets
    modified_rows = []
    unmodified_rows = []
    
    # Get unique (tRNA, position) combinations
    trna_pos_combinations = features_df.groupby(['seq.name', 'seq.pos']).size().reset_index()[['seq.name', 'seq.pos']]
    total_sites = 0
    modified_sites = 0
    unmodified_sites = 0
    
    logger.info(f"Processing {len(trna_pos_combinations)} unique tRNA-position combinations...")
    
    for _, row in trna_pos_combinations.iterrows():
        trna = row['seq.name']
        pos = int(row['seq.pos'])
        
        # 获取该tRNA-position组合的所有行
        site_df = features_df[(features_df['seq.name'] == trna) & (features_df['seq.pos'] == pos)]
        unique_reads = site_df['aln.id'].unique()
        
        # Ensure there are enough reads
        if len(unique_reads) >= min_reads:
            total_sites += 1
            
            # Check whether this site is modified
            is_mod = False
            
            if trna in mod_positions and pos in mod_positions[trna]:
                is_mod = True
            
            # Add to modified or unmodified list according to modification status
            if is_mod:
                modified_rows.append(site_df)
                modified_sites += 1
            else:
                unmodified_rows.append(site_df)
                unmodified_sites += 1
    
    # Merge and save results
    logger.info("Saving preprocessed data...")
    
    if modified_rows:
        modified_df = pd.concat(modified_rows)
        # Add site_id column to match dataset.py format
        modified_df['site_id'] = modified_df['seq.name'] + '_' + modified_df['seq.pos'].astype(str)
        modified_df.to_csv(modified_sites_file, sep='\t', index=False)
        logger.info(f"Saved {len(modified_df)} rows of modified sites to {modified_sites_file}")
    else:
        logger.warning("No modified sites found!")
        # Create an empty file
        empty_df = pd.DataFrame(columns=features_df.columns)
        empty_df['site_id'] = ""
        empty_df.to_csv(modified_sites_file, sep='\t', index=False)
    
    if unmodified_rows:
        unmodified_df = pd.concat(unmodified_rows)
        # Add site_id column to match dataset.py format
        unmodified_df['site_id'] = unmodified_df['seq.name'] + '_' + unmodified_df['seq.pos'].astype(str)
        unmodified_df.to_csv(unmodified_sites_file, sep='\t', index=False)
        logger.info(f"Saved {len(unmodified_df)} rows of unmodified sites to {unmodified_sites_file}")
    else:
        logger.warning("No unmodified sites found!")
        # Create an empty file
        empty_df = pd.DataFrame(columns=features_df.columns)
        empty_df['site_id'] = ""
        empty_df.to_csv(unmodified_sites_file, sep='\t', index=False)
    
    # Compute and return statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    stats = {
        "total_sites": total_sites,
        "modified_sites": modified_sites,
        "unmodified_sites": unmodified_sites,
        "processing_time": elapsed_time
    }
    
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Stats: {stats}")
    
    return modified_sites_file, unmodified_sites_file, stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess tRNA features data")
    parser.add_argument("--features", required=True, help="Path to features TSV file")
    parser.add_argument("--mod-sites", required=True, help="Path to modification sites TSV file")
    parser.add_argument("--output-dir", default="NanoMod_tmp", help="Directory to save preprocessed files")
    parser.add_argument("--min-reads", type=int, default=20, help="Minimum reads per site")
    parser.add_argument("--prefix", default="", help="Prefix for output files")
    parser.add_argument("--mod-type", default="D", help="Modification type")
    
    args = parser.parse_args()
    
    preprocess_features(args.features, args.mod_sites, args.output_dir, args.min_reads, args.prefix, args.mod_type)
