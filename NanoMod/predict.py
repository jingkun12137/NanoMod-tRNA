#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NanoMod-tRNA Prediction Module v0.9.6 (Attention MIL)

This module implements tRNA modification prediction based on Attention MIL.
It performs open-world prediction without requiring a candidate site list.

Model Input:
- 6 electrical signal features per read (default 30 reads per site)
  * dtw.current: DTW-aligned current signal
  * dtw.current_sd: Standard deviation of current
  * dtw.length: Signal length
  * dtw.amplitude: Signal amplitude
  * dtw.skewness: Signal skewness
  * dtw.kurtosis: Signal kurtosis
- tRNA structure embedding (16-dim)

Output:
- Site-level modification probability (0-1)
- Binary classification using threshold 0.5
"""

import os
import numpy as np
import torch
# Add this line to avoid the "Too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import logging
import argparse
import datatable as dt
from collections import defaultdict
from .utils import classify_trna_structure

# Set up logging
logger = logging.getLogger(__name__)

from .model import NanoMod



def pure_predict(model_path, features_file, site_output_file,
                 num_instances=30, kmer_nums=781, hidden_dim=128,
                 num_workers=0, mod_type='D',
                 save_read_level=False, read_output_file=None):
    """Attention MIL prediction function using a trained model.

    Args:
        model_path: Path to trained model file.
        features_file: Path to features TSV file for prediction.
        site_output_file: Full path to site-level prediction TSV.
        num_instances: Number of reads per site (default: 30).
        kmer_nums: Compatibility parameter (unused).
        hidden_dim: Compatibility parameter (unused).
        num_workers: DataLoader worker count (unused in current implementation).
        mod_type: Modification type (used for file naming).
        save_read_level: Whether to also output read-level modification probabilities.
        read_output_file: Full path to read-level prediction TSV (required if save_read_level=True).

    Returns:
        dict: {
          'site_predictions': Site-level prediction probabilities,
          'site_ids': Site ID list,
          'output_file': Site-level TSV path,
        }
    """
    # Resolve and create output paths
    if site_output_file is None:
        site_output_file = os.path.join('NanoMod_output', f'{mod_type}_modification_predictions.tsv')
    site_output_dir = os.path.dirname(site_output_file) or '.'
    os.makedirs(site_output_dir, exist_ok=True)

    if save_read_level:
        if not read_output_file:
            raise ValueError('read_output_file must be provided when save_read_level is True')
        read_output_dir = os.path.dirname(read_output_file) or '.'
        os.makedirs(read_output_dir, exist_ok=True)
    
    # Load features
    logger.info(f"Loading features from {features_file}...")
    dt_data = dt.fread(features_file, sep='\t')
    df = dt_data.to_pandas()

    # Normalize names
    if 'seq.name' not in df.columns or 'seq.pos' not in df.columns:
        raise ValueError("features file must contain 'seq.name' and 'seq.pos'")
    df['seq.name'] = df['seq.name'].astype(str).str.replace('T', 'U')
    df['seq.pos'] = pd.to_numeric(df['seq.pos'], errors='coerce').fillna(0).astype(int)

    # Required signal columns (6 dimensions, without 'mistake')
    signal_cols = [
        'dtw.current', 'dtw.current_sd', 'dtw.length',
        'dtw.amplitude', 'dtw.skewness', 'dtw.kurtosis'
    ]
    
    for col in signal_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")
        df[col] = df[col].replace('*', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=signal_cols)

    # Build site_id
    df['site_id'] = df['seq.name'].astype(str) + '_' + df['seq.pos'].astype(str)

    # Group by site and preserve all reads (MIL)
    grouped = df.groupby(['site_id', 'seq.name', 'seq.pos'])
    
    site_ids = []
    seq_names = []
    seq_positions = []
    structures = []
    instances_list = []  # Each site's reads
    masks_list = []  # Each site's mask
    
    for (site_id, seq_name, seq_pos), group in grouped:
        # Extract 6D features: 6 electrical signals only
        features = group[signal_cols].values.astype(np.float32)  # [actual_reads, 6]
        actual_num_reads = len(features)
        
        # Handle reads count
        if actual_num_reads > num_instances:
            # Randomly sample num_instances reads
            indices = np.random.choice(actual_num_reads, num_instances, replace=False)
            features = features[indices]
            mask = np.ones(num_instances, dtype=bool)
        elif actual_num_reads < num_instances:
            # Zero padding
            padded = np.zeros((num_instances, 6), dtype=np.float32)
            padded[:actual_num_reads, :] = features
            features = padded
            mask = np.zeros(num_instances, dtype=bool)
            mask[:actual_num_reads] = True
        else:
            # Exactly num_instances reads
            mask = np.ones(num_instances, dtype=bool)
        
        # Structure classification
        structure = classify_trna_structure(seq_pos)
        structure = int(np.clip(structure, 0, 8))
        
        # Save
        site_ids.append(site_id)
        seq_names.append(seq_name)
        seq_positions.append(seq_pos)
        structures.append(structure)
        instances_list.append(features)  # [num_instances, 6]
        masks_list.append(mask)  # [num_instances]
    
    logger.info(f"Prepared {len(site_ids)} sites for prediction, each with {num_instances} reads (6 features per read)")

    if len(site_ids) == 0:
        logger.warning("No sites found!")
        return {"site_ids": [], "site_predictions": [], "output_file": site_output_file}
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Attention MIL model
    logger.info(f"Loading Attention MIL model from {model_path}")
    try:
        model = NanoMod(input_dim=6, hidden_dim=128, structure_emb_dim=16,
                       dropout_rate=0.3, num_instances=num_instances)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"site_ids": [], "site_predictions": [], "output_file": site_output_file}
    
    model.eval()
    
    # Run predictions in mini-batches
    logger.info("Running Attention MIL predictions...")
    site_preds = np.zeros((len(site_ids),), dtype=np.float32)
    read_level_rows = [] if save_read_level else None
    bs = 256  # Batch size for prediction
    
    with torch.no_grad():
        for start in range(0, len(site_ids), bs):
            end = min(start + bs, len(site_ids))
            
            # Prepare batch
            instances_batch = torch.from_numpy(np.stack(instances_list[start:end])).to(device)  # [B, N, 6]
            struct_batch = torch.tensor(structures[start:end], dtype=torch.long).to(device)  # [B]
            masks_batch = torch.from_numpy(np.stack(masks_list[start:end])).to(device)  # [B, N]
            
            # Forward pass
            batch_input = {
                'instances': instances_batch,
                'structure': struct_batch,
                'mask': masks_batch
            }

            if save_read_level:
                outputs, read_probs = model(batch_input, return_read_probs=True)
                outputs_np = outputs.detach().cpu().numpy()
                read_probs_np = read_probs.detach().cpu().numpy()
                masks_np = masks_batch.cpu().numpy()

                for bi in range(outputs_np.shape[0]):
                    global_idx = start + bi
                    sid = site_ids[global_idx]
                    sname = seq_names[global_idx]
                    spos = seq_positions[global_idx]
                    sstruct = structures[global_idx]
                    for ri in range(read_probs_np.shape[1]):
                        if not masks_np[bi, ri]:
                            continue
                        rp = float(read_probs_np[bi, ri])
                        read_level_rows.append({
                            'site_id': sid,
                            'seq.name': sname,
                            'seq.pos': spos,
                            'structure': sstruct,
                            'prediction': rp,
                            'modified': int(rp > 0.1),
                        })
            else:
                outputs_np = model(batch_input).detach().cpu().numpy()

            site_preds[start:end] = outputs_np
    
    # Build prediction results
    preds = site_preds.copy()
    
    # Create results dictionary
    results = {
        'site_predictions': preds.tolist(),
        'site_ids': site_ids
    }
    
    # Save site-level predictions to file
    out_df = pd.DataFrame({
        'site_id': site_ids,
        'seq.name': seq_names,
        'seq.pos': seq_pos,
        'structure': structures,
        'prediction': preds.tolist(),
    })
    out_df['modified'] = (out_df['prediction'].values >= 0.5).astype(int)
    out_df.to_csv(site_output_file, sep='\t', index=False)

    logger.info(f"Predictions saved to {site_output_file}")

    if save_read_level and read_level_rows:
        read_df = pd.DataFrame(read_level_rows, columns=['site_id', 'seq.name', 'seq.pos', 'structure', 'prediction', 'modified'])
        read_df.to_csv(read_output_file, sep='\t', index=False)
        logger.info(f"Read-level predictions saved to {read_output_file}")

    return {
        'site_predictions': preds.tolist(),
        'site_ids': site_ids,
        'output_file': site_output_file
    }

def predict(model, loader, device='cuda'):
    """
    Make predictions with the trained model
    
    Args:
        model: Trained NanoMod model
        loader: DataLoader with test data
        device: Device to run predictions on
    
    Returns:
        Dictionary with site predictions, read predictions, and attention weights
    """
    model.eval()  # Set model to evaluation mode
    
    # Lists to store predictions
    site_preds = []
    read_preds = []
    attention_weights = []
    site_ids = []
    
    # Ensure model is on the correct device
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
    model = model.to(device)
    
    try:
        with torch.no_grad():
            for i, (X, y) in enumerate(loader):
                try:
                    # Move data to device
                    continuous_data = X['continuous'].to(device)
                    mistake_data = X['mistake'].to(device)
                    structure_data = X['structure'].to(device)
                    
                    # Forward pass
                    outputs = model({
                        'continuous': continuous_data,
                        'mistake': mistake_data,
                        'structure': structure_data
                    })
                    
                    # Save site-level predictions
                    batch_site_preds = outputs.cpu().numpy()
                    
                    # Read-level predictions
                    batch_read_preds = model.read_level_pred.cpu().numpy() if hasattr(model, 'read_level_pred') and model.read_level_pred is not None else np.ones((continuous_data.size(0), 1)) * 0.5
                    
                    # Attention weights
                    batch_attention = model.attention_weights.cpu().numpy() if hasattr(model, 'attention_weights') and model.attention_weights is not None else np.ones_like(batch_read_preds)
                    
                    site_preds.extend(batch_site_preds.flatten())
                    read_preds.append(batch_read_preds)
                    attention_weights.append(batch_attention)
                    
                    if hasattr(loader.dataset, 'get_site_ids') and callable(getattr(loader.dataset, 'get_site_ids')):
                        batch_site_ids = loader.dataset.get_site_ids(i)
                        site_ids.extend(batch_site_ids)
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
    
    # Process read_preds and attention_weights
    if read_preds:
        try:
            read_preds = np.vstack(read_preds) if len(read_preds) > 0 else np.array([])
            attention_weights = np.vstack(attention_weights) if len(attention_weights) > 0 else np.array([])
        except Exception as e:
            logger.error(f"Error stacking prediction arrays: {e}")
            read_preds = np.array([])
            attention_weights = np.array([])
    else:
        read_preds = np.array([])
        attention_weights = np.array([])
            
    # Combine results in a dictionary
    results = {
        'site_predictions': np.array(site_preds),
        'read_predictions': read_preds,
        'attention_weights': attention_weights,
        'site_ids': site_ids if site_ids else None
    }
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NanoMod-tRNA prediction script')
    
    # Data parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--features-file', type=str, required=True, help='Features file for prediction')
    parser.add_argument('--output-dir', type=str, default='NanoMod_output', help='Output directory')
    
    # Model parameters
    parser.add_argument('--mod-type', type=str, default='D', help='Modification type')
    parser.add_argument('--kmer-nums', type=int, default=781, help='Number of possible kmers for embedding')
    parser.add_argument('--min-reads', type=int, default=20, help='Minimum reads per site')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Run prediction
    results = pure_predict(
        args.model_path, args.features_file, args.output_dir,
        num_instances=args.min_reads, kmer_nums=args.kmer_nums,
        num_workers=args.num_workers, mod_type=args.mod_type
    )
    
    logger.info(f"Prediction completed! Results: {len(results.get('site_ids', []))} sites processed")
    
    if results:
        print(f"Prediction completed! Results saved to: {results.get('output_file', 'N/A')}")
    else:
        print("Prediction failed!")


if __name__ == '__main__':
    main()
