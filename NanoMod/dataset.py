#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NanoMod-tRNA Dataset Module v0.9.6 (MIL with Noisy-OR pooling)

This module implements the Multiple Instance Learning (MIL) dataset
for tRNA modification detection.
It preserves all reads per site and returns:
- instances: [N, 6] tensor where N=num_reads (fixed at 30), 6=6 electrical signals
- structure: categorical index (0-8) computed by classify_trna_structure(seq.pos)
- mask: [N] boolean mask indicating valid reads (True) vs padding (False)

Padding strategy:
- If reads < 30: pad with zeros and set mask=False for padded positions
- If reads > 30: randomly sample 30 reads
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import datatable as dt
from itertools import product
from typing import Dict, List, Tuple, Union, Optional
import logging
import os
from .utils import classify_trna_structure  # structure classification function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NanoMod')

class MILDataset(Dataset):
    def __init__(self, mode: str, features_file: str, mod_sites_file: str, 
                 mod_type: str = 'D', num_reads: int = 30):
        """MIL dataset: keep all reads per site (fixed to num_reads).
        Each read has 6 electrical signal features.
        
        Args:
            mode: 'Train' or 'Val'.
            features_file: Path to features file.
            mod_sites_file: Path to modification sites file.
            mod_type: Modification type.
            num_reads: Fixed number of reads per site (default: 30).
        """
        super().__init__()
        self.mode = mode
        self.features_file = features_file
        self.mod_sites_file = mod_sites_file
        self.mod_type = mod_type
        self.num_reads = num_reads

        logger.info(f"Initializing {mode} MILDataset (preserving all reads, num_reads={num_reads}) ...")

        # Read features
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
        
        # Check 'mistake' column (used for mismatch rate and Bayesian model, not as model input)
        if 'mistake' not in df.columns:
            raise ValueError("features file must contain 'mistake' column")

        # Convert to numeric and drop missing values
        for col in signal_cols:
            if col in df.columns:
                df[col] = df[col].replace('*', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                raise ValueError(f"Missing required feature column: {col}")
        
        df['mistake'] = pd.to_numeric(df['mistake'], errors='coerce').fillna(0).astype(int)
        df = df.dropna(subset=signal_cols)
        
        # Create site_id (must be done before storing mistake_data)
        df['site_id'] = df['seq.name'].astype(str) + '_' + df['seq.pos'].astype(str)
        
        # Store 'mistake' column for mismatch rate calculation
        self.mistake_data = df[['site_id', 'mistake']].copy()

        # Read candidate modification sites file to build labels
        cand_df = pd.read_csv(mod_sites_file, sep='\t')
        cand_df['trna_combined'] = cand_df['trna_combined'].astype(str).str.replace('T', 'U')
        cand_df = cand_df[cand_df['modified_base'].astype(str) == str(mod_type)].copy()
        cand_df['position'] = pd.to_numeric(cand_df['position'], errors='coerce').fillna(0).astype(int)
        cand_df['site_id'] = cand_df['trna_combined'].astype(str) + '_' + cand_df['position'].astype(str)
        self.cand_ids = set(cand_df['site_id'].unique())
        
        logger.info(f"  Loaded {len(self.cand_ids)} candidate modification sites from {mod_sites_file}")

        # Group by site and keep all reads
        grouped = df.groupby(['site_id', 'seq.name', 'seq.pos'])
        
        self.site_ids = []
        self.seq_names = []
        self.seq_positions = []
        self.structures = []
        self.instances_list = []  # reads per site
        self.masks_list = []  # mask per site
        
        for (site_id, seq_name, seq_pos), group in grouped:
            # Extract 6D features: 6 signal dimensions only (without 'mistake')
            features = group[signal_cols].values.astype(np.float32)  # [actual_reads, 6]
            actual_num_reads = len(features)
            
            # Handle reads count
            if actual_num_reads > num_reads:
                # Randomly sample num_reads reads
                indices = np.random.choice(actual_num_reads, num_reads, replace=False)
                features = features[indices]
                mask = np.ones(num_reads, dtype=bool)
            elif actual_num_reads < num_reads:
                # Zero padding
                padded = np.zeros((num_reads, 6), dtype=np.float32)
                padded[:actual_num_reads, :] = features
                features = padded
                mask = np.zeros(num_reads, dtype=bool)
                mask[:actual_num_reads] = True
            else:
                # Exactly num_reads reads
                mask = np.ones(num_reads, dtype=bool)
            
            # Structure classification
            structure = classify_trna_structure(seq_pos)
            structure = int(np.clip(structure, 0, 8))
            
            # Save
            self.site_ids.append(site_id)
            self.seq_names.append(seq_name)
            self.seq_positions.append(seq_pos)
            self.structures.append(structure)
            self.instances_list.append(features)  # [num_reads, 6]
            self.masks_list.append(mask)  # [num_reads]
        
        logger.info(f"{mode} MILDataset built with {len(self.site_ids)} sites, "
                    f"each with {num_reads} reads (6 features per read)")

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        instances = torch.from_numpy(self.instances_list[idx])  # [num_reads, 6]
        structure = torch.tensor(self.structures[idx], dtype=torch.long)
        mask = torch.from_numpy(self.masks_list[idx])  # [num_reads]
        site_id = self.site_ids[idx]
        
        # Generate label based on whether site_id is in candidate set
        # 1.0 = in candidate file (potentially modified), 0.0 = not in candidate file (unmodified)
        y = torch.tensor(1.0 if site_id in self.cand_ids else 0.0, dtype=torch.float32)
        
        out = {
            'instances': instances,
            'structure': structure,
            'mask': mask,
            'site_id': site_id,
        }
        return out, y

def create_dataloaders(train_features_file, val_features_file, mod_sites_file, 
                      batch_size=256, num_instances=30, num_workers=128,
                      use_preprocessed=True, preprocessed_dir="NanoMod_tmp", mod_type='D'):
    """
    Create train and validation dataloaders for MIL
    
    Args:
        train_features_file: Path to training features file
        val_features_file: Path to validation features file
        mod_sites_file: Path to known modification sites file
        batch_size: Batch size (default: 256)
        num_instances: Number of reads per site (default: 30)
        num_workers: Number of worker threads for data loading (default: 128)
        use_preprocessed: Unused, kept for compatibility
        preprocessed_dir: Unused, kept for compatibility
        mod_type: Modification type (default: 'D')
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    # MIL collate: keep instances, masks, and site_ids
    def custom_collate_fn(batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        instances = torch.stack([x['instances'] for x in inputs], dim=0)  # [B, N, 6]
        struct = torch.stack([x['structure'] for x in inputs], dim=0)  # [B]
        masks = torch.stack([x['mask'] for x in inputs], dim=0)  # [B, N]
        site_ids = [x['site_id'] for x in inputs]

        batch_data = {
            'instances': instances,
            'structure': struct,
            'mask': masks,
            'site_ids': site_ids,
        }
        batch_labels = torch.stack(labels)
        return batch_data, batch_labels
    
    # 创建MIL数据集
    logger.info(f"Creating training MILDataset (num_reads={num_instances})...")
    train_dataset = MILDataset(mode="Train", features_file=train_features_file, 
                               mod_sites_file=mod_sites_file, mod_type=mod_type, 
                               num_reads=num_instances)

    logger.info(f"Creating validation MILDataset (num_reads={num_instances})...")
    val_dataset = MILDataset(mode="Val", features_file=val_features_file, 
                             mod_sites_file=mod_sites_file, mod_type=mod_type, 
                             num_reads=num_instances)
    
    # Create dataloaders
    logger.info(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    
    logger.info("DataLoaders created successfully.")
    return train_loader, val_loader, train_dataset, val_dataset
