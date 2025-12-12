#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NanoMod-tRNA Training Module v0.9.6 (Attention MIL with Adaptive Strategy + Structure-Aware Balancing)

This module trains an Attention-based Multiple Instance Learning (MIL) model
with an adaptive training strategy and structure-aware balanced sampling.

Training modes:
1. Mode A (high mismatch rate): Bayesian soft labels
   - Triggered when the mismatch rate at candidate sites > 1.5 × global mismatch rate.
   - A Bayesian model estimates the true modification probability from mismatch patterns.
   - Training: use BCE loss with soft labels.
   - Validation: MSE + AUROC/AUPRC (using the median as threshold).

2. Mode B (low mismatch rate): hard labels
   - Triggered when mismatch rate differences are not significant.
   - Training: use BCE loss with hard labels (candidate=1, non-candidate=0).
   - Validation: AUROC/AUPRC/Accuracy/Precision/Recall/F1.

Model architecture:
- Input: each read has 6 electrical signal features
         (dtw.current, dtw.current_sd, dtw.length,
          dtw.amplitude, dtw.skewness, dtw.kurtosis).
- tRNA structure embedding: 16-dimensional positional encoding.
- Attention mechanism: learns importance weights for each read.
- Output: site-level modification probability.

Structure-aware balanced sampling (added in v0.9.6):
- Perform balancing within each tRNA structure.
  * If the number of positive sites > negative sites: keep all sites in that structure.
  * If the number of positive sites ≤ negative sites: perform 1:1 positive/negative balancing.
- For structures without modified sites, keep 10% of sites as negative examples (at least 1 site).
- Avoid letting the model learn trivial position bias (e.g. "structure=3 implies modification").

Training outputs:
- Best model checkpoint: {mod_type}_best_model.pt
- Training history: {mod_type}_training_history.json
- Training curves: {mod_type}_training_curves.pdf
- ROC/PR curves: {mod_type}_AUROC_AUPRC.pdf
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import datetime
import logging
import random
import matplotlib.pyplot as plt
import argparse
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import datatable as dt

from .model import NanoMod
from .dataset import create_dataloaders
from .bayes import estimate_bayes_params, compute_bayes_posteriors, save_params

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NanoMod')


def compute_mismatch_rates(features_file, mod_sites_file, mod_type):
    """
    Compute mismatch rates for candidate sites and all sites.

    Args:
        features_file: Path to features file.
        mod_sites_file: Path to modification sites file.
        mod_type: Modification type.

    Returns:
        tuple: (average mismatch rate for candidate sites,
                average mismatch rate for all sites,
                number of candidate sites,
                total number of sites)
    """
    # Read features file
    dt_data = dt.fread(features_file, sep='\t')
    df = dt_data.to_pandas()
    
    # Normalize names
    df['seq.name'] = df['seq.name'].astype(str).str.replace('T', 'U')
    df['seq.pos'] = pd.to_numeric(df['seq.pos'], errors='coerce').fillna(0).astype(int)
    df['mistake'] = pd.to_numeric(df['mistake'], errors='coerce').fillna(0).astype(int)
    df['site_id'] = df['seq.name'].astype(str) + '_' + df['seq.pos'].astype(str)
    
    # Read candidate modification sites
    cand_df = pd.read_csv(mod_sites_file, sep='\t')
    cand_df['trna_combined'] = cand_df['trna_combined'].astype(str).str.replace('T', 'U')
    cand_df = cand_df[cand_df['modified_base'].astype(str) == str(mod_type)].copy()
    cand_df['position'] = pd.to_numeric(cand_df['position'], errors='coerce').fillna(0).astype(int)
    cand_df['site_id'] = cand_df['trna_combined'].astype(str) + '_' + cand_df['position'].astype(str)
    cand_ids = set(cand_df['site_id'].unique())
    
    # Aggregate mismatch information per site
    site_stats = df.groupby('site_id').agg({
        'mistake': ['sum', 'count']
    }).reset_index()
    site_stats.columns = ['site_id', 'mismatch_count', 'total_reads']
    site_stats['mismatch_rate'] = site_stats['mismatch_count'] / site_stats['total_reads']
    
    # Compute mismatch rates for candidate sites and all sites
    cand_sites = site_stats[site_stats['site_id'].isin(cand_ids)]
    
    cand_mismatch_rate = cand_sites['mismatch_rate'].mean() if len(cand_sites) > 0 else 0.0
    overall_mismatch_rate = site_stats['mismatch_rate'].mean()
    
    return cand_mismatch_rate, overall_mismatch_rate, len(cand_sites), len(site_stats)


def train_model(train_features_file, val_features_file, mod_sites_file, 
                model_save_dir, batch_size=256, num_epochs=300, learning_rate=0.003,
                num_instances=30, kmer_nums=781, dropout_rate=0.3,
                num_workers=128, mod_type='D', mismatch_threshold=1.5):
    """
    Train NanoMod-tRNA model with Attention MIL (Adaptive Strategy)
    
    Args:
        train_features_file: Path to training features file
        val_features_file: Path to validation features file
        mod_sites_file: Path to known modification sites file
        model_save_dir: Directory to save model and results
        batch_size: Batch size for training (default: 256)
        num_epochs: Number of epochs (default: 300)
        learning_rate: Learning rate (default: 0.003)
        num_instances: Number of reads per site (default: 30)
        kmer_nums: Unused, kept for compatibility (default: 781)
        dropout_rate: Dropout rate (default: 0.3)
        num_workers: Number of worker threads for data loading (default: 128)
        mod_type: Modification type (default: 'D')
        mismatch_threshold: Threshold for mode selection (default: 1.5)
    """
    # Step 1: Compute mismatch rates and decide training mode
    logger.info("="*60)
    logger.info("Step 1: Analyzing mismatch rates to determine training mode...")
    logger.info("="*60)
    
    cand_mismatch_rate, overall_mismatch_rate, n_cand, n_total = compute_mismatch_rates(
        train_features_file, mod_sites_file, mod_type
    )
    
    logger.info(f"Candidate sites: {n_cand}/{n_total}")
    logger.info(f"Candidate site average mismatch rate: {cand_mismatch_rate:.4f}")
    logger.info(f"Overall average mismatch rate: {overall_mismatch_rate:.4f}")
    logger.info(f"Mismatch rate ratio: {cand_mismatch_rate/overall_mismatch_rate if overall_mismatch_rate > 0 else 0:.2f}")
    
    # 判断使用哪种模式
    use_bayesian = (cand_mismatch_rate > mismatch_threshold * overall_mismatch_rate)
    
    if use_bayesian:
        logger.info(f"✓ Mismatch rate ratio > {mismatch_threshold}, using Mode A (Bayesian soft labels)")
        training_mode = "bayesian"
    else:
        logger.info(f"✓ Mismatch rate ratio ≤ {mismatch_threshold}, using Mode B (Hard labels)")
        training_mode = "hard"
    
    logger.info("="*60)
    
    # Step 2: 创建数据加载器
    logger.info("Step 2: Creating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        train_features_file, val_features_file, mod_sites_file,
        batch_size=batch_size, num_instances=num_instances, num_workers=num_workers,
        mod_type=mod_type
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} sites")
    logger.info(f"Validation dataset: {len(val_dataset)} sites")
    
    # Read candidate site information (for balanced sampling)
    cand_df = pd.read_csv(mod_sites_file, sep='\t')
    cand_df['trna_combined'] = cand_df['trna_combined'].astype(str).str.replace('T', 'U')
    cand_df = cand_df[cand_df['modified_base'].astype(str) == str(mod_type)].copy()
    cand_df['position'] = pd.to_numeric(cand_df['position'], errors='coerce').fillna(0).astype(int)
    cand_df['site_id'] = cand_df['trna_combined'].astype(str) + '_' + cand_df['position'].astype(str)
    cand_ids = set(cand_df['site_id'].unique())
    
    # Define structure-aware balanced sampling function
    def balance_dataset_structure_aware(dataset, loader, cand_ids, dataset_name="Dataset", negative_ratio=0.10, num_reads_per_site=30):
        """Perform structure-aware balanced sampling on the dataset.

        Strategy:
        1. For each tRNA structure, perform balancing:
           - If the number of positive sites > negative sites: keep all sites in that structure.
           - If the number of positive sites ≤ negative sites: perform 1:1 positive/negative balancing.
        2. For structures without modified sites, keep negative_ratio of negative sites
           (i.e. max(1, int(n_neg × negative_ratio)) negative sites).
        3. Keep the original number of samples per structure; only balance positives vs negatives within each structure.

        Args:
            dataset: Dataset object.
            loader: Original DataLoader.
            cand_ids: Set of candidate modification site IDs.
            dataset_name: Dataset name (for logging).
            negative_ratio: Proportion of negative sites to keep for structures without modifications (default 0.10, i.e. 10%%).
            num_reads_per_site: Number of reads per site (default 30).

        Returns:
            balanced_loader: Balanced DataLoader.
        """
        present_site_ids = getattr(dataset, 'site_ids', [])
        present_structures = getattr(dataset, 'structures', [])
        
        if not isinstance(present_site_ids, list) or len(present_site_ids) == 0:
            logger.warning(f"{dataset_name}: No site_ids found, skipping balancing")
            return loader
        
        if not isinstance(present_structures, list) or len(present_structures) != len(present_site_ids):
            logger.warning(f"{dataset_name}: Structure information missing or inconsistent, skipping balancing")
            return loader
        
        # Group by structure
        structure_groups = {}
        for idx, (site_id, structure) in enumerate(zip(present_site_ids, present_structures)):
            if structure not in structure_groups:
                structure_groups[structure] = {'positive': [], 'negative': []}
            
            if site_id in cand_ids:
                structure_groups[structure]['positive'].append(idx)
            else:
                structure_groups[structure]['negative'].append(idx)
        
        logger.info(f"{dataset_name} - Structure-aware balancing:")
        logger.info(f"  Found {len(structure_groups)} unique structures")
        
        # 对每个structure进行平衡采样
        selected_indices = []
        total_positive = 0
        total_negative = 0
        
        for structure in sorted(structure_groups.keys()):
            pos_indices = structure_groups[structure]['positive']
            neg_indices = structure_groups[structure]['negative']
            
            n_pos = len(pos_indices)
            n_neg = len(neg_indices)
            
            if n_pos > 0:
                # Structures with modified sites
                if n_pos > n_neg:
                    # Positive sites outnumber negative sites: keep all
                    selected_pos = pos_indices
                    selected_neg = neg_indices
                    
                    selected_indices.extend(selected_pos)
                    selected_indices.extend(selected_neg)
                    
                    total_positive += len(selected_pos)
                    total_negative += len(selected_neg)
                    
                    logger.info(f"  Structure {structure}: {len(selected_pos)} positive + {len(selected_neg)} negative "
                               f"(positive > negative, kept all from {n_pos} pos, {n_neg} neg)")
                else:
                    # Positive sites ≤ negative sites: 1:1 balancing
                    target_n = min(n_pos, n_neg)
                    
                    if n_pos > target_n:
                        selected_pos = random.sample(pos_indices, target_n)
                    else:
                        selected_pos = pos_indices
                    
                    if n_neg > target_n:
                        selected_neg = random.sample(neg_indices, target_n)
                    else:
                        selected_neg = neg_indices
                    
                    selected_indices.extend(selected_pos)
                    selected_indices.extend(selected_neg)
                    
                    total_positive += len(selected_pos)
                    total_negative += len(selected_neg)
                    
                    logger.info(f"  Structure {structure}: {len(selected_pos)} positive + {len(selected_neg)} negative "
                               f"(1:1 balanced from {n_pos} pos, {n_neg} neg)")
            else:
                # Structures without modifications: keep negative_ratio of negative sites
                target_n = max(1, int(n_neg * negative_ratio))
                
                if n_neg > target_n:
                    selected_neg = random.sample(neg_indices, target_n)
                else:
                    selected_neg = neg_indices
                
                selected_indices.extend(selected_neg)
                total_negative += len(selected_neg)
                
                logger.info(f"  Structure {structure}: 0 positive + {len(selected_neg)} negative "
                           f"(no modification sites, kept {len(selected_neg)} sites = {len(selected_neg) * num_reads_per_site} reads, "
                           f"{len(selected_neg)/n_neg*100:.1f}% from {n_neg} neg)")
        
        logger.info(f"{dataset_name} - Total balanced: {len(selected_indices)} sites "
                   f"({total_positive} positive + {total_negative} negative)")
        
        # Save original collate_fn
        collate_fn = loader.collate_fn
        
        # Create balanced DataLoader
        is_train = (dataset_name == "Training")
        balanced_loader = DataLoader(
            Subset(dataset, selected_indices),
            batch_size=batch_size,
            shuffle=is_train,  # 训练集shuffle，验证集不shuffle
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        return balanced_loader
    
    # Apply structure-aware balancing to training set
    logger.info("="*60)
    logger.info("Structure-Aware Balancing for Training Set...")
    logger.info("="*60)
    train_loader = balance_dataset_structure_aware(train_dataset, train_loader, cand_ids, "Training", negative_ratio=0.10, num_reads_per_site=num_instances)
    
    # Apply structure-aware balancing to validation set
    logger.info("="*60)
    logger.info("Structure-Aware Balancing for Validation Set...")
    logger.info("="*60)
    val_loader = balance_dataset_structure_aware(val_dataset, val_loader, cand_ids, "Validation", negative_ratio=0.10, num_reads_per_site=num_instances)
    
    # Step 3: If using Bayesian mode, compute soft labels
    train_bayes_map = {}
    val_bayes_map = {}
    
    if use_bayesian:
        logger.info("="*60)
        logger.info("Step 3: Computing Bayesian soft labels...")
        logger.info("="*60)
        
        # Estimate Bayesian parameters
        logger.info("Estimating Bayesian parameters from training set...")
        bayes_params = estimate_bayes_params(train_features_file, mod_sites_file, mod_type)
        params_path = os.path.join(model_save_dir, f'{mod_type}_bayes_params.json')
        save_params(bayes_params, params_path)
        logger.info(f"Bayesian parameters: π={bayes_params['pi']:.4f}, "
                   f"α1={bayes_params['alpha1']:.2f}, β1={bayes_params['beta1']:.2f}, "
                   f"α0={bayes_params['alpha0']:.2f}, β0={bayes_params['beta0']:.2f}")
        
        # Compute Bayesian posteriors for training and validation sets
        logger.info("Computing Bayesian posteriors for training sites...")
        train_bayes_df = compute_bayes_posteriors(train_features_file, mod_sites_file, mod_type, bayes_params)
        train_bayes_map = {sid: prob for sid, prob in zip(train_bayes_df['site_id'], train_bayes_df['bayes_prob'])}
        
        logger.info("Computing Bayesian posteriors for validation sites...")
        val_bayes_df = compute_bayes_posteriors(val_features_file, mod_sites_file, mod_type, bayes_params)
        val_bayes_map = {sid: prob for sid, prob in zip(val_bayes_df['site_id'], val_bayes_df['bayes_prob'])}
        
        logger.info(f"Training set: {len(train_bayes_map)} sites with Bayesian labels")
        logger.info(f"Validation set: {len(val_bayes_map)} sites with Bayesian labels")
        logger.info("="*60)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Attention MIL model
    model = NanoMod(input_dim=6, hidden_dim=128, structure_emb_dim=16,
                   dropout_rate=dropout_rate, num_instances=num_instances)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler (only in last 30% epochs)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, 
                                  threshold=0.0001, min_lr=1e-6)
    lr_decay_start_epoch = int(num_epochs * 0.7)
    
    # Training stats
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    start_time = time.time()
    
    logger.info(f"Starting training: {num_epochs} epochs, batch size: {batch_size}, learning rate: {learning_rate}")
    logger.info(f"Training mode: {training_mode}")
    logger.info("-" * 60)
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_metrics': []
    }
    
    # Create CSV file for saving metrics
    metrics_csv_path = os.path.join(model_save_dir, f'{mod_type}_training_metrics.csv')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # 两种模式使用相同的指标列
    metrics_columns = ['Epoch', 'Train Loss', 'Val Loss', 'Val Accuracy', 'Val Precision', 
                      'Val Recall', 'Val F1', 'Val AUROC', 'Val AUPRC']
    if use_bayesian:
        metrics_columns.insert(3, 'Val MSE')  # 贝叶斯模式额外添加MSE
    metrics_df = pd.DataFrame(columns=metrics_columns)
    metrics_df.to_csv(metrics_csv_path, index=False)
        
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        batch_count = len(train_loader)
        progress_interval = max(1, batch_count // 10)
        
        for batch_idx, (X, y_dummy) in enumerate(train_loader):
            # Move data to device
            X['instances'] = X['instances'].to(device)
            X['structure'] = X['structure'].to(device)
            X['mask'] = X['mask'].to(device)
            site_ids = X['site_ids']
            
            # Build labels based on training mode
            if use_bayesian:
                # Mode A: Use Bayesian soft labels
                y = torch.tensor([train_bayes_map.get(sid, 0.0) for sid in site_ids], 
                               dtype=torch.float32, device=device)
            else:
                # Mode B: Use hard labels
                y = torch.tensor([1.0 if sid in cand_ids else 0.0 for sid in site_ids], 
                               dtype=torch.float32, device=device)
            
            # Forward pass
            outputs = model(X)
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % progress_interval == 0:
                logger.info(f"  Batch {batch_idx+1}/{batch_count} - {(batch_idx+1)*100/batch_count:.1f}% complete")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        history['train_losses'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        val_site_ids = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (X, y_dummy) in enumerate(val_loader):
                X['instances'] = X['instances'].to(device)
                X['structure'] = X['structure'].to(device)
                X['mask'] = X['mask'].to(device)
                site_ids = X['site_ids']
                
                # Build labels based on training mode
                if use_bayesian:
                    # Mode A: Use Bayesian soft labels
                    y = torch.tensor([val_bayes_map.get(sid, 0.0) for sid in site_ids], 
                                   dtype=torch.float32, device=device)
                else:
                    # Mode B: Use hard labels
                    y = torch.tensor([1.0 if sid in cand_ids else 0.0 for sid in site_ids], 
                                   dtype=torch.float32, device=device)
                
                outputs = model(X)
                
                val_loss += criterion(outputs, y).item()
                val_preds.extend(outputs.detach().cpu().numpy().tolist())
                val_labels.extend(y.detach().cpu().numpy().tolist())
                val_site_ids.extend(site_ids)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        history['val_losses'].append(val_loss)
        
        # Update learning rate scheduler
        if epoch >= lr_decay_start_epoch:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate metrics based on training mode
        val_preds_np = np.array(val_preds)
        val_labels_np = np.array(val_labels)
        
        if use_bayesian:
            # Mode A: Use the median of Bayesian posteriors as the binarization threshold
            mse = np.mean((val_preds_np - val_labels_np) ** 2)
            
            # Binary labels using median of Bayesian posteriors as threshold
            bayes_median = np.median(val_labels_np)
            val_labels_binary = (val_labels_np > bayes_median).astype(int)
            val_preds_binary = (val_preds_np > 0.5).astype(int)
            
            # Compute all classification metrics
            accuracy = accuracy_score(val_labels_binary, val_preds_binary)
            precision = precision_score(val_labels_binary, val_preds_binary, zero_division=0)
            recall = recall_score(val_labels_binary, val_preds_binary, zero_division=0)
            f1 = f1_score(val_labels_binary, val_preds_binary, zero_division=0)
            
            try:
                auroc = roc_auc_score(val_labels_binary, val_preds_np) if len(np.unique(val_labels_binary)) > 1 else 0.0
            except:
                auroc = 0.0
            try:
                auprc = average_precision_score(val_labels_binary, val_preds_np) if len(np.unique(val_labels_binary)) > 1 else 0.0
            except:
                auprc = 0.0
            
            epoch_metrics = {
                'mse': mse,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auroc': auroc,
                'auprc': auprc,
                'bayes_median': bayes_median
            }
        else:
            # Mode B: Use hard labels with 0.5 as threshold
            val_preds_binary = (val_preds_np > 0.5).astype(int)
            val_labels_binary = val_labels_np.astype(int)
            
            accuracy = accuracy_score(val_labels_binary, val_preds_binary)
            precision = precision_score(val_labels_binary, val_preds_binary, zero_division=0)
            recall = recall_score(val_labels_binary, val_preds_binary, zero_division=0)
            f1 = f1_score(val_labels_binary, val_preds_binary, zero_division=0)
            
            try:
                auroc = roc_auc_score(val_labels_binary, val_preds_np) if len(np.unique(val_labels_binary)) > 1 else 0.0
            except:
                auroc = 0.0
            try:
                auprc = average_precision_score(val_labels_binary, val_preds_np) if len(np.unique(val_labels_binary)) > 1 else 0.0
            except:
                auprc = 0.0
            
            epoch_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auroc': auroc,
                'auprc': auprc
            }
        history['val_metrics'].append(epoch_metrics)
        
        # Log metrics based on training mode
        epoch_time = time.time() - epoch_start_time
        
        if use_bayesian:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val MSE: {epoch_metrics['mse']:.4f}, "
                       f"Val Accuracy: {epoch_metrics['accuracy']:.4f}, "
                       f"Val Precision: {epoch_metrics['precision']:.4f}, "
                       f"Val Recall: {epoch_metrics['recall']:.4f}, "
                       f"Val F1: {epoch_metrics['f1']:.4f}, "
                       f"Val AUROC: {epoch_metrics['auroc']:.4f}, "
                       f"Val AUPRC: {epoch_metrics['auprc']:.4f}, "
                       f"Bayes Median: {epoch_metrics['bayes_median']:.4f}, "
                       f"Learning Rate: {current_lr:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Save metrics to CSV
            epoch_metrics_row = pd.DataFrame({
                'Epoch': [epoch+1],
                'Train Loss': [train_loss],
                'Val Loss': [val_loss],
                'Val MSE': [epoch_metrics['mse']],
                'Val Accuracy': [epoch_metrics['accuracy']],
                'Val Precision': [epoch_metrics['precision']],
                'Val Recall': [epoch_metrics['recall']],
                'Val F1': [epoch_metrics['f1']],
                'Val AUROC': [epoch_metrics['auroc']],
                'Val AUPRC': [epoch_metrics['auprc']]
            })
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Accuracy: {epoch_metrics['accuracy']:.4f}, "
                       f"Val Precision: {epoch_metrics['precision']:.4f}, "
                       f"Val Recall: {epoch_metrics['recall']:.4f}, "
                       f"Val F1: {epoch_metrics['f1']:.4f}, "
                       f"Val AUROC: {epoch_metrics['auroc']:.4f}, "
                       f"Val AUPRC: {epoch_metrics['auprc']:.4f}, "
                       f"Learning Rate: {current_lr:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Save metrics to CSV
            epoch_metrics_row = pd.DataFrame({
                'Epoch': [epoch+1],
                'Train Loss': [train_loss],
                'Val Loss': [val_loss],
                'Val Accuracy': [epoch_metrics['accuracy']],
                'Val Precision': [epoch_metrics['precision']],
                'Val Recall': [epoch_metrics['recall']],
                'Val F1': [epoch_metrics['f1']],
                'Val AUROC': [epoch_metrics['auroc']],
                'Val AUPRC': [epoch_metrics['auprc']]
            })
        
        epoch_metrics_row.to_csv(metrics_csv_path, mode='a', header=False, index=False)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_model_path = os.path.join(model_save_dir, f'{mod_type}_best_model.pt')
            torch.save(best_model_state, best_model_path)
            logger.info(f"  New best model saved! (Val Loss: {val_loss:.4f})")
        
        logger.info("-" * 60)
    
    # Save final best model
    torch.save(best_model_state, os.path.join(model_save_dir, f'{mod_type}_best_model.pt'))
    
    # Save training history
    with open(os.path.join(model_save_dir, f'{mod_type}_training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {os.path.join(model_save_dir, f'{mod_type}_best_model.pt')}")
    logger.info(f"Total training time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot validation metrics
    plt.subplot(1, 3, 2)
    plt.plot([x['accuracy'] for x in history['val_metrics']], label='Accuracy')
    plt.plot([x['precision'] for x in history['val_metrics']], label='Precision')
    plt.plot([x['recall'] for x in history['val_metrics']], label='Recall')
    plt.plot([x['f1'] for x in history['val_metrics']], label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    # Plot AUROC and AUPRC
    plt.subplot(1, 3, 3)
    plt.plot([x['auroc'] for x in history['val_metrics']], label='AUROC')
    plt.plot([x['auprc'] for x in history['val_metrics']], label='AUPRC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('ROC and PR Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f'{mod_type}_training_curves.pdf'))
    plt.close()
    
    # Generate AUROC and AUPRC curves using the best model on validation set
    logger.info("Generating AUROC and AUPRC curves...")
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'{mod_type}_best_model.pt')))
    model.eval()
    
    # Get predictions on validation set
    val_preds_final = []
    val_labels_final = []
    
    with torch.no_grad():
        for X, _ in val_loader:
            X['instances'] = X['instances'].to(device)
            X['structure'] = X['structure'].to(device)
            X['mask'] = X['mask'].to(device)
            site_ids = X['site_ids']
            
            # Build labels
            if use_bayesian:
                y = torch.tensor([val_bayes_map.get(sid, 0.0) for sid in site_ids], 
                               dtype=torch.float32, device=device)
                # For ROC/PR curves, use binary labels based on median
                bayes_median = np.median(list(val_bayes_map.values()))
                y_binary = (y > bayes_median).float()
            else:
                y_binary = torch.tensor([1.0 if sid in cand_ids else 0.0 for sid in site_ids], 
                                       dtype=torch.float32, device=device)
            
            outputs = model(X)
            val_preds_final.extend(outputs.detach().cpu().numpy().tolist())
            val_labels_final.extend(y_binary.detach().cpu().numpy().tolist())
    
    val_preds_final = np.array(val_preds_final)
    val_labels_final = np.array(val_labels_final).astype(int)
    
    # Only generate curves if we have both classes
    if len(np.unique(val_labels_final)) > 1:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(val_labels_final, val_preds_final)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(val_labels_final, val_preds_final)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Create figure with both curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot ROC curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Plot Precision-Recall curve
        ax2.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14)
        ax2.legend(loc="lower left", fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        roc_pr_path = os.path.join(model_save_dir, f'{mod_type}_AUROC_AUPRC.pdf')
        plt.savefig(roc_pr_path)
        plt.close()
        
        logger.info(f"AUROC/AUPRC curves saved to {roc_pr_path}")
        logger.info(f"Final AUROC: {roc_auc:.4f}, Final AUPRC: {pr_auc:.4f}")
    else:
        logger.warning("Cannot generate ROC/PR curves: validation set contains only one class")
    
    return os.path.join(model_save_dir, f'{mod_type}_best_model.pt')


def main():
    """Main function for direct script execution"""
    parser = argparse.ArgumentParser(description='NanoMod-tRNA training script (Attention MIL with Adaptive Strategy)')
    
    # Data parameters
    parser.add_argument('--train-file', type=str, required=True, help='Training feature file path')
    parser.add_argument('--val-file', type=str, required=True, help='Validation feature file path')
    parser.add_argument('--mod-site-file', type=str, required=True, help='Modification site file path (candidates)')
    parser.add_argument('--output-dir', type=str, default='NanoMod_output', help='Output directory')
    
    # Model parameters
    parser.add_argument('--mod-type', type=str, default='D', help='Modification type')
    parser.add_argument('--kmer-nums', type=int, default=781, help='Number of possible kmers (unused, kept for compatibility)')
    parser.add_argument('--num-reads', type=int, default=30, help='Number of reads per site')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=128, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Adaptive training strategy parameters
    parser.add_argument('--mismatch-threshold', type=float, default=1.5, help='Mismatch rate ratio threshold for mode selection')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Train model
    best_model_path = train_model(
        args.train_file, args.val_file, args.mod_site_file, 
        args.output_dir, 
        batch_size=args.batch_size, 
        num_epochs=args.epochs, 
        learning_rate=args.learning_rate,
        num_instances=args.num_reads, 
        kmer_nums=args.kmer_nums,
        dropout_rate=args.dropout_rate,
        num_workers=args.num_workers,
        mod_type=args.mod_type,
        mismatch_threshold=args.mismatch_threshold
    )
    
    logger.info(f"Training completed! Model saved at {best_model_path}")


if __name__ == '__main__':
    main()