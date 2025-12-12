#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NanoMod-tRNA Utilities Module

This module provides utility functions for tRNA D modification detection,
including feature importance analysis, optimal threshold finding, and modification pattern analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import torch

def find_optimal_threshold(true_labels, pred_probs):
    """
    Find optimal prediction threshold based on F1 score
    
    Parameters:
        true_labels (np.ndarray): True labels
        pred_probs (np.ndarray): Predicted probabilities
        
    Returns:
        float: Optimal threshold
        float: F1 score at optimal threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        pred_labels = (pred_probs > threshold).astype(int)
        f1 = f1_score(true_labels, pred_labels)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

def analyze_feature_importance(model, dataset, output_dir):
    """
    Analyze feature importance using permutation importance
    
    Parameters:
        model (torch.nn.Module): Trained model
        dataset (tRNADataset): Dataset
        output_dir (str): Output directory
        
    Returns:
        pd.DataFrame: Feature importance results
    """
    print("Analyzing feature importance...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names
    feature_names = [
        'dtw.current', 'dtw.current_sd', 'dtw.length', 
        'dtw.amplitude', 'dtw.skewness', 'dtw.kurtosis'
    ]
    
    # Get data
    X = []
    y = []
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features']  # [batch_size, min_reads, feature_dim]
            labels = batch['label']
            
            # Average features across reads
            avg_features = features.mean(dim=1).numpy()
            
            X.append(avg_features)
            y.append(labels.numpy())
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Train a random forest on the features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance for D Modification Detection', fontsize=22)
    plt.xlabel('Permutation Importance', fontsize=20)
    plt.ylabel('Feature', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save importance results
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.tsv'), sep='\t', index=False)
    
    return importance_df

def analyze_modification_patterns(results_df, mod_sites_df, output_dir):
    """
    Analyze D modification patterns
    
    Parameters:
        results_df (pd.DataFrame): Prediction results
        mod_sites_df (pd.DataFrame): Modification sites information
        output_dir (str): Output directory
        
    Returns:
        dict: Analysis results
    """
    print("Analyzing D modification patterns...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge results with modification sites information
    merged_df = results_df.copy()
    merged_df['seq_id'] = merged_df['site_id'].apply(lambda x: x.split('_')[0])
    merged_df['position'] = merged_df['site_id'].apply(lambda x: int(x.split('_')[1]))
    
    # Add tRNA type and feature information if available
    if 'trna_type' in mod_sites_df.columns:
        site_info = mod_sites_df.set_index('site_id')[['trna_type', 'trna_feature']].to_dict(orient='index')
        
        def get_trna_type(site_id):
            return site_info.get(site_id, {}).get('trna_type', 'Unknown')
        
        def get_trna_feature(site_id):
            return site_info.get(site_id, {}).get('trna_feature', 'Unknown')
        
        merged_df['trna_type'] = merged_df['site_id'].apply(get_trna_type)
        merged_df['trna_feature'] = merged_df['site_id'].apply(get_trna_feature)
    
    # Analyze position distribution
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12, 8))
    sns.histplot(data=merged_df, x='position', hue='true_label', bins=50, alpha=0.7, 
                element='step', palette=['blue', 'red'])
    plt.title('D Modification Position Distribution', fontsize=22)
    plt.xlabel('Position in tRNA', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.legend(['Non-modified', 'D-modified'], fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze tRNA type distribution if available
    if 'trna_type' in merged_df.columns:
        plt.figure(figsize=(12, 8))
        type_counts = merged_df[merged_df['true_label'] == 1]['trna_type'].value_counts()
        sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis')
        plt.title('D Modification Distribution by tRNA Type', fontsize=22)
        plt.xlabel('tRNA Type', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trna_type_distribution.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze tRNA feature distribution if available
    if 'trna_feature' in merged_df.columns:
        plt.figure(figsize=(12, 8))
        feature_counts = merged_df[merged_df['true_label'] == 1]['trna_feature'].value_counts()
        sns.barplot(x=feature_counts.index, y=feature_counts.values, palette='viridis')
        plt.title('D Modification Distribution by tRNA Feature', fontsize=22)
        plt.xlabel('Feature', fontsize=20)
        plt.ylabel('Value', fontsize=20)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trna_feature_distribution.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze prediction performance by position
    position_metrics = merged_df.groupby('position').apply(
        lambda x: pd.Series({
            'accuracy': np.mean(x['pred_label'] == x['true_label']),
            'precision': np.sum((x['pred_label'] == 1) & (x['true_label'] == 1)) / np.sum(x['pred_label'] == 1) if np.sum(x['pred_label'] == 1) > 0 else 0,
            'recall': np.sum((x['pred_label'] == 1) & (x['true_label'] == 1)) / np.sum(x['true_label'] == 1) if np.sum(x['true_label'] == 1) > 0 else 0,
            'count': len(x)
        })
    ).reset_index()
    
    # Plot performance by position
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    sns.scatterplot(data=position_metrics, x='position', y='accuracy', size='count', sizes=(20, 200), alpha=0.7)
    plt.title('Prediction Accuracy by Position', fontsize=22)
    plt.xlabel('Position in tRNA', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=position_metrics, x='position', y='precision', label='Precision', alpha=0.7)
    sns.scatterplot(data=position_metrics, x='position', y='recall', label='Recall', alpha=0.7)
    plt.title('Precision and Recall by Position', fontsize=22)
    plt.xlabel('Position in tRNA', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(['Precision', 'Recall'], fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_position.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis results
    position_metrics.to_csv(os.path.join(output_dir, 'position_metrics.tsv'), sep='\t', index=False)
    
    # Return analysis results
    return {
        'position_metrics': position_metrics
    }

def classify_trna_structure(position):
    """
    Classify tRNA position into structural elements based on standard numbering.
    
    Args:
        position: Integer position in tRNA (0-based)
        
    Returns:
        Integer classification:
        1: Acceptor stem (positions 1-7, 66-72)
        2: D-arm (positions 10-13, 22-25)
        3: D-loop (positions 14-21)
        4: Anticodon stem (positions 27-31, 39-43)
        5: Anticodon loop (positions 32-38)
        6: Variable loop (positions 44-48)
        7: T-arm (positions 49-53, 61-65)
        8: T-loop (positions 54-60)
        0: Other
    """
    position = int(position)
    
    # Acceptor stem (positions 1-7, 66-72)
    if 1 <= position <= 7 or 66 <= position <= 72:
        return 1
    # D-arm (positions 10-13, 22-25)
    elif (10 <= position <= 13) or (22 <= position <= 25):
        return 2
    # D-loop (positions 14-21)
    elif 14 <= position <= 21:
        return 3
    # Anticodon stem (positions 27-31, 39-43)
    elif (27 <= position <= 31) or (39 <= position <= 43):
        return 4
    # Anticodon loop (positions 32-38)
    elif 32 <= position <= 38:
        return 5
    # Variable loop (positions 44-48)
    elif 44 <= position <= 48:
        return 6
    # T-arm (positions 49-53, 61-65)
    elif (49 <= position <= 53) or (61 <= position <= 65):
        return 7
    # T-loop (positions 54-60)
    elif 54 <= position <= 60:
        return 8
    # Other
    else:
        return 0

def visualize_attention_weights(model, dataset, output_dir, num_samples=10):
    """
    Visualize attention weights for selected samples
    
    Parameters:
        model (torch.nn.Module): Trained model
        dataset (tRNADataset): Dataset
        output_dir (str): Output directory
        num_samples (int): Number of samples to visualize
    """
    print("Visualizing attention weights...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names
    feature_names = [
        'dtw.current', 'dtw.current_sd', 'dtw.length', 
        'dtw.amplitude', 'dtw.skewness', 'dtw.kurtosis'
    ]
    
    # Get a batch of data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(dataloader))
    
    features = batch['features'].to(model.device)
    kmer = batch['kmer'].to(model.device)
    labels = batch['label'].to(model.device)
    site_ids = batch['site_id']
    
    # Forward pass to get attention weights
    model.eval()
    with torch.no_grad():
        outputs = model({'features': features, 'kmer': kmer})
        
        # Extract attention weights from the attention layer
        # This is model-specific and may need to be adapted
        attention_layer = model.feature_extractor[-1]
        batch_size, min_reads = kmer.shape
        
        # Reshape features for processing
        reshaped_features = features.reshape(batch_size * min_reads, -1)
        
        # Get attention weights
        attention_weights = torch.softmax(attention_layer.fc(reshaped_features), dim=1)
        attention_weights = attention_weights.reshape(batch_size, min_reads, -1)
        
        # Average across reads
        avg_attention = attention_weights.mean(dim=1).cpu().numpy()
    
    # Plot attention weights for each sample
    plt.rcParams.update({'font.size': 18})
    for i in range(min(num_samples, len(site_ids))):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=np.arange(avg_attention.shape[1]), y=avg_attention[i])
        plt.title(f'Attention Weights for Sample {site_ids[i]} (Label: {"D-modified" if labels[i] == 1 else "Non-modified"})', fontsize=22)
        plt.xlabel('Feature Index', fontsize=20)
        plt.ylabel('Attention Weight', fontsize=20)
        plt.xticks(np.arange(avg_attention.shape[1]), [f'Feature {j+1}' for j in range(avg_attention.shape[1])], fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_weights_sample_{i}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot average attention weights across all samples
    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.arange(avg_attention.shape[1]), y=avg_attention.mean(axis=0))
    plt.title('Average Attention Weights Across All Samples', fontsize=22)
    plt.xlabel('Feature Index', fontsize=20)
    plt.ylabel('Average Attention Weight', fontsize=20)
    plt.xticks(np.arange(avg_attention.shape[1]), [f'Feature {j+1}' for j in range(avg_attention.shape[1])], fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_attention_weights.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
