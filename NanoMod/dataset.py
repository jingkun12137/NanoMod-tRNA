#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NanoMod-tRNA Model Module v0.9.6 (MIL with Noisy-OR pooling)

This module implements a Multiple Instance Learning (MIL) model for tRNA
modification detection.

Model Architecture:
- Input: [B, N, 6] where B=batch_size, N=num_reads (default 30), 6=electrical signal features
- Instance Encoder: Encodes each read independently (6 → 128 dim)
- Per-read classifier: Outputs per-read probabilities p_ij
- Site-level aggregation: Noisy-OR pooling s_i = 1 - prod_j (1 - p_ij)
- Output: [B,] site-level modification probability (0-1)

Features:
- 6 electrical signal features per read
- tRNA structure embedding (16-dim) used as additional input to the per-read classifier
"""

import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NanoMod')


class InstanceEncoder(nn.Module):
    """Encodes each read (instance) independently."""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, dropout_rate: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension (6: 6 electrical signals)
            hidden_dim: Output hidden dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, instances):
        """
        Args:
            instances: [B, N, 6] or [N, 6]
        Returns:
            [B, N, hidden_dim] or [N, hidden_dim]
        """
        return self.encoder(instances)

class ReadClassifier(nn.Module):
    """Per-read classifier to obtain p_ij from instance features."""

    def __init__(self, hidden_dim: int = 128, dropout_rate: float = 0.3):
        """
        Args:
            hidden_dim: Input feature dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, instance_features):
        """
        Args:
            instance_features: [B, N, hidden_dim]
        Returns:
            read_probs: [B, N]
        """
        return self.head(instance_features).squeeze(-1)


class NanoMod(nn.Module):
    """EM (Noisy-OR) model for modification detection."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, 
                 structure_emb_dim: int = 16, dropout_rate: float = 0.3, 
                 kmer_nums=None, num_instances: int = 30):
        """
        Args:
            input_dim: Input feature dimension per read (default: 6)
            hidden_dim: Hidden dimension for instance encoder (default: 128)
            structure_emb_dim: Unused, kept for compatibility
            dropout_rate: Dropout rate (default: 0.3)
            kmer_nums: Unused, kept for compatibility
            num_instances: Number of reads per site (default: 30)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_instances = num_instances
        self.kmer_nums = kmer_nums  # Keep for compatibility
        
        # Components: per-read encoder + per-read classifier; Noisy-OR aggregator in forward
        self.instance_encoder = InstanceEncoder(input_dim, hidden_dim, dropout_rate)
        # Structure embedding (0-8)
        self.structure_embedding = nn.Embedding(9, structure_emb_dim)
        # Read classifier now consumes [hidden_dim + structure_emb_dim]
        self.read_classifier = ReadClassifier(hidden_dim + structure_emb_dim, dropout_rate)
        
        logger.info(f"Initialized EM (Noisy-OR) model (no structure info): input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, num_instances={num_instances}")
    
    def forward(self, x, return_attention=False, return_read_probs=False):
        """
        Args:
            x: dict with keys:
                - 'instances': [B, N, 6] read features
                - 'structure': [B] structure index (unused, kept for compatibility)
                - 'mask': [B, N] optional, boolean mask for valid instances
            return_attention: Unused (kept for compatibility)
        Returns:
            prob: [B] probability
        """
        instances = x['instances']  # [B, N, 6]
        structure = x['structure']  # [B]
        mask = x.get('mask', None)  # [B, N] optional

        # 1. Encode each read
        instance_features = self.instance_encoder(instances)  # [B, N, hidden_dim]

        # 2. Concatenate structure embedding to each read feature
        structure_emb = self.structure_embedding(structure)  # [B, structure_emb_dim]
        structure_emb = structure_emb.unsqueeze(1).expand(-1, instance_features.size(1), -1)  # [B, N, structure_emb_dim]
        combined_features = torch.cat([instance_features, structure_emb], dim=-1)  # [B, N, hidden_dim + structure_emb_dim]

        # 3. Per-read probabilities
        read_probs = self.read_classifier(combined_features)  # [B, N]
        if mask is not None:
            # Invalidate padded reads by setting p=0 so they do not affect Noisy-OR
            read_probs = read_probs.masked_fill(~mask, 0.0)

        # 4. Noisy-OR aggregation: s_i = 1 - prod_j (1 - p_ij)
        one_minus_p = 1.0 - read_probs.clamp(0.0, 1.0)
        # Stable product via log-sum
        log_prod = torch.log(one_minus_p + 1e-12).sum(dim=1)
        prod = torch.exp(log_prod)
        prob = (1.0 - prod).clamp(0.0, 1.0)

        if return_read_probs:
            return prob, read_probs

        return prob
