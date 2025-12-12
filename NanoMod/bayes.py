#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian module for NanoMod-tRNA (site-level)

- Estimate Bayesian parameters from training data (site-level), using Beta(1,1) priors
- Compute posterior modification probabilities per site based on mismatch counts
- Reuse the same parameters for validation and prediction

Assumptions:
- Features file contains at least columns: seq.name, seq.pos, mistake
- mistake: 1 for mismatch, 0 for match
- Candidate modification sites file (.tsv) contains: trna_combined, position (0-based), modified_base
- mod_type provided externally selects one modification type per run
"""

import os
import json
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import datatable as dt

from .utils import classify_trna_structure

# --------------------------
# Helpers
# --------------------------

def _read_features(features_file: str) -> pd.DataFrame:
    dt_data = dt.fread(features_file, sep='\t')
    df = dt_data.to_pandas()
    # Normalize tRNA naming (T -> U) for consistency
    if 'seq.name' in df.columns:
        df['seq.name'] = df['seq.name'].astype(str).str.replace('T', 'U')
    # Ensure types
    if 'mistake' in df.columns:
        df['mistake'] = pd.to_numeric(df['mistake'], errors='coerce').fillna(0).astype(int)
    if 'seq.pos' in df.columns:
        df['seq.pos'] = pd.to_numeric(df['seq.pos'], errors='coerce').fillna(0).astype(int)
    # Build site_id
    df['site_id'] = df['seq.name'].astype(str) + '_' + df['seq.pos'].astype(str)
    return df


def _read_candidate_sites(mod_sites_file: str, mod_type: str) -> pd.DataFrame:
    sites = pd.read_csv(mod_sites_file, sep='\t')
    # Normalize naming
    sites['trna_combined'] = sites['trna_combined'].astype(str).str.replace('T', 'U')
    # Keep only this mod_type
    sites = sites[sites['modified_base'].astype(str) == str(mod_type)].copy()
    sites['position'] = pd.to_numeric(sites['position'], errors='coerce').fillna(0).astype(int)
    # Build site_id
    sites['site_id'] = sites['trna_combined'].astype(str) + '_' + sites['position'].astype(str)
    return sites[['site_id', 'trna_combined', 'position', 'modified_base']]


def _aggregate_site_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    # Group by site_id to compute mismatch counts and totals
    agg = df.groupby(['site_id', 'seq.name', 'seq.pos'], as_index=False).agg(
        total_reads=('mistake', 'count'),
        mismatch_count=('mistake', 'sum')
    )
    agg['mismatch_rate'] = np.where(agg['total_reads'] > 0, agg['mismatch_count'] / agg['total_reads'], 0.0)
    return agg


# --------------------------
# Beta-Binomial with Beta(1,1)
# --------------------------

def _log_beta(a: float, b: float) -> float:
    from math import lgamma
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def _beta_binom_marginal_log_prob(k: int, n: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    # log[C(n,k)] + log B(k+alpha, n-k+beta) - log B(alpha, beta)
    # C(n,k) in log-space
    from math import lgamma
    if n < 0 or k < 0 or k > n:
        return -np.inf
    log_comb = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_beta_num = _log_beta(k + alpha, n - k + beta)
    log_beta_den = _log_beta(alpha, beta)
    return log_comb + (log_beta_num - log_beta_den)


# --------------------------
# Public APIs
# --------------------------

def estimate_bayes_params(train_features_file: str, mod_sites_file: str, mod_type: str) -> Dict:
    """
    Empirical Bayes estimation of Beta-Binomial parameters.

    From site-level counts (k = mismatch count, n = total reads) in the
    training features, aggregate two groups (candidate / non-candidate) and
    obtain two sets of Beta hyperparameters:

        alpha1 = 1 + Σ k_i       (candidate sites)
        beta1  = 1 + Σ (n_i-k_i) (candidate sites)
        alpha0 = 1 + Σ k_i       (non-candidate sites)
        beta0  = 1 + Σ (n_i-k_i) (non-candidate sites)

    If a group has no data, fall back to Beta(1,1).

    π is defined as the fraction of candidate sites among all sites present
    in the training data:

        pi = (#candidate sites present) / (#all sites present)

    Returns:
        dict: {pi, alpha1, beta1, alpha0, beta0, mod_type}
    """
    df = _read_features(train_features_file)
    agg = _aggregate_site_mismatch(df)

    cand = _read_candidate_sites(mod_sites_file, mod_type)
    cand_ids = set(cand['site_id'].unique())

    # Determine sites present in training features
    sites_present = set(agg['site_id'].unique())
    cand_present = cand_ids.intersection(sites_present)

    # Partition aggregated counts
    agg_cand = agg[agg['site_id'].isin(cand_present)].copy()
    agg_non = agg[~agg['site_id'].isin(cand_present)].copy()

    k1 = int(agg_cand['mismatch_count'].sum()) if len(agg_cand) > 0 else 0
    n1 = int(agg_cand['total_reads'].sum()) if len(agg_cand) > 0 else 0
    k0 = int(agg_non['mismatch_count'].sum()) if len(agg_non) > 0 else 0
    n0 = int(agg_non['total_reads'].sum()) if len(agg_non) > 0 else 0

    # Ensure non-negative complements
    comp1 = max(0, n1 - k1)
    comp0 = max(0, n0 - k0)

    # Empirical-Bayes posteriors of class-conditional Beta parameters with Beta(1,1) base
    alpha1 = 1.0 + float(k1)
    beta1 = 1.0 + float(comp1)
    alpha0 = 1.0 + float(k0)
    beta0 = 1.0 + float(comp0)

    # pi = fraction of candidate sites among all sites present in training
    total_sites = len(sites_present)
    pi = (len(cand_present) / total_sites) if total_sites > 0 else 0.0

    params = {
        'pi': float(pi),
        'alpha1': float(alpha1),
        'beta1': float(beta1),
        'alpha0': float(alpha0),
        'beta0': float(beta0),
        'mod_type': str(mod_type)
    }
    return params


def compute_bayes_posteriors(features_file: str, mod_sites_file: str, mod_type: str, params: Dict) -> pd.DataFrame:
    """
    Compute posterior P(mod|data) per site using training-estimated params.
    For non-candidate sites, posterior is set to 0.
    Returns DataFrame with columns: site_id, seq.name, seq.pos, bayes_prob
    """
    df = _read_features(features_file)
    agg = _aggregate_site_mismatch(df)

    cand = _read_candidate_sites(mod_sites_file, mod_type)
    cand_ids = set(cand['site_id'].unique())

    pi = float(params.get('pi', 0.0))
    a1 = float(params.get('alpha1', 1.0))
    b1 = float(params.get('beta1', 1.0))
    a0 = float(params.get('alpha0', 1.0))
    b0 = float(params.get('beta0', 1.0))

    # Compute posterior for candidate sites only; others set to 0
    probs = []
    for _, row in agg.iterrows():
        site_id = row['site_id']
        n = int(row['total_reads'])
        k = int(row['mismatch_count'])
        if site_id not in cand_ids:
            probs.append(0.0)
            continue
        # log-space computations for stability
        log_p_mod = _beta_binom_marginal_log_prob(k, n, a1, b1)
        log_p_unm = _beta_binom_marginal_log_prob(k, n, a0, b0)
        # posterior = pi * p_mod / (pi*p_mod + (1-pi)*p_unm)
        # do in log-space
        # numerator = log(pi) + log_p_mod
        # denominator = logsumexp(log(pi)+log_p_mod, log(1-pi)+log_p_unm)
        if pi <= 0.0:
            probs.append(0.0)
            continue
        if pi >= 1.0:
            probs.append(1.0)
            continue
        log_num = math.log(pi + 1e-15) + log_p_mod
        log_alt = math.log(1.0 - pi + 1e-15) + log_p_unm
        # logsumexp
        m = max(log_num, log_alt)
        log_den = m + math.log(math.exp(log_num - m) + math.exp(log_alt - m))
        post = math.exp(log_num - log_den)
        probs.append(float(post))

    out = agg[['site_id', 'seq.name', 'seq.pos']].copy()
    out['bayes_prob'] = probs
    return out


def save_params(params: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=2)


def load_params(load_path: str) -> Dict:
    with open(load_path, 'r') as f:
        return json.load(f)
