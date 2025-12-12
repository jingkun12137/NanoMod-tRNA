#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tRNA Modification Site Analysis Tool

This module compares modified and unmodified tRNA FASTA sequences to identify
modification sites and generate summary reports. It also implements the
Needleman-Wunsch algorithm for sequence alignment to infer modification sites
for unannotated tRNA sequences.
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ModificationSiteAnalyzer:
    """tRNA modification site analyzer.

    Analyzes modified and unmodified tRNA sequences to identify and report
    modification sites.
    """
    
    def __init__(self, modified_fasta, unmodified_fasta, yeast_tRNA_fasta=None, output_file=None, mod_type='D', pdf_output=None):
        """Initialize the modification site analyzer.

        Parameters:
            modified_fasta (str): FASTA file path for modified tRNAs.
            unmodified_fasta (str): FASTA file path for unmodified tRNAs.
            yeast_tRNA_fasta (str, optional): FASTA file path for yeast reference tRNAs.
            output_file (str, optional): Output TSV path for modification sites.
            mod_type (str, optional): Modification type to search for (default: 'D' for dihydrouridine).
            pdf_output (str, optional): Output PDF path for visualization.
        """
        self.modified_fasta = modified_fasta
        self.unmodified_fasta = unmodified_fasta
        self.yeast_tRNA_fasta = yeast_tRNA_fasta
        self.output_file = output_file
        self.mod_type = mod_type
        self.pdf_output = pdf_output
        
        # Initialize data structures
        self.modified_records = {}
        self.unmodified_records = {}
        self.yeast_records = {}
        self.modification_sites = {}  # mapping: sequence ID -> list of modification positions
        self.final_modification_sites = []  # final list of modification sites for output
        
        # Define standard base set (excluding modified bases)
        self.standard_bases = set('AUCG')
    
    def _parse_fasta(self, fasta_file):
        """Parse a FASTA file and extract sequences and metadata.

        Parameters:
            fasta_file (str): Path to FASTA file.
            
        Returns:
            dict: Mapping from sequence ID to
                  (sequence, tRNA type, feature, combined name, raw header).
        """
        logger.info(f"Loading tRNA sequences from {fasta_file}...")
        
        records = {}
        current_id = None
        current_seq = ""
        current_trna_type = None
        current_feature = None
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('>'):
                    # Handle previous sequence
                    if current_id and current_seq:
                        # Convert DNA sequence to RNA (T → U)
                        current_seq = current_seq.replace('T', 'U').replace('t', 'u')
                        
                        # Combine tRNA type and feature into a standard name, e.g. "Ala-AGC"
                        trna_combined = None
                        if current_trna_type and current_feature:
                            # Replace T with U in feature
                            current_feature = current_feature.replace('T', 'U').replace('t', 'u')
                            trna_combined = f"{current_trna_type}-{current_feature}"
                        
                        records[current_id] = (current_seq, current_trna_type, current_feature, trna_combined, line)
                        current_seq = ""
                    
                    # Parse new header line
                    current_id = line[1:].split('|')[0] if '|' in line else line[1:]
                    current_trna_type = None
                    current_feature = None
                    
                    # Complex FASTA header format
                    if '|' in line:
                        parts = line.split('|')
                        for part in parts:
                            if 'Subtype:' in part:
                                # Safely access split result to avoid index errors
                                part_split = part.split(':')
                                if len(part_split) > 1:
                                    current_trna_type = part_split[1]
                            elif 'Feature:' in part:
                                # Safely access split result to avoid index errors
                                part_split = part.split(':')
                                if len(part_split) > 1:
                                    current_feature = part_split[1]
                    else:
                        # Simple FASTA header format, e.g. ">Ala-AGC"
                        if '-' in line[1:]:
                            header_parts = line[1:].split('-')
                            current_trna_type = header_parts[0]
                            # Safely access split result to avoid index errors
                            current_feature = header_parts[1] if len(header_parts) > 1 else None
                            # Replace T with U in feature
                            if current_feature:
                                current_feature = current_feature.replace('T', 'U').replace('t', 'u')
                else:
                    current_seq += line
            
            # Handle the last sequence
            if current_id and current_seq:
                # Convert DNA sequence to RNA (T → U)
                current_seq = current_seq.replace('T', 'U').replace('t', 'u')
                
                # Combine tRNA type and feature into a standard name
                trna_combined = None
                if current_trna_type and current_feature:
                    # Replace T with U in feature
                    current_feature = current_feature.replace('T', 'U').replace('t', 'u')
                    trna_combined = f"{current_trna_type}-{current_feature}"
                
                records[current_id] = (current_seq, current_trna_type, current_feature, trna_combined, line)
        
        logger.info(f"Loaded {len(records)} tRNA sequences")
        return records
    
    def needleman_wunsch(self, seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
        """
        Implement the Needleman-Wunsch sequence alignment algorithm.
        
        Parameters:
            seq1 (str): First sequence.
            seq2 (str): Second sequence.
            match_score (int): Match score (default: 1).
            mismatch_penalty (int): Mismatch penalty (default: -1).
            gap_penalty (int): Gap penalty (default: -1).
            
        Returns:
            tuple: (aligned_seq1, aligned_seq2, score, identity)
                aligned_seq1: Aligned first sequence.
                aligned_seq2: Aligned second sequence.
                score: Alignment score.
                identity: Sequence identity.
        """
        try:
            # Ensure all Ts are replaced by U in sequences
            seq1 = seq1.replace('T', 'U').replace('t', 'u')
            seq2 = seq2.replace('T', 'U').replace('t', 'u')
            
            # Sequence lengths
            len1 = len(seq1)
            len2 = len(seq2)
            
            # Initialize score matrix
            score_matrix = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
            
            # Initialize traceback matrix
            traceback = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
            
            # Fill first row and first column
            for i in range(len1 + 1):
                score_matrix[i][0] = gap_penalty * i
            for j in range(len2 + 1):
                score_matrix[0][j] = gap_penalty * j
                
            # Initialize traceback boundaries
            for i in range(1, len1 + 1):
                traceback[i][0] = 1  # 上移
            for j in range(1, len2 + 1):
                traceback[0][j] = 2  # 左移
                
            # Fill score and traceback matrices
            for i in range(1, len1 + 1):
                for j in range(1, len2 + 1):
                    # Compute three possible scores
                    match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
                    delete = score_matrix[i-1][j] + gap_penalty
                    insert = score_matrix[i][j-1] + gap_penalty
                    
                    # Choose the maximum score
                    score_matrix[i][j] = max(match, delete, insert)
                    
                    # Update traceback matrix
                    if score_matrix[i][j] == match:
                        traceback[i][j] = 0  # diagonal move
                    elif score_matrix[i][j] == delete:
                        traceback[i][j] = 1  # move up
                    else:
                        traceback[i][j] = 2  # move left
            
            # Trace back to construct aligned sequences
            aligned_seq1 = ""
            aligned_seq2 = ""
            
            i, j = len1, len2
            
            while i > 0 or j > 0:
                if i > 0 and j > 0 and traceback[i][j] == 0:
                    # Diagonal move (match or mismatch)
                    aligned_seq1 = seq1[i-1] + aligned_seq1
                    aligned_seq2 = seq2[j-1] + aligned_seq2
                    i -= 1
                    j -= 1
                elif i > 0 and traceback[i][j] == 1:
                    # Move up (insert gap in seq2)
                    aligned_seq1 = seq1[i-1] + aligned_seq1
                    aligned_seq2 = '-' + aligned_seq2
                    i -= 1
                else:
                    # Move left (insert gap in seq1)
                    aligned_seq1 = '-' + aligned_seq1
                    aligned_seq2 = seq2[j-1] + aligned_seq2
                    j -= 1
            
            # Compute sequence identity
            matches = 0      # number of matches
            mismatches = 0   # number of mismatches
            gaps_in_seq1 = 0 # number of gaps in sequence 1
            gaps_in_seq2 = 0 # number of gaps in sequence 2
            
            for i in range(len(aligned_seq1)):
                if aligned_seq1[i] == '-':
                    gaps_in_seq1 += 1
                elif aligned_seq2[i] == '-':
                    gaps_in_seq2 += 1
                elif aligned_seq1[i] == aligned_seq2[i]:
                    matches += 1
                else:
                    mismatches += 1
            
            # Identity = matches / (matches + mismatches + gaps)
            identity = matches / (matches + mismatches + gaps_in_seq1 + gaps_in_seq2) if (matches + mismatches + gaps_in_seq1 + gaps_in_seq2) > 0 else 0
            
            return aligned_seq1, aligned_seq2, score_matrix[len1][len2], identity
            
        except Exception as e:
            logger.error(f"Error in Needleman-Wunsch algorithm: {e}")
            # On error, return original sequences and low identity
            return seq1, seq2, 0, 0.0
            
    def parse_fasta_files(self):
        """Parse all FASTA files and extract sequence information."""
        # Parse modified tRNA sequences
        if self.modified_fasta:
            self.modified_records = self._parse_fasta(self.modified_fasta)
            
        # Parse unmodified tRNA sequences
        if self.unmodified_fasta:
            self.unmodified_records = self._parse_fasta(self.unmodified_fasta)
            
        # Parse yeast reference tRNA sequences
        if self.yeast_tRNA_fasta:
            logger.info(f"Parsing yeast reference {self.yeast_tRNA_fasta}...")
            self.yeast_records = self._parse_fasta(self.yeast_tRNA_fasta)
            logger.info(f"Loaded {len(self.yeast_records)} yeast reference tRNAs")
    
    def identify_modifications(self):
        """
        Identify modification sites by comparing modified and unmodified sequences.
        """
        logger.info("Identifying modification sites by comparing modified and unmodified sequences...")
        
        # Ensure FASTA files have been parsed
        if not self.modified_records or not self.unmodified_records:
            self.parse_fasta_files()
        
        # Iterate over all sequence pairs to identify modification sites
        for seq_id, (mod_seq, mod_type, mod_feature, mod_combined, _) in self.modified_records.items():
            # Check if there is a corresponding ID in unmodified sequences
            if seq_id not in self.unmodified_records:
                logger.warning(f"Sequence ID {seq_id} found in modified but not in unmodified file")
                continue
            
            # Get the corresponding unmodified sequence
            unmod_seq, _, _, _, _ = self.unmodified_records[seq_id]
            
            # Align sequences using Needleman-Wunsch
            aligned_mod_seq, aligned_unmod_seq, _, _ = self.needleman_wunsch(mod_seq, unmod_seq)
            
            # Initialize modification site list for this sequence
            self.modification_sites[seq_id] = []
            
            # Track position in the real (gapless) sequence
            current_real_pos = 0
            
            # Check base comparison at each position
            for i, (mod_base, unmod_base) in enumerate(zip(aligned_mod_seq, aligned_unmod_seq)):
                # Skip gaps in alignment
                if mod_base == '-' or unmod_base == '-':
                    continue
                    
                # Update real position index
                current_real_pos += 1
                
                # Detect modification
                if mod_base == self.mod_type:
                    # Record modification position (0-based)
                    self.modification_sites[seq_id].append(current_real_pos - 1)
        
        # Summarize number of identified modification sites
        total_mod_sites = sum(len(sites) for sites in self.modification_sites.values())
        logger.info(f"Found {total_mod_sites} modification sites across {len(self.modification_sites)} sequences")
    
    def map_to_yeast_reference(self):
        """
        Map unmodified sequences to yeast reference tRNA sequences.
        """
        logger.info("Mapping sequences to yeast reference tRNAs...")
        
        # Ensure modification sites have been identified
        if not self.modification_sites:
            self.identify_modifications()
        
        # Ensure yeast reference sequences are available
        if not self.yeast_records:
            if self.yeast_tRNA_fasta:
                self.yeast_records = self._parse_fasta(self.yeast_tRNA_fasta)
            else:
                logger.error("No yeast reference file provided")
                return
        
        # Create a set to store final modification sites (reference FASTA name, position)
        final_sites = set()
        
        # Iterate over all unmodified sequences
        for seq_id, (unmod_seq, _, _, _, _) in self.unmodified_records.items():
            # Check if this sequence has any modification sites
            if seq_id not in self.modification_sites or not self.modification_sites[seq_id]:
                continue
            
            # Get modification positions for this sequence
            mod_positions = self.modification_sites[seq_id]
            
            # Iterate over all yeast reference sequences
            for yeast_id, (yeast_seq, yeast_type, yeast_feature, yeast_combined, _) in self.yeast_records.items():
                # Ensure all Ts in yeast sequence are replaced by U
                yeast_seq = yeast_seq.replace('T', 'U').replace('t', 'u')
                
                # Compare sequences using Needleman-Wunsch
                seq1 = unmod_seq
                seq2 = yeast_seq
                _, _, _, identity = self.needleman_wunsch(seq1, seq2)
                
                # If identity > 70%, treat this yeast sequence as the reference for the unmodified sequence
                if identity >= 0.7:
                    logger.info(f"Matched sequence {seq_id} to yeast reference {yeast_id} ({yeast_combined}) with {identity:.2f} identity")
                    
                    # For each modification site of this sequence, add to final results
                    for pos in mod_positions:
                        # Use full reference FASTA name (yeast_id) and add (name, position) to set
                        final_sites.add((yeast_id, pos))
        
        # Convert set to list for storage
        for trna_combined, position in final_sites:
            self.final_modification_sites.append({
                'trna_combined': trna_combined,
                'position': position,  # keep original 0-based position index
                'modified_base': self.mod_type
            })
        
        logger.info(f"Mapped {len(self.final_modification_sites)} unique modification sites to yeast reference tRNAs")
        
    def compare_similar_trnas(self):
        """
        Compare similar tRNAs in yeast.tRNA.ext.fa and merge modification sites.

        Uses the Needleman-Wunsch algorithm to compare similarity between
        different tRNAs, and when identity > 70%, merges their modification sites.
        """
        logger.info("Comparing similar tRNAs to merge modification sites...")
        
        # Ensure mapping to yeast reference tRNAs has been performed
        if not self.final_modification_sites:
            logger.warning("No modification sites found, please run map_to_yeast_reference() method first")
            return
            
        # Group modification sites by tRNA
        trna_modifications = defaultdict(set)
        for site in self.final_modification_sites:
            trna_modifications[site['trna_combined']].add(site['position'])
        
        # Store newly identified modification sites
        new_sites = []
        
        # Extract all tRNA sequences and ensure T is replaced with U
        trna_sequences = {}
        for yeast_id, (yeast_seq, yeast_type, yeast_feature, yeast_combined, _) in self.yeast_records.items():
            # Ensure all Ts are replaced with U
            yeast_seq = yeast_seq.replace('T', 'U').replace('t', 'u')
            # Use full reference FASTA name (yeast_id) as key for consistency
            trna_sequences[yeast_id] = yeast_seq
        
        # Compare each pair of tRNAs
        processed_pairs = set()  # track processed tRNA pairs
        
        for trna1, positions1 in trna_modifications.items():
            if trna1 not in trna_sequences:
                continue
                
            for trna2, positions2 in trna_modifications.items():
                # Skip identical or already processed pairs
                if trna1 == trna2 or (trna1, trna2) in processed_pairs or (trna2, trna1) in processed_pairs:
                    continue
                
                if trna2 not in trna_sequences:
                    continue
                
                # 使用Needleman-Wunsch算法比较序列
                seq1 = trna_sequences[trna1]
                seq2 = trna_sequences[trna2]
                aligned_seq1, aligned_seq2, score, identity = self.needleman_wunsch(seq1, seq2)
                
                # If identity > 70%, merge modification sites
                if identity >= 0.7:
                    logger.info(f"Found similar tRNAs: {trna1} and {trna2} with {identity:.2f} identity")
                    logger.info(f"Alignment score: {score}")
                    
                    # For sites present in trna1 but not trna2, add to trna2
                    for pos in positions1:
                        if pos not in positions2:
                            new_sites.append({
                                'trna_combined': trna2,
                                'position': pos,
                                'modified_base': self.mod_type
                            })
                            logger.info(f"Adding position {pos} from {trna1} to {trna2}")
                    
                    # For sites present in trna2 but not trna1, add to trna1
                    for pos in positions2:
                        if pos not in positions1:
                            new_sites.append({
                                'trna_combined': trna1,
                                'position': pos,
                                'modified_base': self.mod_type
                            })
                            logger.info(f"Adding position {pos} from {trna2} to {trna1}")
                
                # Mark this pair as processed
                processed_pairs.add((trna1, trna2))
        
        # Update final modification site list
        self.final_modification_sites.extend(new_sites)
        logger.info(f"Added {len(new_sites)} additional modification sites from similar tRNAs")
        logger.info(f"Total unique modification sites after merging: {len(self.final_modification_sites)}")

    def analyze_modifications(self):
        """Analyze modification sites and generate summary statistics."""
        if not self.final_modification_sites:
            logger.warning("No modification sites found, please run map_to_yeast_reference() method first")
            return
            
        # Compute statistics for modification sites
        mod_positions = [site['position'] for site in self.final_modification_sites]  
        mod_trna_types = set(site['trna_combined'] for site in self.final_modification_sites)
        
        # Calculate statistics
        distinct_positions = set(mod_positions)
        position_range = (min(mod_positions), max(mod_positions))
        avg_position = sum(mod_positions) / len(mod_positions) if mod_positions else 0
        avg_per_trna = len(mod_positions) / len(mod_trna_types) if mod_trna_types else 0
        
        # Log statistics
        logger.info(f"Found modifications at {len(distinct_positions)} distinct positions")
        logger.info(f"Found modifications in {len(mod_trna_types)} distinct tRNA types")
        logger.info(f"Average number of modifications per tRNA: {avg_per_trna:.2f}")
        logger.info(f"Modification position range: {position_range[0]}-{position_range[1]} (avg: {avg_position:.2f})")
        
    def plot_modification_distribution(self, output_file=None):
        """
        Plot the distribution of modification sites.
        
        Parameters:
            output_file (str, optional): Output image file path.
        """
        if not self.final_modification_sites:
            logger.warning("No modification sites found, please run map_to_yeast_reference() method first")
            return
            
        logger.info(f"Plotting modification distribution...")
        
        # Set up plot
        plt.figure(figsize=(12, 8))
        
        # Extract position data
        positions = [site['position'] for site in self.final_modification_sites]
        
        # Count frequency of positions
        position_counts = Counter(positions)
        
        # Sorted positions and counts
        sorted_positions = sorted(position_counts.keys())
        counts = [position_counts[pos] for pos in sorted_positions]
        
        # Create bar plot
        plt.bar(sorted_positions, counts, width=0.8, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels
        for i, (pos, count) in enumerate(zip(sorted_positions, counts)):
            plt.text(pos, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Configure axes and title
        plt.xlabel('tRNA Position (0-based)', fontsize=20)
        plt.ylabel('Number of Modifications', fontsize=20)
        plt.title(f'tRNA {self.mod_type} Modification Distribution', fontsize=22)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set x-axis ticks to show all positions
        plt.xticks(sorted_positions, fontsize=14)
        plt.yticks(fontsize=14)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if output_file is None:
            if self.pdf_output:
                output_file = self.pdf_output
            elif self.output_file:
                # Use base path of output TSV and append modification type
                output_dir = os.path.dirname(self.output_file)
                output_file = os.path.join(output_dir, f"tRNA_{self.mod_type}_modification_distribution.pdf")
            else:
                # Add modification type to default file name
                output_file = f"tRNA_{self.mod_type}_modification_distribution.pdf"
                
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        logger.info(f"Saved modification distribution plot to {output_file}")
        
    def save_results(self, output_file=None):
        """
        Save final modification sites to a TSV file.
        
        Parameters:
            output_file (str, optional): Output file path. If None, use the
                path provided at initialization.
        """
        if output_file is None:
            output_file = self.output_file
            
        if not output_file:
            logger.warning("No output file specified")
            return
            
        if not self.final_modification_sites:
            logger.warning("No modification sites found, please run map_to_yeast_reference() method first")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.final_modification_sites)
        
        # Save to TSV file
        df.to_csv(output_file, sep='\t', index=False)
        
        logger.info(f"Saved {len(self.final_modification_sites)} modification sites to {output_file}")
        
        return df
        
    def run_analysis(self):
        """
        Run the complete modification site analysis workflow.
        """
        # Step 1: parse all FASTA files
        self.parse_fasta_files()
        
        # Step 2: compare modified and unmodified sequences to identify sites
        self.identify_modifications()
        
        # Step 3: map unmodified sequences to yeast reference tRNAs
        self.map_to_yeast_reference()
        
        # Step 4: analyze modification sites and generate statistics
        self.analyze_modifications()
        
        # Step 5: save results
        if self.output_file:
            self.save_results()
            
            # Step 6: plot modification site distribution
            self.plot_modification_distribution()
