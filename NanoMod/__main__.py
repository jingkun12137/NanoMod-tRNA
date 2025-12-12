#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NanoMod-tRNA Main Module v0.9.6

Main entry point for the NanoMod-tRNA package. Provides the following command-line interfaces:
- train: Train the Attention MIL model with Adaptive Training Strategy and Structure-Aware Balancing.
- predict: Predict tRNA modifications using a trained model.
- mistake: Detect and analyze base mismatches in feature files.
- modification: Analyze modification sites from modified and unmodified tRNA sequences.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
# Avoid importing heavy dependencies (e.g., matplotlib/seaborn) at module import time.
# classify_trna_structure will be imported on demand when running the 'mistake' subcommand.

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NanoMod')

def main():
    """Main entry point for NanoMod-tRNA"""
    parser = argparse.ArgumentParser(description='NanoMod-tRNA v0.9.6: Attention MIL with Adaptive Training Strategy and Structure-Aware Balancing for tRNA modification detection')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Default values for frequently used parameters
    default_num_workers = 128
    default_batch_size = 256
    default_epochs = 300
    default_learning_rate = 0.003
    default_fasta = "NanoMod_data/yeast.tRNA.ext.fa"
    
    # Create the parser for the "train" command
    train_parser = subparsers.add_parser('train', help='Train NanoMod-tRNA model (Attention MIL + Adaptive Training Strategy + Structure-Aware Balancing)')
    train_parser.add_argument('--train-file', type=str, required=True,
                            help='Path to training features TSV file')
    train_parser.add_argument('--val-file', type=str, required=True,
                            help='Path to validation features TSV file')
    train_parser.add_argument('--mod-site-file', type=str, required=True,
                            help='Path to known modification sites TSV file (candidate sites)')
    train_parser.add_argument('--output-dir', type=str, default='NanoMod_model',
                            help='Directory to save trained model, training curves, and AUROC/AUPRC plots (default: NanoMod_model)')
    train_parser.add_argument('--batch-size', type=int, default=default_batch_size,
                            help=f'Batch size for training (default: {default_batch_size})')
    train_parser.add_argument('--epochs', type=int, default=default_epochs,
                            help=f'Number of training epochs (default: {default_epochs})')
    train_parser.add_argument('--learning-rate', type=float, default=default_learning_rate,
                            help=f'Learning rate (default: {default_learning_rate}). Learning rate decay occurs in the last 30%% of epochs with patience of 20')
    train_parser.add_argument('--kmer-nums', type=int, default=781,
                            help='Compatibility parameter, not used (default: 781)')
    train_parser.add_argument('--num-reads', type=int, default=30,
                            help='Number of reads per site for MIL (default: 30)')
    train_parser.add_argument('--dropout-rate', type=float, default=0.3,
                            help='Dropout rate for model regularization (default: 0.3)')
    train_parser.add_argument('--num-workers', type=int, default=default_num_workers,
                            help=f'Number of data loader worker threads (default: {default_num_workers})')
    train_parser.add_argument('--gpu', type=int, default=0, 
                            help='GPU device ID to use (default: 0, use -1 for CPU)')
    train_parser.add_argument('--seed', type=int, default=42, 
                            help='Random seed for reproducibility (default: 42)')
    train_parser.add_argument('--use-preprocessed', action='store_true', 
                            help='Use preprocessed data to accelerate training (optional)')
    train_parser.add_argument('--mod-type', type=str, default='D',
                            help='Modification type (default: D) - used for file naming and model selection')
    # Adaptive training strategy options
    train_parser.add_argument('--mismatch-threshold', type=float, default=1.5,
                            help='Mismatch rate ratio threshold for training mode selection (default: 1.5). '
                                 'If candidate_mismatch_rate > threshold × overall_mismatch_rate, use Mode A (Bayesian soft labels); '
                                 'otherwise use Mode B (hard labels)')
    
    # Create the parser for the "predict" command
    predict_parser = subparsers.add_parser('predict', help='Predict tRNA modifications using trained Attention MIL model (open-world prediction, no candidate list required)')
    predict_parser.add_argument('--features', type=str, required=True,
                             help='Path to features TSV file for prediction')
    predict_parser.add_argument('--model', type=str, default='NanoMod_model/best_model.pt',
                             help='Path to trained model file (default: auto-select based on --mod-type)')
    predict_parser.add_argument('--output', type=str, default=None,
                             help='Path to site-level prediction TSV file (default: NanoMod_output/{MOD}_modification_predictions.tsv)')
    predict_parser.add_argument('--save-read-level', action='store_true',
                             help='Also save read-level modification probabilities to a separate TSV file')
    predict_parser.add_argument('--read-output', type=str, default=None,
                             help='Path to read-level prediction TSV file (required if --save-read-level is set)')
    predict_parser.add_argument('--num-instances', type=int, default=30,
                             help='Number of reads per site used by MIL (default: 30, should match training)')
    predict_parser.add_argument('--num-workers', type=int, default=default_num_workers,
                             help=f'Number of data loader worker threads (default: {default_num_workers})')
    predict_parser.add_argument('--mod-type', type=str, default='D',
                             help='Modification type (default: D) - used to auto-select model file')
    predict_parser.add_argument('--threshold', type=float, default=None,
                             help='Deprecated: prediction uses fixed threshold 0.5 for binary classification')

    
    # Create the parser for the "mistake" command
    mistake_parser = subparsers.add_parser('mistake', help='Process tRNA mismatches')
    mistake_parser.add_argument('--features', type=str, required=True,
                              help='Path to features TSV file')
    mistake_parser.add_argument('--fasta', type=str, default=default_fasta,
                              help=f'Path to reference FASTA file (default: {default_fasta})')
    mistake_parser.add_argument('--output', type=str, required=True,
                              help='Path to output CSV file for mismatch summary')
    mistake_parser.add_argument('--pdf-output', type=str, default=None,
                              help='Path to output PDF file for visualization of mismatches')
    
    
    # Create the parser for the "modification" command
    modification_parser = subparsers.add_parser('modification', help='Analyze tRNA modification sites')
    modification_parser.add_argument('--modified', type=str, required=True,
                                help='Modified tRNA FASTA file')
    modification_parser.add_argument('--unmodified', type=str, required=True,
                                help='Unmodified tRNA FASTA file')
    modification_parser.add_argument('--reference', type=str, default=default_fasta,
                                help='Reference tRNA FASTA file')
    modification_parser.add_argument('--output', type=str, default='tRNA_modification_sites.tsv',
                                help='Output file for modification sites')
    modification_parser.add_argument('--mod-type', type=str, default='D',
                                help='Modification type to search for (default: D for dihydrouridine)')
    modification_parser.add_argument('--identity-threshold', type=float, default=0.8,
                                help='Minimum identity score for alignment')
    modification_parser.add_argument('--no-predict', action='store_true',
                                help='Skip prediction of modification sites for unannotated tRNAs')
    modification_parser.add_argument('--pdf-output', type=str,
                                help='PDF output file for modification distribution plot (default: based on mod-type)')
    # Parse arguments
    if len(sys.argv) <= 1:
        # Default to predict if no arguments provided
        sys.argv.append('predict')
    
    args = parser.parse_args()
    
    try:
        start_time = datetime.now()
        
        if args.command == 'train':
            # Use v4-style Easy Ensemble training (default)
            from .train import train_model
            
            logger.info("Starting NanoMod-tRNA training (Attention MIL with Adaptive Strategy + Structure-Aware Balancing)")
            model_path = train_model(
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
            
            logger.info(f"NanoMod-tRNA training completed successfully!")
            logger.info(f"Model saved to: {model_path}")
            
        elif args.command == 'predict':
            # Open-world prediction only; candidate masking not supported
            
            # Determine model path
            if args.model == 'NanoMod_model/best_model.pt':
                # Auto-detect model based on modification type
                model_path = f'NanoMod_model/{args.mod_type}_best_model.pt'
                
                if not os.path.exists(model_path):
                    logger.error(f"No trained model found for modification type {args.mod_type}")
                    logger.error(f"Looked for: {model_path}")
                    sys.exit(1)
            else:
                model_path = args.model

            # Resolve site-level output path
            if args.output is None:
                site_output_file = os.path.join('NanoMod_output', f'{args.mod_type}_modification_predictions.tsv')
            else:
                site_output_file = args.output

            # Validate read-level output options
            if args.save_read_level and not args.read_output:
                logger.error("--read-output must be provided when --save-read-level is set")
                sys.exit(1)
            read_output_file = args.read_output if args.save_read_level else None
            
            logger.info(f"Predicting tRNA modifications with the following parameters:")
            logger.info(f"- Features: {args.features}")
            logger.info(f"- Model: {model_path}")
            logger.info(f"- Site-level output: {site_output_file}")
            if args.save_read_level:
                logger.info(f"- Read-level output: {read_output_file}")
            logger.info(f"- Workers: {args.num_workers}")
            logger.info(f"- Modification type: {args.mod_type}")
            if args.threshold is not None:
                logger.info(f"- Threshold: {args.threshold}")
            
            # Use single model prediction (v4 style)
            from .predict import pure_predict
            
            results = pure_predict(
                model_path=model_path,
                features_file=args.features,
                site_output_file=site_output_file,
                num_instances=args.num_instances,
                num_workers=args.num_workers,
                mod_type=args.mod_type,
                save_read_level=args.save_read_level,
                read_output_file=read_output_file,
            )
            
            num_sites = len(results['site_ids']) if results['site_ids'] is not None else 0
            logger.info(f"Prediction completed successfully for {num_sites} sites")
            logger.info(f"Site-level results saved to: {results['output_file']}")
            if args.save_read_level:
                logger.info(f"Read-level results saved to: {read_output_file}")
            
        elif args.command == 'mistake':
            # Import necessary functions for mistake handling
            import pandas as pd
            import numpy as np
            from collections import defaultdict
            import pysam
            # 延迟导入，避免 -h 时触发 matplotlib/seaborn 依赖
            from .utils import classify_trna_structure
            
            alphabet = "ACGTdi"  # se
            base2name = {"A": "A", "C": "C", "G": "G", "T": "T",
                        "i": "insertions", "d": "deletions"}
            base2index = {b: i for i, b in enumerate(alphabet)}
            for i, b in enumerate(alphabet.lower()): base2index[b] = i
            # code N as A
            base2index["N"] = 0

            def _match(refi, readi, bases): return refi+bases, readi+bases, True
            def _insertion(refi, readi, bases): return refi, readi+bases, False
            def _deletion(refi, readi, bases): return refi+bases, readi, False
            def _skip(refi, readi, bases): return refi, readi, False
            code2function = {0: _match, 7: _match, 8: _match, 1: _insertion, 6: _insertion,
                            2: _deletion, 3: _deletion, 4: _insertion, 5: _skip}

            def store_blocks(a, start, end, baseq, i, calls):
                """Store base calls from aligned blocks. INDEL aware."""
                # process read blocks and store bases counts and indels as well
                readi, refi = 0, a.pos
                for ci, (code, bases) in enumerate(a.cigar):
                    prefi, preadi = refi, readi
                    refi, readi, data = code2function[code](refi, readi, bases)
                    # skip if current before start
                    if refi<=start: continue
                    # typical alignment part
                    if data:
                        if prefi<start:
                            bases -= start-prefi
                            preadi += start-prefi
                            prefi = start
                        if refi>end: bases -= refi-end
                        if bases<1: break
                        for ii, (b, q) in enumerate(zip(a.seq[preadi:preadi+bases], a.query_qualities[preadi:preadi+bases])):
                            if q>=baseq and b in base2index:
                                calls[prefi-start+ii, i, base2index[b]] += 1
                    elif start<prefi<end:
                        # insertion
                        if code==1: calls[prefi-start, i, 5] += 1
                        # deletion
                        elif code==2: calls[prefi-start, i, 4] += 1
                return calls
            
            def load_reference_sequences(fasta_file):
                """
                Load reference sequences from FASTA file.
                
                Args:
                    fasta_file: Path to FASTA file
                    
                Returns:
                    ref_sequences: Dictionary of reference sequences
                """
                ref_sequences = {}
                
                with open(fasta_file, 'r') as f:
                    current_seq_name = None
                    current_seq = ""
                    
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            # Save previous sequence if exists
                            if current_seq_name is not None:
                                ref_sequences[current_seq_name] = current_seq
                            
                            # Start new sequence
                            current_seq_name = line[1:]  # Remove '>'
                            current_seq = ""
                        else:
                            current_seq += line
                    
                    # Save last sequence
                    if current_seq_name is not None:
                        ref_sequences[current_seq_name] = current_seq
                
                return ref_sequences
            
            def parse_cigar_string(cigar_str):
                """
                Parse CIGAR string to tuples (code, bases).
                
                Args:
                    cigar_str: CIGAR string (e.g., "10M1I5M")
                    
                Returns:
                    cigar: List of tuples (code, bases)
                """
                import re
                cigar_tuples = []
                # Extract operation and length
                pattern = re.compile(r'(\d+)([MIDNSHP=X])')
                for length, operation in pattern.findall(cigar_str):
                    length = int(length)
                    if operation == 'M':
                        code = 0  # match
                    elif operation == 'I':
                        code = 1  # insertion
                    elif operation == 'D':
                        code = 2  # deletion
                    elif operation == 'N':
                        code = 3  # skip
                    elif operation == 'S':
                        code = 4  # soft clip
                    elif operation == 'H':
                        code = 5  # hard clip
                    elif operation == 'P':
                        code = 6  # padding
                    elif operation == '=':
                        code = 7  # sequence match
                    elif operation == 'X':
                        code = 8  # sequence mismatch
                    else:
                        continue  # Unknown operation
                    cigar_tuples.append((code, length))
                return cigar_tuples
            
            def check_base_match_with_cigar(row, ref_sequences):
                """
                Check if the base at seq.pos matches the reference using CIGAR string for proper alignment.
                Uses the data from the features file row.
                
                Args:
                    row: Row from features DataFrame containing bam_seq, bam_cigar, etc.
                    ref_sequences: Dictionary of reference sequences
                    
                Returns:
                    mistake_value: 0 if match, 1 if mismatch or not aligned
                """
                seq_name = row['seq.name']
                seq_pos = row['seq.pos']
                
                if seq_name not in ref_sequences:
                    return 1  # Reference not found
                
                ref_base = ref_sequences[seq_name][seq_pos] if seq_pos < len(ref_sequences[seq_name]) else 'N'
                
                # 计算read_base，不直接从row['base']获取
                if 'bam_seq' not in row or pd.isna(row['bam_seq']) or not row['bam_seq']:
                    return 1  # No sequence data
                
                # 使用与check_base_match_with_cigar相似的逻辑计算read_base
                read_seq = row['bam_seq']
                
                if 'bam_cigar' not in row or pd.isna(row['bam_cigar']) or not row['bam_cigar']:
                    return 1  # No CIGAR information
                    
                # Parse CIGAR string from the row
                try:
                    cigar_tuples = parse_cigar_string(row['bam_cigar'])
                except Exception as e:
                    logger.error(f"Error parsing CIGAR string: {row['bam_cigar']} - {str(e)}")
                    return 1
                
                # Map read positions to reference positions using CIGAR
                ref_index = int(row['bam_pos']) if 'bam_pos' in row and not pd.isna(row['bam_pos']) else 0
                read_index = 0
                
                # Keep track of the reference position we're interested in
                target_read_index = None
                
                # Process CIGAR operations
                for code, length in cigar_tuples:
                    # Call the appropriate function based on the CIGAR operation
                    next_ref_index, next_read_index, consume_both = code2function[code](ref_index, read_index, length)
                    
                    # If this operation includes our position of interest
                    if ref_index <= seq_pos < next_ref_index:
                        # Calculate the corresponding position in the read
                        if consume_both:
                            target_read_index = read_index + (seq_pos - ref_index)
                        else:
                            # Handle special cases (insertions, deletions, etc.)
                            if code == 1:  # Insertion
                                return 1  # Position in an insertion, consider as mismatch
                            elif code == 2 or code == 3:  # Deletion or skip
                                return 1  # Position in a deletion or skip, consider as mismatch
                    
                    # Update indices
                    ref_index, read_index = next_ref_index, next_read_index
                    
                    # If we've found our target, break
                    if target_read_index is not None:
                        break
                
                # If we didn't find a matching position
                if target_read_index is None or target_read_index >= len(read_seq):
                    return 1
                
                # Check for match between read and reference
                try:
                    read_base = read_seq[target_read_index].upper()
                    ref_base = ref_sequences[seq_name][seq_pos].upper()
                    return 0 if read_base == ref_base else 1
                except Exception as e:
                    logger.error(f"Error comparing bases: {str(e)}")
                    return 1
            
            def get_mismatch_type(row, ref_sequences):
                if row['mistake'] == 0:  # If there's a match
                    return 'match'
                
                seq_name = row['seq.name']
                seq_pos = row['seq.pos']
                
                if seq_name not in ref_sequences:
                    return 'unknown'
                
                ref_base = ref_sequences[seq_name][seq_pos] if seq_pos < len(ref_sequences[seq_name]) else 'N'
                
                # 计算read_base，不直接从row['base']获取
                if 'bam_seq' not in row or pd.isna(row['bam_seq']) or not row['bam_seq']:
                    return 'unknown'
                
                # 使用与check_base_match_with_cigar相似的逻辑计算read_base
                read_seq = row['bam_seq']
                
                if 'bam_cigar' not in row or pd.isna(row['bam_cigar']) or not row['bam_cigar']:
                    return 'unknown'
                
                try:
                    cigar_tuples = parse_cigar_string(row['bam_cigar'])
                except Exception:
                    return 'unknown'
                
                ref_index = int(row['bam_pos']) if 'bam_pos' in row and not pd.isna(row['bam_pos']) else 0
                read_index = 0
                target_read_index = None
                
                for code, length in cigar_tuples:
                    next_ref_index, next_read_index, consume_both = code2function[code](ref_index, read_index, length)
                    
                    if ref_index <= seq_pos < next_ref_index:
                        if consume_both:
                            target_read_index = read_index + (seq_pos - ref_index)
                        else:
                            return 'unknown'
                    
                    ref_index, read_index = next_ref_index, next_read_index
                    
                    if target_read_index is not None:
                        break
                
                if target_read_index is None or target_read_index >= len(read_seq):
                    return 'unknown'
                
                try:
                    read_base = read_seq[target_read_index].upper()
                    return read_base
                except Exception:
                    return 'unknown'
            
            def process_features_file(features_file, ref_fasta, output_csv, pdf_output):
                """
                Process features file to add mismatch information and generate summary CSV.
                
                Args:
                    features_file: Path to features TSV file
                    ref_fasta: Path to reference FASTA file
                    output_csv: Path to output CSV file
                    pdf_output: Path to output PDF file
                    
                Returns:
                    None
                """
                logger.info(f"Loading features from {features_file}")
                df = pd.read_csv(features_file, sep='\t')
                
                logger.info(f"Loading reference sequences from {ref_fasta}")
                ref_sequences = load_reference_sequences(ref_fasta)
                
                logger.info("Processing reads and checking mismatches using CIGAR strings...")
                # Add mistake column (0 for match, 1 for mismatch)
                df['mistake'] = df.apply(lambda row: check_base_match_with_cigar(row, ref_sequences), axis=1)
                
                # Let's also add a column for the type of mismatch
                df['mismatch_type'] = df.apply(lambda row: get_mismatch_type(row, ref_sequences), axis=1)
                
                # Add tRNA structure classification variable (1:Acceptor stem, 2:D-arm, 3:D-loop,
                # 4:Anticodon stem, 5:Anticodon loop, 6:Variable loop, 7:T-arm, 8:T-loop, 0:other)
                logger.info("Adding tRNA structure classification variable...")
                df['structure'] = df['seq.pos'].apply(classify_trna_structure)
                
                # Save updated features file
                df.to_csv(features_file, sep='\t', index=False)
                logger.info(f"Updated features file saved to {features_file}")
                
                # Generate summary statistics
                logger.info("Generating summary statistics...")
                summary = defaultdict(lambda: defaultdict(int))
                
                for _, row in df.iterrows():
                    key = (row['seq.name'], row['seq.pos'])
                    summary[key]['total'] += 1
                    summary[key]['match'] += (1 - row['mistake'])  # Now 0 means match
                    
                    # Count different types of mismatches
                    if row['mismatch_type'] not in ['match', 'unknown']:
                        mismatch_type = row['mismatch_type']
                        summary[key][mismatch_type] += 1
                
                # Convert to DataFrame for output
                summary_data = []
                for (seq_name, seq_pos), counts in summary.items():
                    total = counts['total']
                    match = counts['match']
                    mismatch = total - match
                    mismatch_rate = mismatch / total if total > 0 else 0
                    
                    entry = {
                        'seq.name': seq_name,
                        'seq.pos': seq_pos,
                        'total': total,
                        'match': match,
                        'mismatch': mismatch,
                        'mismatch_rate': mismatch_rate
                    }
                    
                    # Add counts for each type of mismatch (A, C, G, T)
                    for base in "ACGT":
                        entry[f'{base}_mismatch'] = counts.get(base, 0)
                        entry[f'{base}_mismatch_rate'] = counts.get(base, 0) / total if total > 0 else 0
                    
                    summary_data.append(entry)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.sort_values(['seq.name', 'seq.pos'], inplace=True)
                
                # Save summary CSV
                summary_df.to_csv(output_csv, index=False)
                logger.info(f"Summary statistics saved to {output_csv}")
                
                if pdf_output is not None:
                    logger.info(f"Generating visualization PDF at {pdf_output}")
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_pdf import PdfPages
                    import matplotlib.patches as mpatches
                    import numpy as np
                    
                    # Define colors for different bases and states
                    base_colors = {
                        'A': 'green',
                        'C': 'blue',
                        'G': 'orange',
                        'T': 'red',
                        'match': 'lightgrey',
                        'missing': 'black'  # Add black color for missing bases
                    }
                    
                    # Group by tRNA type
                    tRNA_groups = summary_df.groupby('seq.name')
                    
                    with PdfPages(pdf_output) as pdf:
                        for tRNA_name, tRNA_df in tRNA_groups:
                            logger.info(f"Creating visualization for {tRNA_name}")
                            
                            # Sort by position
                            tRNA_df = tRNA_df.sort_values('seq.pos')
                            
                            # Get max position to determine figure width
                            max_pos = tRNA_df['seq.pos'].max() + 1
                            
                            # Get reads for this tRNA type
                            tRNA_reads = df[df['seq.name'] == tRNA_name].copy()
                            
                            # Get unique read IDs - safely check if 'read.id' or 'aln.id' exists
                            id_column = 'aln.id' if 'aln.id' in tRNA_reads.columns else 'read.id'
                            if id_column not in tRNA_reads.columns:
                                logger.error(f"Error: Could not find read identifier column ('read.id' or 'aln.id') in dataset")
                                continue
                            
                            read_ids = tRNA_reads[id_column].unique()
                            
                            # Limit to a reasonable number of reads for visualization
                            if len(read_ids) > 100:
                                np.random.seed(42)  # For reproducibility
                                sampled_reads = np.random.choice(read_ids, 100, replace=False)
                                tRNA_reads = tRNA_reads[tRNA_reads[id_column].isin(sampled_reads)]
                            
                            # Number of reads to show
                            n_reads = len(tRNA_reads[id_column].unique())
                            
                            # Create figure with title showing the tRNA type and adapters
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(25, max_pos/3), max(7, n_reads/15 + 3)), 
                                                           gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
                            
                            # Use a fixed font size to avoid cumulative scaling inside the loop.
                            # Do not update global rcParams here to prevent font size from growing across pages.
                            
                            # Add title with tRNA information (no species name)
                            plt.suptitle(f'tRNA{tRNA_name} ({max_pos} nt)', fontsize=22)
                            
                            # Upper plot - positional statistics
                            bar_width = 0.8
                            positions = tRNA_df['seq.pos'].values
                            
                            # Create stacked bar for each position
                            bottom = np.zeros(len(positions))
                            
                            # Add mismatch types in order: A, C, G, T
                            for base in "ACGT":
                                values = tRNA_df[f'{base}_mismatch_rate'].values if f'{base}_mismatch_rate' in tRNA_df.columns else np.zeros(len(positions))
                                ax1.bar(positions, values, bar_width, bottom=bottom, color=base_colors[base], 
                                       label=f'{base} mismatch')
                                bottom += values
                            
                            # Add match rate on top (grey)
                            match_values = 1 - bottom  # Remaining height is matches
                            ax1.bar(positions, match_values, bar_width, bottom=bottom, color=base_colors['match'],
                                   label='Match to reference')
                            
                            ax1.set_ylabel('Biological', fontsize=18)
                            ax1.set_title('')  # Remove title from subplot
                            
                            # Add a legend on the right side
                            legend_elements = [
                                mpatches.Patch(color=base_colors['A'], label='A mismatch'),
                                mpatches.Patch(color=base_colors['G'], label='G mismatch'),
                                mpatches.Patch(color=base_colors['T'], label='T mismatch'),
                                mpatches.Patch(color=base_colors['C'], label='C mismatch'),
                                mpatches.Patch(color=base_colors['match'], label='Match to reference')
                            ]
                            ax1.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=14)
                            
                            # Add adapter labels as in the example
                            ax1.text(0, 1.05, "5' adapter (24 nt)", ha='left', va='bottom', fontsize=17, transform=ax1.transAxes)
                            ax1.text(1, 1.05, "3' adapter (30 nt)", ha='right', va='bottom', fontsize=17, transform=ax1.transAxes)
                            
                            # Add markers for important positions (like modifications)
                            # This is placeholder - actual positions would need to be determined from data
                            common_mod_positions = []
                            for pos in common_mod_positions:
                                if 0 <= pos < max_pos:
                                    ax1.axvline(x=pos, color='gray', linestyle='--', alpha=0.3)
                            
                            # Lower plot - individual reads
                            read_groups = tRNA_reads.groupby(id_column)
                            y_pos = 0
                            
                            # Prepare data for heatmap-like visualization
                            for read_id, read_group in read_groups:
                                read_group = read_group.sort_values('seq.pos')
                                
                                # Fill in the entire row with a background color
                                rect_bg = mpatches.Rectangle((-0.5, y_pos-0.4), max_pos, 0.8, 
                                                        color='lightgrey', alpha=0.3)
                                ax2.add_patch(rect_bg)
                                
                                # Plot each position's mismatch status
                                for _, row in read_group.iterrows():
                                    pos = row['seq.pos']
                                    
                                    # Determine color based on mismatch type
                                    if 'mismatch_type' in row:
                                        mismatch_type = row['mismatch_type']
                                    else:
                                        # If mismatch_type isn't available, derive it from mistake flag
                                        if 'mistake' in row and row['mistake'] == 0:
                                            mismatch_type = 'match'
                                        else:
                                            # Get the actual base from the sequence if possible
                                            mismatch_type = row.get('base', 'missing')
                                    
                                    color = base_colors.get(mismatch_type, base_colors['match'])
                                    
                                    # Plot rectangle for this position
                                    rect = mpatches.Rectangle((pos-0.4, y_pos-0.4), 0.8, 0.8, 
                                                            color=color, alpha=0.8)
                                    ax2.add_patch(rect)
                                
                                y_pos += 1
                            
                            # Set y-axis limits
                            ax2.set_ylim(-0.5, n_reads-0.5)
                            ax2.set_yticks([])  # Hide y-axis ticks
                            ax2.set_ylabel('Individual reads', fontsize=18)
                            
                            # Set x-axis labels and ticks
                            ax2.set_xlabel('Position', fontsize=18)
                            ax2.set_xlim(-0.5, max_pos-0.5)
                            ax2.set_xticks(range(0, max_pos, 5))
                            
                            # Add markers for common modification sites
                            # These positions would need to be determined from actual tRNA data
                            common_mod_positions = []
                            for pos in common_mod_positions:
                                if 0 <= pos < max_pos:
                                    ax2.axvline(x=pos, color='green', linestyle='-', alpha=0.3)
                            
                            # 调整刻度标签大小
                            ax1.tick_params(axis='both', which='major', labelsize=16)
                            ax2.tick_params(axis='both', which='major', labelsize=16)
                            
                            # Adjust layout and save to PDF
                            plt.tight_layout()
                            plt.subplots_adjust(top=0.9)  # Make room for suptitle
                            pdf.savefig(fig)
                            plt.close()
                            
                    logger.info(f"Visualization PDF saved to {pdf_output}")
            
            logger.info(f"Processing tRNA mismatches with the following parameters:")
            logger.info(f"- Features: {args.features}")
            logger.info(f"- Reference FASTA: {args.fasta}")
            logger.info(f"- Output CSV: {args.output}")
            logger.info(f"- PDF Output: {args.pdf_output}")
            
            # Process features file
            process_features_file(args.features, args.fasta, args.output, args.pdf_output)
            logger.info("Mismatch processing completed successfully!")
            
        elif args.command == 'modification':
            # Import the ModificationSiteAnalyzer class
            from .site_analysis import ModificationSiteAnalyzer
            
            logger.info(f"Analyzing tRNA modification sites with the following parameters:")
            logger.info(f"- Modified tRNA file: {args.modified}")
            logger.info(f"- Unmodified tRNA file: {args.unmodified}")
            logger.info(f"- Reference tRNA file: {args.reference}")
            logger.info(f"- Output file: {args.output}")
            logger.info(f"- Modification type: {args.mod_type}")
            if args.pdf_output:
                logger.info(f"- PDF output file: {args.pdf_output}")
            
            # 创建带输出文件路径的分析器
            analyzer = ModificationSiteAnalyzer(
                modified_fasta=args.modified, 
                unmodified_fasta=args.unmodified, 
                yeast_tRNA_fasta=args.reference,
                output_file=args.output,
                mod_type=args.mod_type,
                pdf_output=args.pdf_output
            )
            
            # 运行完整的分析流程
            analyzer.run_analysis()
            
            logger.info(f"Modification site analysis completed successfully. Results saved to {args.output}")
            
        else:
            parser.print_help()
        
        # Calculate elapsed time
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logger.info(f"Total execution time: {elapsed_time}")
        
    except KeyboardInterrupt:
        logger.error("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
