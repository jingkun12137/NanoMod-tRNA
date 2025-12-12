#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in1', required=True, help='Path to first prediction TSV')
    parser.add_argument('--in2', required=True, help='Path to second prediction TSV')
    parser.add_argument('--out', required=True, help='Output TSV path')
    args = parser.parse_args()

    if not os.path.exists(args.in1):
        print(f'Error: file not found: {args.in1}')
        sys.exit(1)
    if not os.path.exists(args.in2):
        print(f'Error: file not found: {args.in2}')
        sys.exit(1)

    df1 = pd.read_csv(args.in1, sep='\t')
    df2 = pd.read_csv(args.in2, sep='\t')

    if 'site_id' not in df1.columns or 'prediction' not in df1.columns:
        print('Error: input1 must contain columns: site_id, prediction')
        sys.exit(1)
    if 'site_id' not in df2.columns or 'prediction' not in df2.columns:
        print('Error: input2 must contain columns: site_id, prediction')
        sys.exit(1)

    a = df1[['site_id', 'prediction']].rename(columns={'prediction': 'prediction_1'})
    b = df2[['site_id', 'prediction']].rename(columns={'prediction': 'prediction_2'})
    m = a.merge(b, on='site_id', how='inner')

    m['prediction'] = (pd.to_numeric(m['prediction_1'], errors='coerce') +
                       pd.to_numeric(m['prediction_2'], errors='coerce')) / 2.0
    m = m.dropna(subset=['prediction'])

    out_cols = ['site_id', 'prediction']

    extra_cols = []
    for c in ['seq.name', 'seq.pos', 'structure']:
        if c in df1.columns:
            extra_cols.append(c)
    if extra_cols:
        base = df1[['site_id'] + extra_cols].drop_duplicates('site_id')
        m = m.merge(base, on='site_id', how='left')
        out_cols = ['site_id'] + extra_cols + ['prediction']

    m['modified'] = (m['prediction'] >= 0.5).astype(int)
    out_cols = out_cols + ['modified']

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    m[out_cols].to_csv(args.out, sep='\t', index=False)
    print(f'Saved averaged predictions to: {args.out}  (n={len(m)})')


if __name__ == '__main__':
    main()
