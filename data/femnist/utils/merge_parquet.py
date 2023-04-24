#!/bin/python3

import argparse
import gc
import pandas as pd
import os
import sys


def merge_parquet_files(input_directory: str, output: str):
    '''
    Merges all '.parquet' files in 'input_directory' in one single file 'output'.

    Parameters
    ----------
    input_directory: str,
        Directory containing all input parquet files
    output: str
        Output path of single parquet file

    Example
    -------
    >>> merge_parquet_files('data/femnist/data/niid/train', 'data/femnist/compressed/niid/training.parquet')
    '''

    frames = []
    print(f"loading dataframes from directory '{input_directory}' for single merge")
    # read single dataframes from parquet files
    for input_file in os.listdir(input_directory):
        print(f" * reading dataframe '{input_file}'...", end = ' ', flush = True)
        frames.append(pd.read_parquet(os.path.join(input_directory, input_file)))
        print('done')
    # merge dataframes
    print(f"merging all files into single output '{output}'...", end = ' ', flush = True)
    pd.concat(frames, axis = 0).set_index('user', drop = True).to_parquet(output)
    print('done')
    # frees RAM memory in order to avoid saturation
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = sys.argv[0],
        usage = f'Usage: {sys.argv[0]} <INPUT-DIRECTORY> --output <OUTPUT-FILE>',
        description = 'Merge all input parquet files into a single parquet output file'
    )
    # command options
    parser.add_argument('input', type = str, help = 'Input directory with parquet files')
    parser.add_argument('--output', type = str, required = True, help = 'Output file path')
    # get command line arguments
    args = parser.parse_args()
    # create output directory if it does not exists
    output_directory = os.path.dirname(args.output)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # merge all files
    merge_parquet_files(args.input, args.output)