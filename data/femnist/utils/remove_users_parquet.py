
'''
removes users with less than the given number of samples
'''

import argparse
import json
import os
import pandas as pd

from constants import DATASETS

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                help='name of dataset to parse; default: sent140;',
                type=str,
                choices=DATASETS,
                default='sent140')

parser.add_argument('--min_samples',
                help='users with less than x samples are discarded; default: 10;',
                type=int,
                default=10)

args = parser.parse_args()

print('------------------------------')
print('removing users with less than %d samples' % args.min_samples)

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
dir = os.path.join(parent_path, args.name, 'data')
subdir = os.path.join(dir, 'sampled_data')
files = []
if os.path.exists(subdir):
    files = os.listdir(subdir)
if len(files) == 0:
    subdir = os.path.join(dir, 'all_data')
    files = os.listdir(subdir)
files = [f for f in files if f.endswith('.parquet')]

for f in files:
    users = []
    # TODO: add hierarchies support
    num_samples = []
    user_data = {}

    file_dir = os.path.join(subdir, f)
    data = pd.read_parquet(file_dir)

    grouped_users_above = data.value_counts(subset=['user'], sort=False) >= args.min_samples # true for users above threshold
    users = grouped_users_above[grouped_users_above.values].reset_index('user')['user'] # selects only users above threshold
    all_data = data[data['user'].isin(users)]

    file_name = '%s_keep_%d.parquet' % ((f[:-8]), args.min_samples)
    ouf_dir = os.path.join(dir, 'rem_user_data', file_name)

    print('writing %s' % file_name)
    all_data.to_parquet(ouf_dir, index=False)

