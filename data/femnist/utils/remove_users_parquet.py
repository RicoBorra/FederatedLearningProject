
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
    #hierarchies = []
    # TODO: add hierarchies support
    num_samples = []
    user_data = {}

    file_dir = os.path.join(subdir, f)
    #with open(file_dir, 'r') as inf:
    #    data = json.load(inf)
    data = pd.read_parquet(file_dir)

    #num_users = len(data['user'])
    #for i in range(num_users):
    #    curr_user = data['users'][i]
    #    curr_hierarchy = None
        #if 'hierarchies' in data:
        #    curr_hierarchy = data['hierarchies'][i]
    #    curr_num_samples = data['num_samples'][i]
    #    if (curr_num_samples >= args.min_samples):
    #        user_data[curr_user] = data['user_data'][curr_user]
    #        users.append(curr_user)
            #if curr_hierarchy is not None:
            #    hierarchies.append(curr_hierarchy)
    #        num_samples.append(data['num_samples'][i])
    
    grouped_users_above = data.value_counts(subset=['user']) >= argparse.min_samples # true for users above threshold
    users = grouped_users_above[grouped_users_above.values].reset_index('user')['user'] # selects only users above threshold
    all_data = data[data['user'].isin(users)]

    #all_data = {}
    #all_data['users'] = users
    #if len(hierarchies) == len(users):
    #    all_data['hierarchies'] = hierarchies
    #all_data['num_samples'] = num_samples
    #all_data['user_data'] = user_data

    file_name = '%s_keep_%d.parquet' % ((f[:-5]), args.min_samples)
    ouf_dir = os.path.join(dir, 'rem_user_data', file_name)

    print('writing %s' % file_name)
    #with open(ouf_dir, 'w') as outfile:
    #    json.dump(all_data, outfile)
    all_data.to_parquet(ouf_dir, index=False)

