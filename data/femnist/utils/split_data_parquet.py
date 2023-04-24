'''
splits data into train and test sets
'''

import argparse
import json
import os
import random
import time
import sys
import pandas as pd

from collections import OrderedDict

from constants import DATASETS, SEED_FILES

def create_parquet_for(user_files, which_set, max_users, include_hierarchy):
    """used in split-by-user case"""
    user_count = 0
    parquet_index = 0
    prev_dir = None

    data = []

    for (i, t) in enumerate(user_files):
        (u, f) = t

        file_dir = os.path.join(subdir, f)
        if prev_dir != file_dir:
            df = pd.read_parquet(file_dir)
            prev_dir = file_dir
        data.append(df[df['user'] == u])

        user_count += 1
    all_data = pd.concat(data, axis=0)

    if (user_count == max_users) or (i == len(user_files) - 1):

        data_i = f.find('data')
        num_i = data_i + 5
        num_to_end = f[num_i:]
        param_i = num_to_end.find('_')
        param_to_end = '.parquet'
        if param_i != -1:
            param_to_end = num_to_end[param_i:]
        nf = '%s_%d%s' % (f[:(num_i-1)], parquet_index, param_to_end)
        file_name = '%s_%s_%s.parquet' % ((nf[:-5]), which_set, arg_label)
        ouf_dir = os.path.join(dir, which_set, file_name)

        print('writing %s' % file_name)
        all_data.to_parquet(ouf_dir, index=False)

        user_count = 0
        parquet_index += 1
        data = []

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                help='name of dataset to parse; default: sent140;',
                type=str,
                choices=DATASETS,
                default='sent140')
parser.add_argument('--by_user',
                help='divide users into training and test set groups;',
                dest='user', action='store_true')
parser.add_argument('--by_sample',
                help="divide each user's samples into training and test set groups;",
                dest='user', action='store_false')
parser.add_argument('--frac',
                help='fraction in training set; default: 0.9;',
                type=float,
                default=0.9)
parser.add_argument('--seed',
                help='seed for random partitioning of test/train data',
                type=int,
                default=None)

parser.set_defaults(user=False)

args = parser.parse_args()

print('------------------------------')
print('generating training and test sets')

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
dir = os.path.join(parent_path, args.name, 'data')
subdir = os.path.join(dir, 'rem_user_data')
files = []
if os.path.exists(subdir):
    files = os.listdir(subdir)
if len(files) == 0:
    subdir = os.path.join(dir, 'sampled_data')
    if os.path.exists(subdir):
        files = os.listdir(subdir)
if len(files) == 0:
    subdir = os.path.join(dir, 'all_data')
    files = os.listdir(subdir)
files = [f for f in files if f.endswith('.parquet')]

rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
rng = random.Random(rng_seed)
if os.environ.get('LEAF_DATA_META_DIR') is not None:
    seed_fname = os.path.join(os.environ.get('LEAF_DATA_META_DIR'), SEED_FILES['split'])
    with open(seed_fname, 'w+') as f:
        f.write("# split_seed used by sampling script - supply as "
                "--spltseed to preprocess.sh or --seed to utils/split_data.py\n")
        f.write(str(rng_seed))
    print ("- random seed written out to {file}".format(file=seed_fname))
else:
    print ("- using random seed '{seed}' for sampling".format(seed=rng_seed))

arg_label = str(args.frac)
arg_label = arg_label[2:]

# TODO: support hierarchies
# check if data contains information on hierarchies
include_hierarchy = False

if (args.user):
    print('splitting data by user')

    # 1 pass through all the parquet files to instantiate arr
    # containing all possible (user, .parquet file name) tuples
    user_files = []
    for f in files:
        file_dir = os.path.join(subdir, f)
        data = pd.read_parquet(file_dir)
        user_files.extend([(u, f) for u in data['user'].unique()])

    # randomly sample from user_files to pick training set users
    num_users = len(user_files)
    num_train_users = int(args.frac * num_users)
    indices = range(num_users)
    train_indices = rng.sample(indices, num_train_users)
    train_blist = [False for i in range(num_users)]  # FIXME 
    for i in train_indices:
        train_blist[i] = True
    train_user_files = []
    test_user_files = []
    for i in range(num_users):
        if (train_blist[i]):
            train_user_files.append(user_files[i])
        else:
            test_user_files.append(user_files[i])

    max_users = sys.maxsize
    if args.name == 'femnist':
        max_users = 50 # max number of users per json file
    create_parquet_for(train_user_files, 'train', max_users, include_hierarchy)
    create_parquet_for(test_user_files, 'test', max_users, include_hierarchy)

else:
    print('splitting data by sample')

    for f in files:
        file_dir = os.path.join(subdir, f)
        data = pd.read_parquet(file_dir)

        user_data_train = []
        user_data_test = []
        user_indices = [] # indices of users in data['users'] that are not deleted

        removed = 0
        users = data['user'].unique()
        for i, u in enumerate(users):

            data_of_user = data[data['user'] == u]
            curr_num_samples = len(data_of_user)
            if curr_num_samples >= 2:
                # ensures number of train and test samples both >= 1
                num_train_samples = max(1, int(args.frac * curr_num_samples))
                if curr_num_samples == 2:
                    num_train_samples = 1

                num_test_samples = curr_num_samples - num_train_samples

                indices = range(curr_num_samples)
                if args.name in ['shakespeare']:
                    train_indices = range(num_train_samples)
                    test_indices = range(num_train_samples + 80 - 1, curr_num_samples)
                else:
                    train_indices = rng.sample(indices, num_train_samples)
                    test_indices = [i for i in range(curr_num_samples) if i not in train_indices]

                if len(train_indices) >= 1 and len(test_indices) >= 1:
                    user_indices.append(i)

                    train_blist = [False for _ in range(curr_num_samples)]
                    test_blist = [False for _ in range(curr_num_samples)]

                    for j in train_indices:
                        train_blist[j] = True
                    for j in test_indices:
                        test_blist[j] = True

                    user_data_train.append(data_of_user[train_blist])
                    user_data_test.append(data_of_user[test_blist])

        all_data_train = pd.concat(user_data_train, axis=0)
        all_data_test = pd.concat(user_data_test, axis=0)

        file_name_train = '%s_train_%s.parquet' % ((f[:-8]), arg_label)
        file_name_test = '%s_test_%s.parquet' % ((f[:-8]), arg_label)
        ouf_dir_train = os.path.join(dir, 'train', file_name_train)
        ouf_dir_test = os.path.join(dir, 'test', file_name_test)
        print('writing %s' % file_name_train)
        all_data_train.to_parquet(ouf_dir_train, index=False)
        print('writing %s' % file_name_test)
        all_data_test.to_parquet(ouf_dir_test, index=False)

