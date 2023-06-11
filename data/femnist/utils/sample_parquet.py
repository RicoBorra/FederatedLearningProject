'''
samples from all raw data;
by default samples in a non-iid manner; namely, randomly selects users from 
raw data until their cumulative amount of data exceeds the given number of 
datapoints to sample (specified by --fraction argument);
ordering of original data points is not preserved in sampled data
'''

import argparse
import json
import os
import random
import time
import pandas as pd

from collections import OrderedDict

from constants import DATASETS, SEED_FILES
from util import iid_divide

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                help='name of dataset to parse; default: sent140;',
                type=str,
                choices=DATASETS,
                default='sent140')
parser.add_argument('--iid',
                help='sample iid;',
                action="store_true")
parser.add_argument('--niid',
                help="sample niid;",
                dest='iid', action='store_false')
parser.add_argument('--fraction',
                help='fraction of all data to sample; default: 0.1;',
                type=float,
                default=0.1)
parser.add_argument('--u',
                help=('number of users in iid data set; ignored in niid case;'
                      'represented as fraction of original total number of users; '
                      'default: 1.0;'),
                type=float,
                default=1.0)
parser.add_argument('--seed',
                help='seed for random sampling of data',
                type=int,
                default=None)
parser.set_defaults(iid=False)

args = parser.parse_args()

print('------------------------------')
print('sampling data')

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(parent_path, args.name, 'data')
subdir = os.path.join(data_dir, 'all_data')
files = os.listdir(subdir)
files = [f for f in files if f.endswith('.parquet')]

rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
print ("Using seed {}".format(rng_seed))
rng = random.Random(rng_seed)
print (os.environ.get('LEAF_DATA_META_DIR'))
if os.environ.get('LEAF_DATA_META_DIR') is not None:
    seed_fname = os.path.join(os.environ.get('LEAF_DATA_META_DIR'), SEED_FILES['sampling'])
    with open(seed_fname, 'w+') as f:
        f.write("# sampling_seed used by sampling script - supply as "
                "--smplseed to preprocess.sh or --seed to utils/sample.py\n")
        f.write(str(rng_seed))
    print ("- random seed written out to {file}".format(file=seed_fname))
else:
    print ("- using random seed '{seed}' for sampling".format(seed=rng_seed))

new_user_count = 0 # for iid case
for f in files:
    file_dir = os.path.join(subdir, f)
    data = pd.read_parquet(file_dir)
   
    num_users = len(data['user'].unique())

    tot_num_samples = len(data)
    num_new_samples = int(args.fraction * tot_num_samples)

    # TODO: hierarchies support
    hierarchies = None

    if(args.iid):

        num_new_users = int(round(args.u * num_users))
        if num_new_users == 0:
            num_new_users += 1

        indices = range(tot_num_samples)
        new_indices = rng.sample(indices, num_new_samples)
        users = [str(i+new_user_count) for i in range(num_new_users)]

        user_data = {}
        data_no_user = data.drop(columns=['user'])
        sampled_data = data_no_user.iloc[new_indices]

        sample_groups = iid_divide(sampled_data, num_new_users)
        
        for i in range(num_new_users):
            user_data[users[i]] = pd.DataFrame(sample_groups[i])
            user_data[users[i]]['user'] = users[i]
        
        new_user_count += num_new_users

    else:

        ctot_num_samples = 0

        users = data['user'].unique()
        rng.shuffle(users)
        user_i = 0
        num_samples = []
        user_data = {}

        while(ctot_num_samples < num_new_samples):
            hierarchy = None
            user = users[user_i]

            cdata = data[data['user'] == user]
            cnum_samples = len(cdata)


            if (ctot_num_samples + cnum_samples > num_new_samples):
                cnum_samples = num_new_samples - ctot_num_samples
                indices = range(cnum_samples)
                new_indices = rng.sample(indices, cnum_samples)
                cdata = cdata.iloc[new_indices]
            
            user_data[user] = cdata

            ctot_num_samples += cnum_samples
            user_i += 1

        users = users[:user_i]
    
    # ------------
    # create .parquet file

    all_data = pd.concat(user_data.values(), axis=0)

    slabel = ''
    if(args.iid):
        slabel = 'iid'
    else:
        slabel = 'niid'

    arg_frac = str(args.fraction)
    arg_frac = arg_frac[2:]
    arg_nu = str(args.u)
    arg_nu = arg_nu[2:]
    arg_label = arg_frac
    if(args.iid):
        arg_label = '%s_%s' % (arg_nu, arg_label)
    file_name = '%s_%s_%s.parquet' % ((f[:-8]), slabel, arg_label)
    ouf_dir = os.path.join(data_dir, 'sampled_data', file_name)

    print('writing %s' % file_name)
    all_data.to_parquet(ouf_dir, index=False)
