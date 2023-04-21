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
                      'default: 0.01;'),
                type=float,
                default=0.01)
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
        #raw_list = list(data['user_data'].values())
        #raw_x = [elem['x'] for elem in raw_list]
        #raw_y = [elem['y'] for elem in raw_list]
        #x_list = [item for sublist in raw_x for item in sublist] # flatten raw_x
        #y_list = [item for sublist in raw_y for item in sublist] # flatten raw_y

        num_new_users = int(round(args.u * num_users))
        if num_new_users == 0:
            num_new_users += 1

        #indices = [i for i in range(tot_num_samples)]
        indices = range(tot_num_samples)
        new_indices = rng.sample(indices, num_new_samples)
        users = [str(i+new_user_count) for i in range(num_new_users)]

        user_data = {}
        #for user in users:
        #    user_data[user] = {'x': [], 'y': []}
        #all_x_samples = [x_list[i] for i in new_indices]
        #all_y_samples = [y_list[i] for i in new_indices]
        sampled_data = data.drop(columns=['user'])[new_indices]
        sample_groups = iid_divide(sampled_data, num_new_users)
        #x_groups = iid_divide(all_x_samples, num_new_users)
        #y_groups = iid_divide(all_y_samples, num_new_users)
        for i in range(num_new_users):
            user_data[users[i]] = sample_groups[i]
        #    user_data[users[i]]['x'] = x_groups[i]
        #    user_data[users[i]]['y'] = y_groups[i]
        
        num_samples = [len(user_data[u]) for u in users]

        new_user_count += num_new_users

    else:

        ctot_num_samples = 0

        users = data['user'].unique()
        #users_and_hiers = None
        #if 'hierarchies' in data:
        #    users_and_hiers = list(zip(users, data['hierarchies']))
        #    rng.shuffle(users_and_hiers)
        #else:
        #    rng.shuffle(users)
        rng.shuffle(users)
        user_i = 0
        num_samples = []
        user_data = {}

        #if 'hierarchies' in data:
        #    hierarchies = []

        while(ctot_num_samples < num_new_samples):
            hierarchy = None
            #if users_and_hiers is not None:
            #    user, hier = users_and_hiers[user_i]
            #else:
            #    user = users[user_i]
            user = users[user_i]

            #cdata = data['user_data'][user]
            cdata = data[data['user'] == user]

            #cnum_samples = len(data['user_data'][user]['y'])
            cnum_samples = len(cdata)


            if (ctot_num_samples + cnum_samples > num_new_samples):
                cnum_samples = num_new_samples - ctot_num_samples
                #indices = [i for i in range(cnum_samples)]
                indices = range(cnum_samples)
                new_indices = rng.sample(indices, cnum_samples)
                #x = []
                #y = []
                #for i in new_indices:
                #    x.append(data['user_data'][user]['x'][i])
                #    y.append(data['user_data'][user]['y'][i])
                #cdata = {'x': x, 'y': y}
                cdata = cdata[new_indices]
            
            #if 'hierarchies' in data:
            #    hierarchies.append(hier)

            num_samples.append(cnum_samples)
            user_data[user] = cdata

            ctot_num_samples += cnum_samples
            user_i += 1

        #if 'hierarchies' in data:
        #    users = [u for u, h in users_and_hiers][:user_i]
        #else:
        #    users = users[:user_i]
        users = users[:user_i]

    # ------------
    # create .parquet file

    #all_data = {}
    #all_data['users'] = users
    #if hierarchies is not None:
    #    all_data['hierarchies'] = hierarchies
    #all_data['num_samples'] = num_samples
    #all_data['user_data'] = user_data
    #for user in users:
    #    user_data[user] = pd.concat([pd.Series(len(user_data[user])*user, name='user'),user_data[user]], axis=1)
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
    #with open(ouf_dir, 'w') as outfile:
    #    json.dump(all_data, outfile)
    all_data.to_parquet(ouf_dir, index=False)
