# Converts a list of (writer, [list of (file,class)]) tuples into a parquet file
# each row has the form:
# user, class, pixel1, pixel2, ..., pixel784

from __future__ import division
import json
import math
import numpy as np
import os
import sys
import pandas as pd

from PIL import Image

utils_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

import util

MAX_WRITERS = 100  # max number of writers per file.


def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90: # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

by_writer_dir = os.path.join(parent_path, 'data', 'intermediate', 'images_by_writer')
writers = util.load_obj(by_writer_dir)

rows = []

writer_count, all_writers = 0, 0
parquet_index = 0

for (w, l) in writers:

    #users.append(w)
    #num_samples.append(len(l))
    #user_data[w] = {'x': [], 'y': []}

    size = 28, 28  # original image size is 128, 128
    
    for (f, c) in l:
        file_path = os.path.join(parent_path, f)
        img = Image.open(file_path)
        gray = img.convert('L')
        gray.thumbnail(size, Image.ANTIALIAS)
        arr = np.asarray(gray).copy()
        vec = arr.flatten()
        vec = vec / 255  # scale all pixel values to between 0 and 1

        nc = relabel_class(c)

        row = w, nc, vec
        rows.append(row)

    writer_count += 1
    all_writers += 1
    
    if writer_count == MAX_WRITERS or all_writers == len(writers):

        df = pd.DataFrame(rows, columns=['user', 'y', 'x']).set_index('user')
        features = pd.DataFrame(df['x'].to_list(), columns=[f"x{i}" for i in range(1, size[0]*size[1] + 1)], index=df.index)
        df = pd.concat([df.drop(columns=['x']), features], axis=1)

        filename = f'all_data_{parquet_index}.parquet'
        print(f"writing {filename}")
        path = os.path.join(parent_path, 'data', 'all_data', filename)
        df.reset_index().to_parquet(path=path, index=False)

        writer_count = 0
        parquet_index += 1

        del df
        rows = []

print("Done")
