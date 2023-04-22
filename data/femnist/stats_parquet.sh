#!/usr/bin/env bash

NAME="femnist"

cd ./utils

python3 stats_parquet.py --name $NAME

cd ..