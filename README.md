# Instructions for MLDL23-FL

This is a set of instructions for getting started with the project.

## Clone

First you need to clone the project into whatever directory you prefer.

```console
git clone https://github.com/RicoBorra/FederatedLearningProject
```

Then, for the following operations, move into the project folder.

```console
cd mldl23-fl
```

## Environment creation

This creates the virtual enviroment within a directory `env`.

```console
python -m venv env
```

Then you can activate the environment.

```console
source env/bin/activate
```

Alternatively, in case of fish shell.

```console
source env/bin/activate.fish
```

## Install dependencies

Automatically install all required packages.

```console
pip install -r requirements.txt
```

## Download the dataset

Create dataset directory

```console
cd data/femnist
```

Run the following commands to generate the datasets (and download the necessary files if missing)

To generate the non-iid distribution
```console
./preprocess_parquet.sh -s niid --sf 1.0 -k 0 -t sample
```

To generate the iid distribution
```console
./preprocess_parquet.sh -s iid --sf 1.0 --u 1.0 -k 0 -t sample
```
Note that --u parameter is the correct one (even if official documentation says otherwise) to pass to the script in order to have the same number of clients for the iid case as for the non-iid one.


## Access to Weights & Biases

Execute the following command and insert this API key `b578cc4325e4b0652255efe8f2878be1d5fad2f2`.

```console
wandb login
```

This will provide automatic access and log to the Weights & Biases platform for model and experiments tracking.

## Run the application

Run the application and test the model.

```console
python main.py --niid --seed 0 --dataset femnist --model cnn \
    --num_rounds 1000 \
    --num_epochs 1 \
    --clients_per_round 10 \
    --print_train_interval 1000 --print_test_interval 1000 \
    --eval_interval 10 --test_interval 10
```

By tweaking the parameters, we can have multiple experiments and outcomes.
