# Instructions for MLDL23-FL

This is a set of instructions for getting started with the project.

## Clone

First you need to clone the project into whatever directory you prefer.

```console
git clone https://github.com/buttaciemanuel/mldl23-fl
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
mkdir -p data/femnist
```

Download zipped file.

```console
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AXJ5uuswGkv9dzVGMAJGRbGViWZFgImE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AXJ5uuswGkv9dzVGMAJGRbGViWZFgImE" -O compressed.zip && rm -rf /tmp/cookies.txt
```

Unzip file and remove zipped dataset.

```console
unzip compressed.zip -d data/femnist/ 
rm -rf compressed.zip
```

## Access to Weights & Biases

Execute the following command and insert this API key `0f5f688303bf29bb93b7ec05b9550e3c5efdb7fe`.

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
