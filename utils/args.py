import argparse


def get_parser():
    epilog = "note: argument \'--lrs\' accept different learning rate scheduling choices (\'exp\' or \'step\') followed by decaying factor and decaying period\n\n" \
        "examples:\n\n" \
        ">>> python3 experiments/script.py --bs 256 --lr 0.1 --m 0.9 --wd 0.0001 --num_rounds 1000 --num_epochs 1 --lrs exp 0.5\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 256\n" \
        " [+] learning rate: 0.1 decaying exponentially with multiplicative factor 0.5 every central round\n" \
        " [+] SGD momentum: 0.9\n" \
        " [+] SGD weight decay penalty: 0.0001\n" \
        " [+] running server rounds for training and validation: 1000\n" \
        " [+] running local client epoch: 1\n\n" \
        ">>> python3 experiments/script.py --bs 512 --lr 0.01 --num_epochs 5 --num_rounds 1000 --lrs step 0.75 3\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 decaying using step function with multiplicative factor 0.75 every 3 central rounds\n" \
        " [+] running server rounds for training and validation: 1000\n" \
        " [+] running local client epoch: 5\n\n" \
        ">>> python3 experiments/script.py --niid --bs 512 --lr 0.01 --num_epochs 5 --num_rounds 500 --lrs onecycle 0.1\n\n" \
        "This command executes the experiment using\n" \
        " [+] dataset distribution: niid (unbalanced across clients)\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 with one cycle cosine annealing rising up to a peak of 0.1 and then decreasing\n" \
        " [+] running server rounds for training and validation: 500\n" \
        " [+] running local client epoch: 5"
    parser = argparse.ArgumentParser(
        usage='run experiment on baseline FEMNIST dataset (federated, so decentralized) with a CNN architecture',
        description='This program is used to log to Weights & Biases training and validation results\nevery epoch of training on the EMNIST dataset. The employed architecture is a\nconvolutional neural network with two convolutional blocks and a fully connected layer.\nStochastic gradient descent is used by default as optimizer along with cross entropy loss.',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist'], default='femnist', help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], default='cnn', help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--selection', choices=['uniform', 'hybrid', 'poc'], default='uniform', type=str, help='criterion for selecting partecipating clients each round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--lrs', metavar=('scheduler', 'params'), nargs='+', type=str, default=['none'], help='Learning rate decay scheduling')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')
    return parser
