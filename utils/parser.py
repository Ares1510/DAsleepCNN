import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='model to train')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=3, help='patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    return parser.parse_args()