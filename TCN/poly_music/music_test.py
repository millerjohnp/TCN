import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from model import RNNModel
from utils import data_generator
import numpy as np


parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--stabilize', action='store_true',
                    help='use STABLE (default: False)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='JSB',
                    help='the dataset to run (default: Nott)')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='the model type to use')
parser.add_argument('--logdir', type=str, help="where to store shit")
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
input_size = 88
X_train, X_valid, X_test = data_generator(args.data)

nhid = args.nhid
dropout = args.dropout
rnn_type = args.rnn_type

model = RNNModel(rnn_type, input_size, input_size, nhid)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(X_data):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    for idx in eval_idx_list:
        data_line = X_data[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1-y), torch.log(1-output).float().t()))
        total_loss += loss.data[0]
        count += output.size(0)
    eval_loss = total_loss / count
    print("Validation/Test loss: {:.5f}".format(eval_loss))
    return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1 - y), torch.log(1 - output).float().t()))
        total_loss += loss.data[0]
        count += output.size(0)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        if args.stabilize:
            model.stabilize()
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            message = "Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss)
            print(message)
            with open(args.logdir, "a") as handle:
                handle.write(message + "\n")
            total_loss = 0.0
            count = 0


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "poly_music_{0}.pt".format(args.data)
    for ep in range(1, args.epochs+1):
        train(ep)
        vloss = evaluate(X_valid)
        tloss = evaluate(X_test)
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test)

