import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from utils import data_generator
from model import RNNModel
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--num_layers', type=int, default=1,
                    help='# of layers (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer (default: 128)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
parser.add_argument('--stabilize', action='store_true',
                    help='use a stable LSTM (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
n_inputs = 1
nhid = args.nhid
dropout = args.dropout
seq_length = 784
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
model = RNNModel(rnn_type="LSTM", ntoken=n_classes, ninp=n_inputs, nhid=nhid,
                 nlayers=args.num_layers)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
#optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)


def train(ep):
    global steps
    train_loss = 0
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, 1, seq_length)
        if args.permute:
            data = data[:, :, permute]
        # Data should be seq_len, batch, input_size, 
        data = data.permute(2, 0, 1)
        data, target = Variable(data), Variable(target)
        model.zero_grad()
        output = model(data)
 
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()
#        for p in model.parameters():
#            p.data.add_(-lr, p.grad.data)

        if args.stabilize:
            # One set of weights satisfying stability requirement
            recur_weights = model.rnn.weight_hh_l0.data
            wi, wf, wz, wo = recur_weights.chunk(4, 0)
            
            trimmed_wi =  wi * 0.395  / torch.sum(torch.abs(wi), 0)
            trimmed_wf =  wf * 0.155  / torch.sum(torch.abs(wf), 0)
            trimmed_wz =  wz * 0.099  / torch.sum(torch.abs(wz), 0)
            trimmed_wo =  wo * 0.395 / torch.sum(torch.abs(wo), 0)
            new_recur_weights = torch.cat([trimmed_wi, trimmed_wf, trimmed_wz, trimmed_wo], 0)
            model.rnn.weight_hh_l0.data.set_(new_recur_weights)

            # Also trim the input to hidden weight for the forget gate
            ih_weights = model.rnn.weight_ih_l0.data
            ui, uf, uz, uo = ih_weights.chunk(4, 0)
            trimmed_uf =  uf * 0.25  / torch.sum(torch.abs(uf), 0)
            new_ih_weights = torch.cat([ui, trimmed_uf, uz, uo], 0)
            model.rnn.weight_ih_l0.data.set_(new_ih_weights)

        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.data[0]/args.log_interval, steps))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, 1, seq_length)
        if args.permute:
            data = data[:, :, permute]
        # Data should be seq_len, batch, input_size, 
        data = data.permute(2, 0, 1)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
