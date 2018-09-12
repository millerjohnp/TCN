import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=1,
                 tie_weights=False, dropout=0.0):
        super(RNNModel, self).__init__()
        self.encoder = nn.Linear(ninp, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)

        if dropout > 0.0:
            raise

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
 
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(10.)

    def forward(self, input):
        output, _ = self.rnn(input)
        decoded = self.decoder(output)
        return torch.sigmoid(decoded)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def stabilize(self):
        if self.rnn_type == "LSTM":
            #            wvecs = self.encoder.weight.data
            #            ones = torch.ones_like(wvecs)
            #            trimmed_wvecs = torch.min(torch.max(wvecs, -ones), ones)
            #            model.encoder.weight.data.set_(trimmed_wvecs)
            #
            # One set of weights satisfying stability requirement
            recur_weights = self.rnn.weight_hh_l0.data
            wi, wf, wz, wo = recur_weights.chunk(4, 0)
            
            trimmed_wi =  wi * 0.395  / torch.sum(torch.abs(wi), 0)
            trimmed_wf =  wf * 0.155  / torch.sum(torch.abs(wf), 0)
            trimmed_wz =  wz * 0.099  / torch.sum(torch.abs(wz), 0)
            trimmed_wo =  wo * 0.395 / torch.sum(torch.abs(wo), 0)
            new_recur_weights = torch.cat([trimmed_wi, trimmed_wf, trimmed_wz, trimmed_wo], 0)
            self.rnn.weight_hh_l0.data.set_(new_recur_weights)

            # Also trim the input to hidden weight for the forget gate
            ih_weights = self.rnn.weight_ih_l0.data
            ui, uf, uz, uo = ih_weights.chunk(4, 0)
            trimmed_uf =  uf * 0.25  / torch.sum(torch.abs(uf), 0)
            new_ih_weights = torch.cat([ui, trimmed_uf, uz, uo], 0)
            self.rnn.weight_ih_l0.data.set_(new_ih_weights)

            self.rnn.flatten_parameters()

        elif self.rnn_type in ["RNN_TANH", "RNN_RELU"]:
            # Sometimes the projection fails apparently.
            # This is slow-- a better strategy would be to use the power method!
            # (e.g. power on A^T A, and then normalize by sqrt(\lambda).
            try:
                U, s, V = torch.svd(self.rnn.weight_hh_l0.data)
                s = torch.min(s, torch.ones_like(s))
                projected =  torch.mm(torch.mm(U, torch.diag(s)), V.t()) 
                self.rnn.weight_hh_l0.data.set_(projected)
            except:
                print("Projection failed!")
                raise
