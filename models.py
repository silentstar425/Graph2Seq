import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm
from collections import namedtuple, defaultdict
from rnn_encoder_decoder.modules.Decoder import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # 邻居聚合
class Aggregator(nn.Module):
    def __init__(self, input_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(2*input_dim, input_dim))
        self.activation = F.relu
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()  # 自定义参数初始化

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, node_features, adjlist):
        if self.aggr_method == "mean":
            aggr_neighbor = torch.matmul(adjlist.float(), node_features)/(adjlist.sum(1)+0.0001).unsqueeze(1)
        elif self.aggr_method == "sum":
            aggr_neighbor = torch.matmul(adjlist.float(), node_features)
        else:
            raise ValueError("Unknown aggr type, expected sum or mean, but got {}"
                             .format(self.aggr_method))
        hidden = torch.matmul(torch.cat((node_features, aggr_neighbor),1), self.weight)
        if self.use_bias:
            hidden += self.bias

        return self.activation(hidden)

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


# # Readout函数
class ReadOut(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReadOut, self).__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim),
                                 nn.ReLU(),
                                 nn.Linear(input_dim, input_dim))
        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim),
                                  nn.ReLU(),
                                  nn.Linear(input_dim, input_dim),
                                  nn.Sigmoid())
        self.aggr = nn.Sequential(nn.Linear(input_dim, input_dim),
                                  nn.ReLU(),
                                  nn.Linear(input_dim, output_dim))

    def forward(self, node_features, _):
        x = self.mlp(node_features)
        gate = self.gate(node_features)
        x = (gate*x).mean(dim=0)
        return self.aggr(x), node_features


# # GraphSage网络
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 layers):
        super(GIN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.node_features = nn.Embedding(self.input_dim,hidden_dim)
        self.num_layers = layers
        self.aggr = nn.ModuleList()
        for index in range(layers):
            self.aggr.append(Aggregator(hidden_dim))
        self.aggr.append(ReadOut(hidden_dim,output_dim))

    def forward(self, node_list, adjlist):
        node_features = self.node_features(node_list)
        for aggr in self.aggr:
            node_features = aggr(node_features, adjlist)
        graph_emb, node_emb = node_features
        return graph_emb, node_emb

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


class AttentionNet(nn.Module):

    def __init__(self, rnn_hidden, gnn_hidden):
        super(AttentionNet, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(rnn_hidden, gnn_hidden))
        self.linear = nn.Linear(rnn_hidden+gnn_hidden, rnn_hidden)
        self.activation = F.relu
        self.reset_parameters()  # 自定义参数初始化

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, encoder_output, hidden):
        score = hidden.matmul(self.weight).matmul(encoder_output.t())
        soft = F.softmax(score,dim=0)
        ctx = soft.matmul(encoder_output)
        next_hidden = self.activation(self.linear(torch.cat((hidden, ctx))))
        return next_hidden


class GIN_RNN(nn.Module):

    def __init__(self, input_dim, gnn_hidden, rnn_hidden, vocab_size, layers=5):
        super(GIN_RNN, self).__init__()
        self.GIN = GIN(input_dim, gnn_hidden, rnn_hidden, layers)
        self.Decoder = Decoder(vocab_size, hidden_size=rnn_hidden)
        self.attention = AttentionNet(rnn_hidden, gnn_hidden)

    def forward(self, node_list, adjlist, seq=None):
        hidden, node_emb = self.GIN(node_list, adjlist)
        sentence = []
        if seq is not None:
            for word in seq:
                hidden = self.attention(node_emb, hidden)
                _, pred, hidden = self.Decoder(word, hidden)
                hidden = hidden.squeeze()
                sentence.append(pred)
        else:
            limit = 50
            word = torch.LongTensor([0]).to(DEVICE)
            while limit and word != 1:
                hidden = self.attention(node_emb, hidden)
                _, pred, hidden = self.Decoder(word, hidden)
                hidden = hidden.squeeze()
                word = torch.LongTensor([np.argmax(pred.detach().cpu().numpy())]).to(DEVICE)
                sentence.append(pred)
                limit -= 1
        return sentence

    def inferring(self, node_list, adjlist):
        hidden, node_emb = self.GIN(node_list, adjlist)
        word = torch.LongTensor([0]).to(DEVICE)
        res = hidden
        for i in range(5):
            hidden = self.attention(node_emb, hidden)
            _, pred, hidden = self.Decoder(word, hidden)
            hidden = hidden.squeeze()
            word = torch.LongTensor([np.argmax(pred.detach().cpu().numpy())]).to(DEVICE)
            res = torch.cat((res, hidden), 0)

        return res.cpu().detach().tolist()