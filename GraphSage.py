#!/usr/bin/env python
# coding: utf-8
# # 导包
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from input_data import load_ast
from tqdm import tqdm
from time import sleep
from random import choice
from collections import namedtuple, defaultdict
from rnn_encoder_decoder.modules.Decoder import Decoder


# # 邻居采样
def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表，邻接矩阵
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样 
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()  # 拉伸为1维


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射 /邻接矩阵
    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]  # 首先包含源节点
    for k, hopk_num in enumerate(sample_nums):  # 先对源节点进行1阶采样 在与源节点距离为1的节点中采样hopk_num个节点； 再对源节点进行2阶采样，即对源节点的所有1阶邻居进行1阶采样
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)  # 追加源节点的1阶邻居 和 2阶邻居(2层网络，代表采样到2阶)
    return sampling_result


# # 邻居聚合
class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self._weight = nn.Parameter(torch.Tensor(139, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()  # 自定义参数初始化

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)  # 先聚合再做线性变换
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


# # GraphSage层
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation

        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        self.reset_parameters()  # 自定义参数初始化方式

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        # 得到邻居节点的聚合特征(经过线性变换)
        neighbor_hidden = self.aggregator(neighbor_node_features)
        # 对中心节点的特征作线性变换
        self_hidden = torch.matmul(src_node_features, self.weight)

        # 对中心节点的特征和邻居节点的聚合特征进行求和或拼接
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        # 通过激活函数
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


# # Readout函数
class ReadOut(nn.Module):
    def __init__(self, input_dim):
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
                                  nn.Linear(input_dim, 1000))

    def forward(self, node_features):
        x = self.mlp(node_features)
        gate = self.gate(node_features)
        x = (gate*x).mean(dim=0)
        return self.aggr(x)


# # GraphSage网络
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim  # 1433
        self.hidden_dim = hidden_dim  # [128,7]
        self.num_neighbors_list = num_neighbors_list  # [10,10] 两层 1阶节点采样10个 2阶节点采样10个
        self.num_layers = len(num_neighbors_list)  # 2层
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
        self.gcn.append(ReadOut(hidden_dim[-1]))

    def forward(self, node_features_list):
        hidden = node_features_list  # [[源节点对应的特征],[1阶邻居对应的特征],[2阶邻居对应的特征]]
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]  # 源节点对应的特征  (batch_size,feature_size=1433)
                src_node_num = len(src_node_features)  # batch_size
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                # (batch_size*10,feature_size) -> (batch_size,10,feature_size)
                h = gcn(src_node_features, neighbor_node_features)  # (batch_size,hidden_size=128)
                next_hidden.append(h)
            hidden = next_hidden
        return self.gcn[-1](hidden[0])  # (batch_size,hidden_size=7)

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


# # 仿造数据
'''
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])
x = np.zeros((2000,100))
y = np.zeros(2000)
adj = defaultdict(lambda:np.ndarray(0,dtype=int))
for i in range(2000):
    index = np.random.choice(range(100),25)
    x[i,index] = 1
    y[i] = np.random.choice(range(7))
    adj[i] = np.append(adj[i], np.random.choice(range(i+1,2000),10) if i < 1800 else np.ndarray(0,dtype=int))
    for node in adj[i]:
        adj[node] = np.append(adj[node],i)
train_index = np.arange(1200)
val_index = np.arange(1200, 1600)
test_index = np.arange(1600, 2000)
train_mask = np.zeros(2000, dtype=np.bool)
val_mask = np.zeros(2000, dtype=np.bool)
test_mask = np.zeros(2000, dtype=np.bool)
train_mask[train_index] = True
val_mask[val_index] = True
test_mask[test_index] = True
data = Data(x,y,adj,train_mask,val_mask,test_mask)
'''
# # 网络测试
"""
基于Cora的GraphSage示例
"""

py_ast_dir = 'lc-python'
py_vocab_file = 'lc-python-vocab.pth'
java_ast_dir = 'lc-java'
java_vocab_file = 'lc-java-vocab.pth'
INPUT_DIM = 76  # 输入维度 (节点/样本特征向量维度)
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 100]  # 隐藏单元节点数  两层
NUM_NEIGHBORS_LIST = [10, 10]  # 每阶/每层采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
py_data = load_ast(py_ast_dir, py_vocab_file)
java_data = load_ast(java_ast_dir, java_vocab_file)
data = []
vocab = {'SOS':0,'EOS':1}
counter = defaultdict(int)
idx = ['SOS','EOS']
for name,graph in py_data.items():
    if name not in java_data:
        continue
    _name = name.split('_')[1:]
    data.append([1, _name, graph])
    data.append([0, _name, java_data[name]])
    for word in _name:
        counter[word] += 1
        if word not in vocab:
            vocab[word] = len(vocab)
            idx.append(word)
counter = sorted(counter.items(),key=lambda x:x[1])
'''for name in java_data.keys():
    data.append(['java_'+str(name), java_data[name], java_data_p[name], java_data_n[name]])
'''

BATCH_SIZE = 20  # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20  # 每个epoch循环的批次数
LEARNING_RATE = 0.001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pmodel = GraphSage(input_dim=76, hidden_dim=HIDDEN_DIM,
                     num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
jmodel = GraphSage(input_dim=139, hidden_dim=HIDDEN_DIM,
                     num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
dcr = Decoder(len(vocab)).to(DEVICE)
# print(model)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = optim.Adam(list(jmodel.parameters())+list(pmodel.parameters())+list(dcr.parameters()),
                       lr=LEARNING_RATE, weight_decay=5e-4)  # Adam优化器\
norm = nn.BatchNorm1d(1000).to(DEVICE)


import random

#random.shuffle(data)

train_data = data[:len(data)*3//4]
test_data = data[len(data)*3//4:]

def train(pmodel,jmodel):
    pmodel.train()  # 训练模式
    jmodel.train()
    for e in range(EPOCHS):
        random.shuffle(train_data)
        total_loss = 0
        tmp_res = []
        for l,name,d in tqdm(train_data):
            model = pmodel if l else jmodel
            try:
                # data = CoraData().data #获取预处理数据
                x = d.x / d.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
                batch_src_index = np.arange(len(x))  # 随机选择BATCH_SIZE个训练节点 作为源节点
                # 对源节点进行0-K阶(k等于网络层数 k=2)采样
                batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST,
                                                          d.adjacency_dict)  # [[源节点索引列表],[源节点的1阶邻居索引]=10,[源节点的2阶邻居索引]=10]
                batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in
                                    batch_sampling_result]  # 获取采样的节点对应的特征向量
                x = model(batch_sampling_x)    # 获取模型的输出 (BATCH_SIZE,hidden_size[-1]=7)
                # loss = criterion(batch_train_logits, batch_src_label)

                x = x.unsqueeze(0).unsqueeze(0)
                input = torch.LongTensor([0]).to(DEVICE)
                loss = 0
                pred = []
                for word in name + ['EOS']:
                    word = torch.LongTensor([vocab[word]]).to(DEVICE)
                    _, softmax, x = dcr(input, x)
                    loss += criterion(softmax, word)
                    _word = np.argmax(softmax.data.cpu().numpy())
                    input = word
                    pred.append(idx[_word])

                if not (e+1)%10:
                    tmp_res.append(["{:13}".format('target:')+' '.join(name+['EOS']),"{:13}".format('prediction:')+' '.join(pred)])

                optimizer.zero_grad()
                loss.backward()  # 反向传播计算参数的梯度
                optimizer.step()  # 使用优化方法进行梯度更新

                total_loss += float(loss)

            except ValueError:
                pass
        print("Epoch {:03d} Loss: {:.4f}".format(e, total_loss / len(train_data)))

        if not e%10:
            for target,pred in tmp_res:
                print(target)
                print(pred)
        # _test()  # 每一epoch做一次测试

    for l, name, d in data:
        model = pmodel if l else jmodel
        x = d.x / d.x.sum(1, keepdims=True)
        batch_src_index = np.arange(len(x))
        batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST,
                                                  d.adjacency_dict)
        batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in
                            batch_sampling_result]
        x = model(batch_sampling_x)
        print(' '.join(name)+['j','p'][l], x.detach().cpu().tolist())


def _test():
    model.eval()  # 测试模型
    with torch.no_grad():
        for d,d_p,d_n in train_data:
            # data = CoraData().data #获取预处理数据
            x = d.x / d.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
            batch_src_index = np.arange(len(x))  # 随机选择BATCH_SIZE个训练节点 作为源节点
            # 对源节点进行0-K阶(k等于网络层数 k=2)采样
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST,
                                                      d.adjacency_dict)  # [[源节点索引列表],[源节点的1阶邻居索引]=10,[源节点的2阶邻居索引]=10]
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in
                                batch_sampling_result]  # 获取采样的节点对应的特征向量
            x = model(batch_sampling_x)    # 获取模型的输出 (BATCH_SIZE,hidden_size[-1]=7)
            # loss = criterion(batch_train_logits, batch_src_label)

            x_p = d_p.x / d_n.x.sum(1, keepdims=True)
            batch_src_index = np.arange(len(x_p))
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST,
                                                      d.adjacency_dict)
            batch_sampling_x = [torch.from_numpy(x_p[idx]).float().to(DEVICE) for idx in
                                batch_sampling_result]
            x_p = model(batch_sampling_x)

            x_n = d_n.x / d_n.x.sum(1, keepdims=True)
            batch_src_index = np.arange(len(x_n))
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST,
                                                      d.adjacency_dict)
            batch_sampling_x = [torch.from_numpy(x_n[idx]).float().to(DEVICE) for idx in
                                batch_sampling_result]
            x_n = model(batch_sampling_x)

            x = x.unsqueeze(0)
            x_p = x_p.unsqueeze(0)
            x_n = x_n.unsqueeze(0)
            loss = criterion(x, x_p, x_n).to(DEVICE)
            print(loss)
    # model.train()


if __name__ == '__main__':
    train(pmodel,jmodel)