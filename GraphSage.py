#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import GIN_RNN
from input_data import load_ast
from collections import namedtuple, defaultdict
import random

py_ast_dir = 'lc-python'
py_vocab_file = 'lc-python-vocab.pth'
java_ast_dir = 'lc-java'
java_vocab_file = 'lc-java-vocab.pth'
INPUT_DIM = 76  # 输入维度 (节点/样本特征向量维度)
GNN_HIDDEN = 100
RNN_HIDDEN = 100
py_data = load_ast(py_ast_dir, py_vocab_file)
java_data = load_ast(java_ast_dir, java_vocab_file)
data = []
word2idx = {'SOS':0,'EOS':1}
counter = defaultdict(int)
idx2word = ['SOS','EOS']
for name,graph in py_data.items():
    if name not in java_data:
        continue
    _name = name.split('_')[1:]
    data.append([1, _name, graph])
    data.append([0, _name, java_data[name]])
    for word in _name:
        counter[word] += 1
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word.append(word)
counter = sorted(counter.items(),key=lambda x:x[1])

BATCH_SIZE = 20  # 批处理大小
EPOCHS = 2000
NUM_BATCH_PER_EPOCH = 20  # 每个epoch循环的批次数
LEARNING_RATE = 0.001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISP_FREQ = 100

pmodel = GIN_RNN(input_dim=76, gnn_hidden=GNN_HIDDEN, rnn_hidden=RNN_HIDDEN, vocab_size=len(idx2word)).to(DEVICE)
jmodel = GIN_RNN(input_dim=139, gnn_hidden=GNN_HIDDEN, rnn_hidden=RNN_HIDDEN, vocab_size=len(idx2word)).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = optim.Adam(list(jmodel.parameters())+list(pmodel.parameters()),
                       lr=LEARNING_RATE, weight_decay=5e-4)  # Adam优化器\

train_data = data[:len(data)*3//4]
test_data = data[len(data)*3//4:]

def train(pmodel,jmodel):
    for e in range(EPOCHS):
        random.shuffle(train_data)
        total_loss = 0
        tmp_res = []
        for l,name,d in tqdm(train_data):
            model = pmodel if l else jmodel
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)

                seq = torch.LongTensor([word2idx[i] for i in ['SOS']+name]).to(DEVICE) if (e+1)%DISP_FREQ else None
                sentence = model(x, adjlist, seq)
                loss = 0
                _name = name+['EOS']
                pred = []

                if not (e+1)%DISP_FREQ:
                    for i in range(len(sentence)):
                        pred.append(idx2word[np.argmax(sentence[i].detach().cpu().numpy())])
                    tmp_res.append(["{:13}".format('target:')+' '.join(name+['EOS']),"{:13}".format('prediction:')+' '.join(pred)])

                else:
                    for i in range(len(_name)):
                        loss += criterion(sentence[i], torch.LongTensor([word2idx[_name[i]]]).to(DEVICE))

                    optimizer.zero_grad()
                    loss.backward()  # 反向传播计算参数的梯度
                    optimizer.step()  # 使用优化方法进行梯度更新

                total_loss += float(loss)

            except ValueError:
                pass
        print("Epoch {:03d} Loss: {:.4f}".format(e, total_loss / len(train_data)))

        if tmp_res:
            for target,pred in tmp_res:
                print(target)
                print(pred)
        # _test()  # 每一epoch做一次测试


if __name__ == '__main__':
    train(pmodel,jmodel)