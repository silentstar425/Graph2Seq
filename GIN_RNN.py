#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import GIN_RNN
from input_data import load_ast
import os
import sys
from collections import namedtuple, defaultdict
import random
import argparse

DATA_DIR = 'E:\\python3\\jupyter\\meta learning\\dataset\\'
CODE_DIR = DATA_DIR+'ast\\'
DSCPT_DIR = DATA_DIR+'dscpts_u\\'
SEQ_VOCAB = DATA_DIR+'g2s_seq_vocab.pth'
GRAPH_VOCAB = DATA_DIR+'g2s_graph_vocab.pth'
IGNORE = DATA_DIR+'ignore.pth'
RESULTS = 'results\\'

GNN_HIDDEN = 100
RNN_HIDDEN = 100

EPOCHS = 1000
LEARNING_RATE = 0.001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISP_FREQ = 100
SCALE = 20
SAVE = False

standards = torch.load(IGNORE)
ignore = set()
for s in standards['size'][-2:]:
    ignore.update(s)
for s in standards['length'][:1]:
    ignore.update(s)
for s in standards['length'][-4:]:
    ignore.update(s)
for s in standards['UKN'][1:]:
    ignore.update(s)

graph_vocab = torch.load(GRAPH_VOCAB)
seq_vocab = torch.load(SEQ_VOCAB)
data = [_ for _ in os.listdir(CODE_DIR) if _[:-4] not in ignore]
word2idx = {'SOS':0,'EOS':1}
idx2word = ['SOS','EOS']
for word in seq_vocab.keys():
    idx2word.append(word)
    word2idx[word] = len(word2idx)

INPUT_DIM = len(graph_vocab)

model = GIN_RNN(input_dim=INPUT_DIM, gnn_hidden=GNN_HIDDEN, rnn_hidden=RNN_HIDDEN, vocab_size=len(idx2word)).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  # Adam优化器\

random.shuffle(data)
train_data = data[:len(data)*3//(4*SCALE)]
test_data = data[len(data)*3//(4*SCALE):len(data)*4//(4*SCALE)]


def train(model, DISPLAY = True):
    for e in range(EPOCHS):
        print(f'Epoch: {e}')
        random.shuffle(train_data)
        if (e+1)%DISP_FREQ:
            epoch(model, train_data, DISPLAY=DISPLAY)
        else:
            train_res = epoch(model, train_data, train=False, DISPLAY=DISPLAY)
            test_res = epoch(model, test_data, train=False, DISPLAY=DISPLAY)
            if SAVE:
                if not os.path.isdir(f"results/g2s_results/{e//DISP_FREQ}"):
                    os.mkdir(f"results/g2s_results/{e//DISP_FREQ}")
                with open(f"results/g2s_results/{e//DISP_FREQ}/train.res",'w') as t:
                    for target, pred in train_res:
                        t.write(f"{target}\n{pred}\n\n")
                with open(f"results/g2s_results/{e//DISP_FREQ}/test.res",'w') as t:
                    for target, pred in test_res:
                        t.write(f"target: {target}\npred: {pred}\n\n")
            else:
                print('testing')
                for target, pred in train_res:
                    print(target)
                    print(pred)
                print('===========================================')
                print('evaluation')
                for target, pred in test_res:
                    print(target)
                    print(pred)
                '''
                torch.save(model,'model.pth')
                model = torch.load('model.pth')
                global optimizer
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
                '''


def epoch(model, data, train=True, DISPLAY=True):
    dataset = tqdm(data) if DISPLAY else data
    if train:
        total_loss = 0
        for file in dataset:
            d = load_ast(CODE_DIR + file, graph_vocab)
            with open(DSCPT_DIR + file[:-7] + '.des', 'r') as des:
                y = des.read().split(' ')
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)

                seq = torch.LongTensor([word2idx[i] if i in word2idx else word2idx['UKN'] \
                                        for i in ['SOS'] + y]).to(DEVICE)
                sentence = model(x, adjlist, seq)
                loss = 0

                for pred, target in zip(sentence, y + ['EOS']):
                    loss += criterion(pred, torch.LongTensor([word2idx[target] \
                                                                  if target in word2idx else word2idx['UKN']]).to(
                        DEVICE))
                loss /= len(y) + 1
                total_loss += float(loss)
                optimizer.zero_grad()
                loss.backward()  # 反向传播计算参数的梯度
                optimizer.step()

            except ValueError:
                pass

        print(f"Loss: {total_loss / len(train_data)}")

    else:
        tmp_res = []
        for file in dataset:
            d = load_ast(CODE_DIR + file, graph_vocab)
            with open(DSCPT_DIR + file[:-7] + '.des', 'r') as des:
                y = des.read().split(' ')
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
                sentence = model(x, adjlist, None)
                pred = []

                for i in range(len(sentence)):
                    pred.append(idx2word[np.argmax(sentence[i].detach().cpu().numpy())])
                tmp_res.append([f"{'target:':{15 if SAVE else 13}}" + ' '.join([_ if _ in word2idx else 'UKN' \
                                                                      for _ in y + ['EOS']]),
                                f"{'prediction:':13}" + ' '.join(pred)])

            except ValueError:
                pass

        '''
        for target, pred in tmp_res:
            print(target)
            print(pred)
        '''
        return tmp_res


def show_hyper():
    print(f'#Epochs = {EPOCHS}')
    print(f'Learning rate = {LEARNING_RATE}')
    print(f'Display frequence = {DISP_FREQ}')
    print(f'Data set scale = 1/{SCALE}')
    print(f'Save results = {SAVE}')


def infer(data, save=None):
    for file in tqdm(data):
        d = load_ast(CODE_DIR + file, graph_vocab)
        x = torch.LongTensor(d.x).to(DEVICE)
        adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
        if not save:
            print(' '.join(f'{_:7.3f}' for _ in model.inferring(x, adjlist)))
        else:
            with open(save+file[:-7]+".emb",'w') as emb:
                emb.write(' '.join(f'{_:}' for _ in model.inferring(x, adjlist)))


if __name__ == '__main__':
    train(model)
    infer(data,'results/emb/')
else:
    from tqdm.notebook import tqdm