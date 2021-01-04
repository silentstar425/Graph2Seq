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
import shutil
import sys
from collections import namedtuple, defaultdict
from torch.optim.lr_scheduler import StepLR
import random
import argparse

DATA_DIR = 'E:/python3/U-tokyo/lc-dataset-merged/'
P_CODE_DIR = DATA_DIR+'lc-python/'
J_CODE_DIR = DATA_DIR+'lc-java/'
DSCPT_DIR = DATA_DIR+'dscpts_u/'
SEQ_VOCAB = DATA_DIR+'question-vocab.pth'
P_GRAPH_VOCAB = DATA_DIR+'lc-python-vocab.pth'
J_GRAPH_VOCAB = DATA_DIR+'lc-java-vocab.pth'
UNION_VOCAB = DATA_DIR+'union-vocab.pth'
IGNORE = DATA_DIR+'ignore.pth'
SAVE_DIR = "results/g2s_results/"
EMB_DIR = 'results/emb/'

p_w = 1
j_w = 1

NOTE = f'unbalanced training {p_w}/{j_w},without attention'

GNN_HIDDEN = 400
P_GNN_HIDDEN = 400
J_GNN_HIDDEN = 400
P_AGGR_DEPTH = 10
J_AGGR_DEPTH = 10

RNN_HIDDEN = 500

EPOCHS = 2000
LEARNING_RATE = 0.0001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISP_FREQ = 100
SAVE_FREQ = 500
SCALE = 1
DISPLAY = True
SAVE = True


'''
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
'''
seq_vocab = torch.load(SEQ_VOCAB)
word2idx = {'SOS':0,'EOS':1}
idx2word = ['SOS','EOS']
for word in seq_vocab.keys():
    idx2word.append(word)
    word2idx[word] = len(word2idx)
data = [_[:-7] for _ in os.listdir(P_CODE_DIR)]

#p_graph_vocab = torch.load(P_GRAPH_VOCAB)
#j_graph_vocab = torch.load(J_GRAPH_VOCAB)

p_graph_vocab = j_graph_vocab = torch.load(UNION_VOCAB)
P_INPUT_DIM = len(p_graph_vocab)
J_INPUT_DIM = len(j_graph_vocab)
INPUT_DIM = [P_INPUT_DIM , J_INPUT_DIM]
GNN_HIDDEN = [P_GNN_HIDDEN , J_GNN_HIDDEN]


model = GIN_RNN(input_dim=INPUT_DIM,
                gnn_hidden=GNN_HIDDEN,
                rnn_hidden=RNN_HIDDEN,
                vocab_size=len(idx2word),
                layers=[P_AGGR_DEPTH,J_AGGR_DEPTH],
                siamese=True).to(DEVICE)


criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  # Adam优化器\
scheduler = StepLR(optimizer, 50, gamma=0.9, last_epoch=-1)


random.shuffle(data)
train_data = data[:len(data)*3//(4*SCALE)]
test_data = data[len(data)*3//(4*SCALE):len(data)*4//(4*SCALE)]


def train(model, train_data, test_data, optimizer, DISPLAY = True):
    if SAVE:
        if os.path.isdir(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.mkdir(SAVE_DIR)
    for e in range(EPOCHS):
        print(f'Epoch: {e}')
        random.shuffle(train_data)
        if not (e+1)%SAVE_FREQ:
            torch.save(model,SAVE_DIR+f'model_{(e+1)//SAVE_FREQ}.pth')
        if (e+1)%DISP_FREQ:
            epoch(model, train_data, optimizer, DISPLAY=DISPLAY)
            scheduler.step()
        else:
            train_res = epoch(model, train_data, DISPLAY=DISPLAY)
            test_res = epoch(model, test_data, DISPLAY=DISPLAY)

            '''
            torch.save(model, 'model.pth')
            _model = torch.load('model.pth')

            _train_res = epoch(_model, train_data, train=False, DISPLAY=DISPLAY)
            _test_res = epoch(_model, test_data, train=False, DISPLAY=DISPLAY)
            '''

            if SAVE:
                os.mkdir(SAVE_DIR+f"{e//DISP_FREQ}")
                with open(SAVE_DIR+f"{e//DISP_FREQ}/train.res",'w') as t:
                    for target, pred in train_res:
                        t.write(f"{target}\n{pred}\n\n")
                with open(SAVE_DIR+f"{e//DISP_FREQ}/test.res",'w') as t:
                    for target, pred in test_res:
                        t.write(f"{target}\n{pred}\n\n")
                '''
                with open(SAVE_DIR+f"{e//DISP_FREQ}/_train.res",'w') as t:
                    for target, pred in _train_res:
                        t.write(f"{target}\n{pred}\n\n")
                with open(SAVE_DIR+f"{e//DISP_FREQ}/_test.res",'w') as t:
                    for target, pred in _test_res:
                        t.write(f"{target}\npred: {pred}\n\n")
                '''
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


def epoch(model, data, optimizer=None, DISPLAY=True):
    dataset = tqdm(data) if DISPLAY else data
    if optimizer:
        p_total_loss = 0
        j_total_loss = 0
        for file in dataset:
            d = load_ast(P_CODE_DIR+file+'.py.pth', p_graph_vocab)
            '''
            with open(DSCPT_DIR + file[:-7] + '.des', 'r') as des:
                y = des.read().split(' ')
            '''
            y = file.split('_')[1:]
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)

                seq = torch.LongTensor([word2idx[i] if i in word2idx else word2idx['UKN'] \
                                        for i in ['SOS'] + y]).to(DEVICE)
                sentence = model(x, adjlist, 0, seq)
                loss = 0

                for pred, target in zip(sentence, y + ['EOS']):
                    loss += criterion(pred, torch.LongTensor([word2idx[target] \
                                                                  if target in word2idx else word2idx['UKN']]).to(
                        DEVICE))
                loss /= len(y) + 1
                p_total_loss += float(loss)
                loss *= p_w
                optimizer.zero_grad()
                loss.backward()  # 反向传播计算参数的梯度
                optimizer.step()

            except ValueError:
                pass

            d = load_ast(J_CODE_DIR + file + '.java.pth', j_graph_vocab)
            '''
            with open(DSCPT_DIR + file[:-7] + '.des', 'r') as des:
                y = des.read().split(' ')
            '''
            y = file.split('_')[1:]
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)

                seq = torch.LongTensor([word2idx[i] if i in word2idx else word2idx['UKN'] \
                                        for i in ['SOS'] + y]).to(DEVICE)
                sentence = model(x, adjlist, 1, seq)
                _loss = 0

                for pred, target in zip(sentence, y + ['EOS']):
                    _loss += criterion(pred, torch.LongTensor([word2idx[target] \
                                                                  if target in word2idx else word2idx['UKN']]).to(
                        DEVICE))
                _loss /= len(y) + 1
                j_total_loss += float(_loss)
                _loss *= j_w
                loss += _loss
                optimizer.zero_grad()
                _loss.backward()  # 反向传播计算参数的梯度
                optimizer.step()

            except ValueError:
                pass
        print(f"p_Loss: {p_total_loss / len(data)}\nj_Loss: {j_total_loss / len(data)}")

    else:
        tmp_res = []
        
        for file in dataset:
            d = load_ast(P_CODE_DIR + file + '.py.pth', p_graph_vocab)
            ''' 
            with open(DSCPT_DIR + file[:-7] + '.des', 'r') as des:
                y = des.read().split(' ')
            '''
            y = file.split('_')[1:]
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
                sentence = model(x, adjlist, 0, None)
                emb = model.inferring(x,adjlist,0)
                pred = []

                for i in range(len(sentence)):
                    pred.append(idx2word[np.argmax(sentence[i].detach().cpu().numpy())])
                tmp_res.append([f"{'target:':{15 if SAVE else 13}}" + ' '.join([_ if _ in word2idx else 'UKN' \
                                                                      for _ in y + ['EOS']]),
                                f"{'prediction:':13}" + ' '.join(pred)])

            except ValueError:
                pass

        tmp_res.append(['========java========','========code========'])
        
        for file in dataset:
            d = load_ast(J_CODE_DIR + file + '.java.pth', j_graph_vocab)
            '''
            with open(DSCPT_DIR + file[:-7] + '.des', 'r') as des:
                y = des.read().split(' ')
            '''
            y = file.split('_')[1:]
            try:
                x = torch.LongTensor(d.x).to(DEVICE)
                adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
                sentence = model(x, adjlist, 1, None)
                emb = model.inferring(x,adjlist,1)
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

'''
def infer(data, save=None):
    for f in os.listdir(save):
        os.remove(save+f)
    for file in tqdm(data):
        d = load_ast(CODE_DIR + file, graph_vocab)
        x = torch.LongTensor(d.x).to(DEVICE)
        adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
        if not save:
            print(' '.join(f'{_:7.3f}' for _ in model.inferring(x, adjlist)))
        else:
            with open(save+file[:-7]+".emb",'w') as emb:
                emb.write(' '.join(f'{_:}' for _ in model.inferring(x, adjlist)))
'''

if __name__ == '__main__':
    print('\n\n\n', f'lr:{LEARNING_RATE},G_hidden:{P_GNN_HIDDEN}|{J_GNN_HIDDEN},R_hidden:{RNN_HIDDEN},k:{P_AGGR_DEPTH}|{J_AGGR_DEPTH}')
    train(model, train_data, test_data, optimizer, DISPLAY)
    # infer(data,EMB_DIR)
    print('\n\n\n',f'lr:{LEARNING_RATE},G_hidden:{P_GNN_HIDDEN}|{J_GNN_HIDDEN},R_hidden:{RNN_HIDDEN},k:{P_AGGR_DEPTH}|{J_AGGR_DEPTH},save_dir={SAVE_DIR}')
    with open(SAVE_DIR+'params','w') as params:
        params.write(f'lr:{LEARNING_RATE},G_hidden:{P_GNN_HIDDEN}|{J_GNN_HIDDEN},R_hidden:{RNN_HIDDEN},k:{P_AGGR_DEPTH}|{J_AGGR_DEPTH}\n'+NOTE)
else:
    from tqdm.notebook import tqdm
