import torch
from code_tree import Code
from random import random, randint
import numpy as np
from collections import defaultdict,namedtuple
from random import choice
import os


def format_data(ASTS, word2idx):
    data = []
    # filter the unnecessary information in AST, ?can be simplified
    for paragraph in range(len(ASTS)):
        code = ASTS[paragraph]
        for node in code:
            children = []
            if 'children' in node:
                for child in node['children']:
                    children.append(word2idx[code[child]['type']])
                data.append([paragraph, word2idx[node['type']], children])
    return data


def load_ast(ast_file, vocab, directed=False):           # load the saved AST and vocabulary of code blocks                                              # load generated ast
    Data = namedtuple('Data', ['x', 'adjlist'])
    vocab_list = sorted(vocab.keys(), key=lambda d: vocab[d])   # sort the vocabulary list by appearance counts
    word2idx = {}
    for i in range(len(vocab_list)):
        word2idx[vocab_list[i]] = i
    Graphs = {}
    ast = torch.load(ast_file)
    adj = [[0]*len(ast) for i in range(len(ast))]
    x = []
    for i in range(len(ast)):
        node = ast[i]
        word=node['type']
        x.append(word2idx[word] if word in word2idx else word2idx['UKN'])
        if 'children' in node:
            for child in node['children']:
                adj[i][child] = 1
                adj[child][i] = int(directed)

    return Data(x, adj)


def reshape(ast, node_idx=0):
    if 'children' not in ast[node_idx]:
        return ast[node_idx]['type']
    else:
        return {ast[node_idx]['type']: {idx: content for idx, content in
                                        enumerate([reshape(ast, idx) for idx in ast[node_idx]['children']])}}


def _reshape(ast, flag=0):
    if flag:
        _reshape.n = 0
    idx = _reshape.n
    _reshape.n += 1
    if isinstance(ast, str):
        return [{'id':idx, 'type':ast}]
    content = list(ast.keys())[0]
    children = [_reshape(_) for _ in ast[content].values()]
    current = [{'id':idx, 'type':content, 'children': [_[0]['id'] for _ in children]}]
    for _ in children:
        current.extend(_)
    return current


def find_subtree(tree, name, depth=0):
    result = []
    child = list(tree.values())[0]
    max_depth = depth
    for _ in child.values():
        if isinstance(_, dict):
            subs, _depth = find_subtree(_, name, depth+1)
            result += subs
            max_depth = _depth if _depth>max_depth else max_depth
    if name in tree:
        result += [(max_depth-depth+2, tree)]
    return result, max_depth


# randomly change type of nodes to generate positive sample
def graph_aug(graph, vocab, pos=True, prob=0.3):
    res = []
    prob = abs(pos-prob)
    for node in graph:
        res.append(node.copy())
        if random() > prob:
            res[-1]['type'] = choice(vocab)
    return res

