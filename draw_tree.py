"""
@author: lishihang
@software: PyCharm
@file: TreeVis.py
@time: 2018/11/29 22:20
"""
from graphviz import Digraph
import torch
import os

def plot_model(tree, name):
    g = Digraph("G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0")
    g.view()


root = "0"


def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0])
            g.edge(inc, root, str(i))
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, tree[first_label][i])
            g.edge(inc, root, str(i))


def reshape_d(ast, node_idx=0):
    if 'children' not in ast[node_idx]:
        return ast[node_idx]['type']
    else:
        return {ast[node_idx]['type']: {idx: content for idx, content in
                                        enumerate([reshape_d(ast, idx) for idx in ast[node_idx]['children']])}}


def reshape_l(ast, node_idx=0):
    children = [reshape_l(ast, idx) for idx in ast[node_idx][-1]]

    res = {ast[node_idx][1]: {link+str(idx): content for idx,(link, content) in
                               enumerate(children)}}
    return ast[node_idx][-2], res


def clip_reshape(ast, depth=-1, node_idx=0):
    if 'children' not in ast[node_idx] or depth == 0:
        return ast[node_idx]['type']
    else:
        depth -= 1
        return {ast[node_idx]['type']: {idx: content for idx, content in
                                        enumerate([clip_reshape(ast, depth, idx) for idx in ast[node_idx]['children']])}}


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


def draw(ast, name='tmp', clip=0):
    tree = [reshape_d, clip_reshape][clip](ast)
    plot_model(tree, name)


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



if __name__=='__main__':
    '''
    examples
    d1 = {"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}}

    d2 = {'tearRate': {'': 'no lenses', '': {'': {'yes': {
        'prescript': {'myope': 'hard', 'hyper': {'age': {'young': 'hard', 'presbyopic': 'no lenses', 'pre': 'no lenses'}}}},
        'no': {'age': {'young': 'soft', 'presbyopic': {
            'prescript': {'myope': 'no lenses',
                          'hyper': 'soft'}},
                       'pre': {'soft': {}}}}}}}}
    '''
    # py = torch.load('lc-python/'+file_py[0])
    def draw(dir, file, lang):
        graph = torch.load(dir+file)
        graph = reshape_d(graph)
        plot_model(graph, f'tmp/{file[:-4]}')

    java_dir = '../lc-dataset/lc-java/'
    python_dir = '../lc-dataset/lc-python/'
    for p,j in zip(os.listdir(python_dir),os.listdir(java_dir)):
        draw(python_dir, p, 'py')
        draw(java_dir, j, 'java')
        input()