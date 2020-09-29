class Code:

    def __init__(self, AST):
        self.ast = AST
        self.nodes = []
        self.depth = 0
        self._AST2nodes(AST)

    def _AST2nodes(self, AST, parent=-1, dist=0):
        if parent == -1:
            current = AST[0]
            self.nodes.append(self.Node(0, current['type'], dist=dist, children=current['children']))
            self._AST2nodes(AST, 0)
        else:
            dist += 1
            if dist > self.depth:
                self.depth = dist
            for n in AST[parent]['children']:
                current = AST[n]
                if 'children' in current:
                    self.nodes.append(self.Node(n, current['type'], dist=dist, parent=[parent], children=current['children']))
                    self._AST2nodes(AST, dist=dist, parent=n)
                else:
                    self.nodes.append(self.Node(n, current['type'], dist=dist, parent=[parent]))


    def copy(self):
        return Code(self.ast)

    def __str__(self):
        # self.show()
        return 'information of this tree'

    class Node:

        word2idx = None
        idx2word = None

        def __init__(self, idx, content, dist=0, parent=None, children=None):
            self.idx = idx
            self.content = content
            self.parent = parent
            self.children = children
            self.distance = dist
            self.voc_idx = self.word2idx[content]
