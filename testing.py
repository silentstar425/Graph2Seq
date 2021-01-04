from GIN_RNN import *

model = torch.load('model.pth')

res = epoch(model, data, train=False, DISPLAY=DISPLAY)

for target, pred in res:
    print(target)
    print(pred)

j_res = {}
p_res = {}

for file in tqdm(data):
    d = load_ast(J_CODE_DIR + file + '.java.pth', j_graph_vocab)
    x = torch.LongTensor(d.x).to(DEVICE)
    adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
    j_res[file] = (np.array(model.inferring(x, adjlist, 1)))

    d = load_ast(P_CODE_DIR + file + '.py.pth', p_graph_vocab)
    x = torch.LongTensor(d.x).to(DEVICE)
    adjlist = torch.LongTensor(d.adjlist).to(DEVICE)
    p_res[file] = np.array(model.inferring(x, adjlist, 0))