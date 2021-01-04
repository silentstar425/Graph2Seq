from GIN_RNN import *

FILTER_EPOCHS = 500


def filter_train(model, data, optimizer, EPOCHS):
    for e in range(EPOCHS):
        print(f'Epoch: {e}')
        random.shuffle(train_data)
        epoch(model, data, optimizer, DISPLAY=DISPLAY)


def filter(model, data):
    score = {}
    for file in data:
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
            score[file] = float(loss)
        except ValueError:
            pass
    avrg = sum(score.values())/len(score)
    delta = sum((x-avrg)**2/len(score) for x in score.values())**(0.5)
    for file in score:
        score[file] = (score[file] - avrg)/delta
    return score


if __name__=="__main__":
    score = {file:0 for file in data}
    for i in range(100):
        model = GIN_RNN(input_dim=INPUT_DIM, gnn_hidden=GNN_HIDDEN, rnn_hidden=RNN_HIDDEN, vocab_size=len(idx2word)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  # Adam优化器
        random.shuffle(data)
        filter_data = data[:len(data)//2]
        filter_train(model, filter_data, optimizer, 10)
        for file, s in filter(model, filter_data).items():
            score[file] -= s
        torch.save(score, f'tmp/{i}.pth')