from RNN import *
from random import shuffle,randint

max_length = 20
num_epochs = 1000
num_batches = 750
batch_size = 100
vocab_size = 15000

def main():
    rnn = RNN(11, 11)
    losses = []
    seq = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    for epoch in range(num_epochs):
        print("=" * 50 + ("  EPOCH %i  " % epoch) + "=" * 50)

        shuffle(seq)
        clip = randint(4,12)
        _seq = seq[:clip]
        _ = np.array(_seq)
        target = np.array([[0]] + list(reversed(_seq)) + [[10]])
        loss, outputs, sentence = rnn.train(Variable(torch.from_numpy(_).long()).cuda(), Variable(torch.from_numpy(target).long().cuda()))
        losses.append(loss)
        print(loss)
        #print(input)
        #print(sentence)

        # rnn.save()
    _ = input()
    while _:
        _ = np.array([[int(i)] for i in _])
        res = rnn.eval(torch.from_numpy(_).long().cuda())
        print([i.data[0,0]for i in res])
        _ = input()

'''def translate():
    data = LanguageLoader(en_path, fr_path, vocab_size, max_length)
    rnn = RNN(data.input_size, data.output_size)

    vecs = data.sentence_to_vec("the president is here <EOS>")

    translation = rnn.eval(vecs)
    print(data.vec_to_sentence(translation))
'''

main()
#translate()