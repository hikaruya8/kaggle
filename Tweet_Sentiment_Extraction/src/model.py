import math
import torch
import torch.nn as nn
from dataloader import get_tweets_and_sentiment_label_loaders

train_dl, val_dl, test_dl, TEXT1, TEXT2 = get_tweets_and_sentiment_label_loaders()

class Embedder(nn.Module):
    # idで示される単語をベクトルへ変換
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True) # freeze=True バックプロパゲージョンが無効になる

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec
'''
# test
# prepare minibatch
batch = next(iter(train_dl))
# compose model
net1 = Embedder(TEXT1.vocab.vectors)

# input & output
x = batch.Text1[0]
x1 = net1(x) # words to vectors
print("input tensor size:{}".format(x.shape))
print("output tensor size:{}".format(x1.shape))
'''

class PositionalEncoder(nn.Module):
    # 入力された単語の位置情報を示すベクトル情報を付与
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()
        self.d_model = d_model # 単語ベクトルの次元数

        # 単語の順番(pos)と埋め込みベクトルの次元の位置(i)によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # use GPU if it is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        # positinoal encoding 計算
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
        # table pe の先頭にミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)
        # 勾配計算をしななくする

    def forward(self, x):
        # 入力xとPositinoal Encodingの足し算
        # xがpeよりも小さいため大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret

'''
# test
# prepare minibatch
batch = next(iter(train_dl))
# compose model
net1 = Embedder(TEXT1.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)

# input & output
x = batch.Text1[0]
x1 = net1(x) # words to vectors
x2 = net2(x1)
print("input tensor size:{}".format(x1.shape))
print("output tensor size:{}".format(x2.shape))
'''

