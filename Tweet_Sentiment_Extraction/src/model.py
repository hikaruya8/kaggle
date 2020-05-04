import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()
        #全結合層で特徴量を変換
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # calc Atention, 各値を足し算するだけだと大きくなってしまいすぎるため、root(d_k)でわって調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化
        normalized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueと計算
        output = torch.matmul(normalized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)

        return output, normalized_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = Attention(d_model)

        # Attention後全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(x_normalized, x_normalized, x_normalized, mask)
        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized2))

        return output, normalized_weights

'''test'''
# model 定義
net1 = Embedder(TEXT1.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)
net3 = TransformerBlock(d_model=300)

# mask 作成
batch = next(iter(train_dl))
x = batch.Text1[0]
input_pad = 1 # 単語のIDで'<pad>: 1 のため
input_mask = (x != input_pad)
print(input_mask[0])

# 入出力
x1 = net1(x)
x2 = net2(x1)
x3, normalized_weights = net3(x2, input_mask) # self-attentionで特徴量変換

print('入力テンソルサイズ:{}:'.format(x2.shape))
print('出力テンソルサイズ:{}:'.format(x3.shape))
print('Attentionのサイズ:{}'.format(normalized_weights.shape))



