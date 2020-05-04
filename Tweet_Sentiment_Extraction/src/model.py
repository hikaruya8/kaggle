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


