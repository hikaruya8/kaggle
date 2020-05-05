from dataloader import get_tweets_and_sentiment_label_loaders
from model import TransformerClassification
import torch.nn as nn

# データの読み込み
train_dl, val_dl, test_dl, TEXT1, TEXT2 = get_tweets_and_sentiment_label_loaders(max_length=256, batch_size=64)
# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}

# compose model
net = TransformerClassification(text_embedding_vectors=TEXT1.vocab.vectors, d_model=300, max_seq_len=256, output_dim=3)

# ネットワークの初期化を定義
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # init Linear Layer
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# train mode
# TransformerBlockモジュールを初期化実行
net.net3_1.apply(weights_init)
net.net3_2.apply(weights_init)

print("ネットワーク設定完了")


# if __name__ == '__main__':
#     train_svm()
