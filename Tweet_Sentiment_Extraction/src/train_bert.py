from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import time
import pdb
from dataloader_with_bert import get_tweets_and_sentiment_label_loaders
from bert_model import BertForTweetSentimentClassification
import logging
logging.basicConfig(level=logging.INFO)

train_dl, val_dl, test_dl, TEXT1, TEXT2, TEST_TEXT, dataloaders_dict = get_tweets_and_sentiment_label_loaders()

pre_trained_bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BertForTweetSentimentClassification(pre_trained_bert_model)


'''BERT のファインチューニングのための設定'''

# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

# 1. 全部を、勾配計算Falseにする
for name, param in model.named_parameters():
    param.requires_grad = False

# 2. 最後のBertLayerモジュールを勾配計算ありに変更
for name, param in model.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

# 3. 識別器を勾配計算ありに変更
for name, param in model.cls.named_parameters():
    param.requires_grad = True

# 最適化手法の設定

# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': model.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': model.cls.parameters(), 'lr': 5e-5}
], betas=(0.9, 0.999))

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算


# モデルを学習させる関数を作成


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    model.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # モデルを訓練モードに
            else:
                model.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            iteration = 1

            # 開始時刻を保存
            t_epoch_start = time.time()
            t_iter_start = time.time()

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text1[0].to(device)  # 文章
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # pdb.set_trace()
                    # BertForTweetSentimentClassificationに入力
                    outputs = model(inputs, token_type_ids=None)

                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(), duration, acc))
                            t_iter_start = time.time()

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))
            t_epoch_start = time.time()

    return model


if __name__ == '__main__':
    num_epochs = 2
    model_trained = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
