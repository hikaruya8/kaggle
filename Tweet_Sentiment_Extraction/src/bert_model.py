from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
import pdb
from dataloader_with_bert import get_tweets_and_sentiment_label_loaders

train_dl, val_dl, test_dl, TEXT1, TEXT2, TEXT_TEXT = get_tweets_and_sentiment_label_loaders()

bert_model = BertModel.from_pretrained('bert-base-uncased')

class BertForTweetSentimentClassification(nn.Module):
    # BERT Model にkaggle Tweet Sentiment Extraction の neutral, positive, negartiveの判定を加えたモデル
    def __init__(self, bert_model):
        super(BertForTweetSentimentClassification, self).__init__()
        # BERT module
        self.bert = bert_model # BERT MODEL

        # headにポジネガを予測を追加
        # inputはBERTの出力特徴量の次元、出力はニュートラル、ポジ、ネガの３つの値
        self.cls = nn.Linear(in_features=768, out_features=3)

        # 重みの初期化処理
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # BERTの基本モデル部分のforward
        # attention_showのときは、attention_probsもreturnする
        if attention_show_flg == True:
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)


        # 入力文章の1単語目[CLS]の特徴量を使用して、ポジ・ネガを分類します
        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, 768)  # sizeを[batch_size, hidden_sizeに変換
        out = self.cls(vec_0)

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out


if __name__ == '__main__':
    model = BertForTweetSentimentClassification(bert_model)
    model.train()
    print('model構築完了')









































# class BertLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         super(BertLayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(hidden_size)) # =weight
#         self.beta = nn.Parameter(torch.zeros(hidden_size)) # =bias
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x-u).pow(2).mean(-1, keepdim=True)
#         x = (x-u)/torch.sqrt(s + self.variance_epsilon)
#         return self.gamma*x self.beta


