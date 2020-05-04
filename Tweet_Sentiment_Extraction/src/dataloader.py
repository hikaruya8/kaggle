import torchtext
import string
import re
import random
from torchtext.vocab import Vectors

def preprocessing_text(text):
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

        text = text.replace(".", " . ")
        text = text.replace(",", " , ")

        return text

# 分かち書き
def tokenizer_punctuation(text):
    return text.strip().split()

# 前処理と分かち書きをまとめる
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret

# test tokenizer with preprocessing
# print(tokenizer_with_preprocessing('I like dogs.'))


#  データを読み込む際の読み込んだ内容に対して行う処理を定義
max_length = 256
ID = torchtext.data.Field(sequential=False, use_vocab=False)
TEXT1 = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>") # raw text
TEXT2 = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>") # selected_text
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: 0 if l == 'neutral' else 1 if l == 'positive' else 2, is_target=True) # sentiment label

train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='../data/', train='train.csv',
    test='test.csv', format='csv',
    fields=[('ID', ID), ('Text1', TEXT1), ('Text2', TEXT2), ('Label', LABEL)])

# # test dataloader
# print('訓練 検証のデータ数: {}'.format(len(train_val_ds)))
# print('１つ目の訓練&検証データ:{}'.format(vars(train_val_ds[27476])))

train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))
# # test split data
# print(len(train_ds))
# print(len(val_ds))
# print(vars(train_ds[0]))

# make vocab
fasttext_vectors = Vectors(name='../data/wiki-news-300d-1M.vec')
# test vectors
print(fasttext_vectors.dim)
print(len(fasttext_vectors.itos))

#  ベクトル化したボキャブラリーを作成
TEXT1.build_vocab(train_ds, vectors=fasttext_vectors, min_freq=10)
TEXT2.build_vocab(train_ds, vectors=fasttext_vectors, min_freq=10)
# ボキャブラリのベクトル確認
print(TEXT1.vocab.vectors.shape)
print(TEXT1.vocab.vectors)
print(TEXT1.vocab.stoi)