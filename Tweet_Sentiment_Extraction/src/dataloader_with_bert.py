import torchtext
import string
import re
import random
import torch
from torchtext.vocab import Vectors
import tensorflow as tf
from transformers import *
import pdb


def preprocessing_text(text):
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

        text = text.replace(".", " . ")
        text = text.replace(",", " , ")

        return text

# make vocab with pre-trained BERT model
# load pre-trained tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# test tokenizer
# text = "Here is the sentence I want embeddings for."
# marked_text = "[CLS] " + text + " [SEP]"
# print(tokenizer.tokenize(marked_text))
# model
# model = BertModel.from_pretrained('bert-base-uncased')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0) # vocavb idsに
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple



def get_tweets_and_sentiment_label_loaders(max_length=256, batch_size=64):
    #  データを読み込む際の読み込んだ内容に対して行う処理を定義
    max_length = max_length
    batch_size = batch_size
    ID = torchtext.data.Field(sequential=False, use_vocab=False)
    TEXT1 = torchtext.data.Field(sequential=True, tokenize=bert_tokenizer.encode, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>") # raw text
    TEXT2 = torchtext.data.Field(sequential=True, tokenize=bert_tokenizer.encode, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>") # selected_text
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: 0 if l == 'neutral' else 1 if l == 'positive' else 2, is_target=True) # sentiment label
    TEST_TEXT = torchtext.data.Field(sequential=True, tokenize=bert_tokenizer.encode, use_vocab=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>") # raw_text
    TEST_LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: 0 if l == 'neutral' else 1 if l == 'positive' else 2, is_target=True) # sentiment label

    train_val_ds = torchtext.data.TabularDataset(
        path='../data/train.csv', format='csv',
        fields=[('ID', None), ('Text1', TEXT1), ('Text2', TEXT2), ('Label', LABEL)],
        skip_header=True)

    test_ds = torchtext.data.TabularDataset(
        path='../data/test.csv', format='csv',
        fields=[('ID', None), ('Test_Text', TEST_TEXT), ('Test_Label', TEST_LABEL)],
        skip_header=True)

    # test dataloader
    print('訓練 検証のデータ数: {}'.format(len(train_val_ds)))
    print('１つ目の訓練&検証データ:{}'.format(vars(train_val_ds[0])))
    print('テストのデータ数: {}'.format(len(test_ds)))
    print('１つ目のテストデータ:{}'.format(vars(test_ds[0])))
    print('TEXT1のids tensor: {}'.format(vars(torch.IntTensor(train_val_ds[0].Text1))))

    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))
    # # test split data
    # print(len(train_ds))
    # print(len(val_ds))
    # print(vars(train_ds[0]))

    # test vectors
    # print(fasttext_vectors.dim)
    # print(len(fasttext_vectors.itos))
    pdb.set_trace()
    #  ベクトル化したボキャブラリーを作成
    # TEXT1.build_vocab(train_ds, vectors=fasttext_vectors, min_freq=10)
    # TEXT2.build_vocab(train_ds, vectors=fasttext_vectors, min_freq=10)
    # TEST_TEXT.build_vocab(test_ds, vectors=fasttext_vectors, min_freq=10)
    # # ボキャブラリのベクトル確認
    # print(TEXT1.shape)
    # print(TEXT1.vectors)
    # print(TEXT1.vocab.stoi)
    # print(TEST_TEXT.vocab.vectors.shape)
    # print(TEST_TEXT.vocab.vectors)
    # print(TEST_TEXT.vocab.stoi)


    # make Dataloader
    train_dl = torchtext.data.Iterator(train_ds, batch_size=24, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=24, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=24, sort=False)

    # test
    batch = next(iter(train_dl))
    print(batch.Text1)
    print(batch.Text2)
    print(batch.Label)

    return train_dl, val_dl, test_dl, TEXT1, TEXT2, TEST_TEXT


if __name__ == '__main__':
    get_tweets_and_sentiment_label_loaders(max_length=256, batch_size=64)