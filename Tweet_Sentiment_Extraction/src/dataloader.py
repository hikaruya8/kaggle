import torchtext
import string
import re

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



# max_length = 256
# TEXT = torchtext.data.Field(sequential=True, tokenize=Tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
# LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# train_val_ds, test_ds = torchtext.data.TabularDataset

# def
# if __name__ == '__main__':
