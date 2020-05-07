from dataloader import get_tweets_and_sentiment_label_loaders
from IPython.display import HTML

def highlight(word, attn):
    'Attentionの値が大きければ文字の背景が濃い赤になるHTMLを出力させる関数'
    html_color = '#%02X%02X%02X' % (
        255, int(255*(1-attn)), int(255*(1-attn)))
    return '<span style="bachgreound-color: {}"> {},/span>'.format(html_color, word)

def mk_html(index, batch, preds, normalized_weights_1, normalized_weights_2, TEXT1):
    # HTMLデータの作成

    # index の結果抽出
    sentence = batch.Text1[0][index] #文章
    label = batch.Label[index]
    pred = preds[index]

    # indexのattentionを抽出、規格化
    attens1 = normalized_weights_1[index, 0, :] # 0番目 <cls>のAttention
    attens1 /= attens1.max
    attens2 = normalized_weights_2[index, 0, :] # 0番目 <cls>のAttention
    attens2 /= attens2.max

    # ラベルの予測結果を文字に置き換え
    if label == 0:
        label_str = 'Neutral'
    elif label == 1:
        label_str = 'Positive'
    else:
        label_str = 'Negative'

    if pred == 0:
        pred_str = 'Neutral'
    elif label == 1:
        pred_str = 'Positive'
    else:
        pred_str = 'Negative'

    # 表示用のHTMLを作成
    html = '正解ラベル: {}<br>推論ラベル: {}<br><br>'.format(label_str, pred_str)

    # １段目のAttention
    html += '[TransformerBlockの１段目のAttentionを可視化]<br>'
    for word, attn, in zip(sentence, attens1):
        html += highlight(TEXT1.vocab.itos[word], [attn])
    html += "<br><br>"

    # 2段目のAttention
    html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    for word, attn, in zip(sentence, attens2):
        html += highlight(TEXT1.vocab.itos[word], [attn])

    html += "<br><br>"

    return html

def visualize():
    # データの読み込み
    train_dl, val_dl, test_dl, TEXT1, TEXT2, TEST_TEXT = get_tweets_and_sentiment_label_loaders(max_length=256, batch_size=64)
    # load model
    net_trained = TransformerClassification(text_embedding_vectors=TEXT1.vocab.vectors, d_model=300, max_seq_len=256, output_dim=3)
    net_trained.load_state_dict(torch.load(saved_model_path))

    batch = next(iter(test_dl))

    inputs = batch.Test_Text[0].to(device)
    labels = batch.Test_Label.to(device)

    #make mask
    input_pad = 1
    input_mask = (inputs != input_pad)


    outputs, normalized_weights_1, normalized_weights_2 = net_trained(inputs, input_mask)
    _, preds = torch.max(outputs, 1)

    index = 3 # 出力させたいデータ
    html_output = mk_html(index, batch, preds, normalized_weights_1, normalized_weights_2, TEXT1)
    HTML(html_output)

if __name__ == '__main__':
    visualize()