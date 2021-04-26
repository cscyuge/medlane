PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
from tqdm import tqdm
import re
import nltk

def build_dataset(config):

    def load_dataset(data, pad_size=50):
        contents = []
        data = data.lower()
        data = data.split('\n\n')
        for lines in tqdm(data):
            lines = lines.split('\n')
            src = lines[0]
            tar = lines[1]
            token_src = config.tokenizer.tokenize(src)
            token_src = [CLS] + token_src + [SEP]
            token_tar = config.tokenizer.tokenize(tar)
            token_tar = [CLS] + token_tar + [SEP]

            seq_len_src = len(token_src)
            mask_src = []
            token_ids_src = config.tokenizer.convert_tokens_to_ids(token_src)

            if pad_size:
                if len(token_src) < pad_size:
                    mask_src = [1] * len(token_ids_src) + [0] * (pad_size - len(token_src))
                    token_ids_src += ([0] * (pad_size - len(token_src)))
                else:
                    mask_src = [1] * pad_size
                    token_ids_src = token_ids_src[:pad_size]
                    seq_len_src = pad_size


            seq_len_tar = len(token_tar)
            mask_tar = []
            token_ids_tar = config.tokenizer.convert_tokens_to_ids(token_tar)

            if pad_size:
                if len(token_tar) < pad_size:
                    mask_tar = [1] * len(token_ids_tar) + [0] * (pad_size - len(token_tar))
                    token_ids_tar += ([0] * (pad_size - len(token_tar)))
                else:
                    mask_tar = [1] * pad_size
                    token_ids_tar = token_ids_tar[:pad_size]
                    seq_len_tar = pad_size

            contents.append((token_ids_src, int(0), seq_len_src, mask_src, token_ids_tar, int(0), seq_len_tar, mask_tar))
        return contents
    train_data = open('./data/train(12809)-v2.txt', 'r', encoding='utf-8').read()
    train_data = load_dataset(train_data, pad_size=config.pad_size)
    return train_data

def build_dataset_eval(config):

    def load_dataset(data, pad_size=50):
        contents = []
        data = data.lower()
        data = data.split('\n\n')
        for lines in tqdm(data):
            lines = lines.split('\n')
            cudic = {}
            tars = []
            for sid, sentence in enumerate(lines):
                if sid == 0:
                    src = lines[0]
                    token_src = config.tokenizer.tokenize(src)
                    token_src = [CLS] + token_src + [SEP]
                else:
                    sentence = sentence[2:]
                    text = re.sub('\[[^\[\]]*\]', '', sentence)
                    pairs = re.findall('[^\[\] ]+\[[^\[\]]+\]', sentence)
                    for pair in pairs:
                        pair = re.split('[\[\]]', pair)
                        cudic[pair[0]] = pair[1]
                    words = nltk.word_tokenize(text)
                    for wid, word in enumerate(words):
                        if word in cudic.keys():
                            words[wid] = cudic[word]
                    new_text = ''
                    for word in words:
                        new_text += word
                        new_text += ' '
                    tars.append(new_text)
            #token_tar = config.tokenizer.tokenize(tar)
            #token_tar = [CLS] + token_tar + [SEP]

            seq_len_src = len(token_src)
            mask_src = []
            token_ids_src = config.tokenizer.convert_tokens_to_ids(token_src)

            if pad_size:
                if len(token_src) < pad_size:
                    mask_src = [1] * len(token_ids_src) + [0] * (pad_size - len(token_src))
                    token_ids_src += ([0] * (pad_size - len(token_src)))
                else:
                    mask_src = [1] * pad_size
                    token_ids_src = token_ids_src[:pad_size]
                    seq_len_src = pad_size


            contents.append((token_ids_src, int(0), seq_len_src, mask_src, tars, cudic))
        return contents
    test_data = open('./data/test(2030)-v2.txt', 'r', encoding='utf-8').read()
    test_data = load_dataset(test_data, pad_size=config.pad_size)
    val_data = test_data[0:len(test_data)//2]
    test_data = test_data[len(test_data)//2:]
    return val_data, test_data
