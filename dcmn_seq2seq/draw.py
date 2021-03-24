import pickle
import re
import nltk
import time
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, BertPooler
import torch
from dcmn_seq2seq.models.bert import Config
from keras.preprocessing.sequence import pad_sequences
from torch.nn.functional import softmax

from tqdm import tqdm
import numpy as np


def merge_mask(src, mask, tokenizer):
    mask_new = []
    i = 0
    for word in src:
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        if n_subwords > 1:
            if 1 in mask[i:i + n_subwords]:
                mask_new.append(1)
            else:
                mask_new.append(0)
        else:
            mask_new.append(mask[i])
        i += n_subwords

    return mask_new


def simplify(word):
    new_word = word.lower().replace('( ', '(').replace(' )', ')').replace('  ', ' ').replace(' ,', ',').strip()
    return new_word


def min_distance(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def is_similar(word_1, word_2, stop=False):
    word_1 = simplify(word_1)
    word_2 = simplify(word_2)
    flag = False
    if ('(' in word_1 or '(' in word_2) and not stop:
        _word_1 = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", word_1)
        _word_2 = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", word_2)
        flag = is_similar(_word_1, _word_2, True)

    return word_1 == word_2 or \
           word_1 in word_2 or \
           word_2 in word_1 or \
           min_distance(word_1, word_2) <= 2 or \
           flag


def get_train_src_tar_txt(train_txt_path):
    src = []
    tar_1 = []
    tar_2 = []
    txt = ''
    try:
        txt += open(train_txt_path, 'r').read()
    except:
        txt += open(train_txt_path, 'r', encoding='utf-8').read()

    txt = txt.split('\n\n')
    for para in txt:
        sentences = para.split('\n')
        if len(sentences) < 2:
            continue
        for sid, sentence in enumerate(sentences[0:3]):
            if sid == 0:
                src.append(sentence)
            elif sid == 1:
                tar_1.append(sentence)
            elif sid == 2:
                tar_2.append(sentence)
    return src, tar_1, tar_2


def get_test_src_tar_txt(test_txt_path):
    txt = open(test_txt_path, 'r').read()
    #     txt = txt.lower()
    txt = txt.split('\n\n')
    src = []
    tar_1 = []
    tar_2 = []
    for para in txt:
        sentences = para.split('\n')
        src_sentence = ''
        if len(sentences) < 2 or len(sentences[0]) < 3 or len(sentences[1]) < 3:
            continue
        for sid, sentence in enumerate(sentences):
            if sid == 0:
                src.append(sentence)
            elif sid <= 2:
                cudic = {}
                sentence = sentence[2:]
                sentence = sentence.replace('].', '] .')
                text = re.sub('\[[^\[\]]*\]', '', sentence)
                pairs = re.findall('[^\[\] ]+\[[^\[\]]+\]', sentence)
                for pair in pairs:
                    pair = re.split('[\[\]]', pair)
                    cudic[pair[0]] = pair[1]
                words = nltk.word_tokenize(text)
                for wid, word in enumerate(words):
                    if word in cudic.keys():
                        words[wid] = cudic[word]
                new_text = ' '.join(words)
                if sid == 1:
                    tar_1.append(new_text)
                else:
                    tar_2.append(new_text)
    return src, tar_1, tar_2


def get_dcmn_data_from_gt(src_words, tar_words, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    if tar_words[-1] != '.':
        tar_words.append('.')
    i = 0
    j = 0
    sentences = []
    labels = []
    srcs = []
    keys = []
    key_ans = {}

    while i < len(src_words):
        if src_words[i] == tar_words[j]:
            i += 1
            j += 1
        else:
            p = i + 1
            q = j + 1

            while p < len(src_words):
                while q < len(tar_words) and tar_words[q] != src_words[p]:
                    q += 1
                if q == len(tar_words):
                    p = p + 1
                    q = j + 1
                else:
                    break
            if p - i == 1:
                pre = src_words[i]
                aft = " ".join(tar_words[j:q])
                if pre in abbrs.keys():
                    pass
                elif pre.upper() in abbrs.keys():
                    pre = pre.upper()
                elif pre.lower() in abbrs.keys():
                    pre = pre.lower()

                if pre in abbrs.keys():
                    temp = [' '.join(src_words), 'what is {} ?'.format(pre)]
                    label = -1
                    skip_cnt = 0
                    for index, u in enumerate(abbrs[pre]):
                        if index - skip_cnt >= max_pad_length - 2:
                            break
                        if len(u.split(' ')) > 10:
                            skip_cnt += 1
                            continue
                        h = u
                        temp.append(h)
                        if is_similar(u, aft):
                            label = index
                    while len(temp) < max_pad_length:
                        temp.append('[PAD]')
                    if len(tokenizer.tokenize(temp[0])) + len(tokenizer.tokenize(temp[1])) + len(
                            tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length \
                            or label < 0 or label >= max_pad_length - 2:
                        pass
                    else:
                        sentences.append(temp)
                        labels.append(label)
                        keys.append(pre)
                        key_ans[pre] = label
                        srcs.append('[CLS] ' + ' '.join(src_words[:i]) + ' [MASK] ' + ' '.join(src_words[p:]))

            i = p
            j = q
    return sentences, labels, srcs, keys, key_ans


def get_dcmn_data_from_step1(src_words, masks, k_a, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    sentences = []
    srcs = []
    keys = []
    labels = []
    for i, mask in enumerate(masks):
        if mask == 0:
            continue
        key = src_words[i]
        if (key in abbrs.keys() and key in k_a.keys() and 0 <= k_a[key] < max_pad_length - 2) or \
                (key in abbrs.keys() and key not in k_a.keys() and len(abbrs[key]) == 1):
            temp = [' '.join(src_words), 'what is {} ?'.format(key)]
            if key in k_a.keys():
                label = k_a[key]
            else:
                label = 0

            skip_cnt = 0
            for index, u in enumerate(abbrs[key]):
                if index - skip_cnt >= max_pad_length - 2:
                    break
                if len(u.split(' ')) > 10:
                    skip_cnt += 1
                    continue
                h = u
                temp.append(h)

            while len(temp) < max_pad_length:
                temp.append('[PAD]')

            if len(tokenizer.tokenize(temp[0])) + len(tokenizer.tokenize(temp[1])) + len(
                    tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length:
                continue
            sentences.append(temp)
            keys.append(key)
            labels.append(label)
            srcs.append('[CLS] ' + ' '.join(src_words[:i]) + ' [SEP] [MASK] [SEP] ' + ' '.join(src_words[i + 1:]))

    return sentences, labels, srcs, keys


def add_sep(train_srcs, train_tars, train_order):
    train_src_new = []
    train_tar_new = []
    i, j = 0, 0
    for src in train_srcs:
        src = src.split(' ')
        src_new = src
        if src_new[-1] != '.':
            src_new.append('.')
        while j >= train_order[i]:
            i += 1
            j = 0
        j += 1
        tar = train_tars[i]
        tar = tar.split(' ')
        tar_new = []
        p = 0

        for i, u in enumerate(src):
            if u == '[MASK]':
                while p < len(tar) and tar[p] != src[i - 1]:
                    tar_new.append(tar[p])
                    p += 1
                if p < len(tar):
                    tar_new.append(tar[p])
                    p += 1
                tar_new.append('[SEP]')
                while p < len(tar) and tar[p] != src[i + 1]:
                    tar_new.append(tar[p])
                    p += 1
                tar_new.append('[SEP]')
        while p < len(tar):
            tar_new.append(tar[p])
            p += 1
        train_src_new.append(src_new)
        train_tar_new.append(tar_new)

    for i, u in enumerate(train_src_new):
        tmp = []
        for j, v in enumerate(u):
            if v == '[MASK]':
                tmp.append('[SEP]')
                tmp.append('[MASK]')
                tmp.append('[SEP]')
            else:
                tmp.append(v)
        train_src_new[i] = tmp
    train_input = []
    for u in train_src_new:
        train_input.append(' '.join(u))
    train_output = []
    for u in train_tar_new:
        train_output.append(' '.join(u))

    return train_input, train_output


def get_embs(bert, tokenizer,device, dcmn_keys, abbrs, max_pad_length):

    pad_tokens = ['[PAD]']
    ids = tokenizer.convert_tokens_to_ids(pad_tokens)
    inputs = [ids]
    inputs = torch.tensor(inputs)
    inputs = inputs.to(device)
    with torch.no_grad():
        _, pad_embs = bert(inputs)
    pad_embs = pad_embs.cpu().detach().numpy()[0]
    dcmn_embs = []
    for key in tqdm(dcmn_keys):
        emb_values = []
        skip_cnt = 0
        for i, value in enumerate(abbrs[key]):
            if len(value.split(' ')) > 10:
                skip_cnt += 1
                continue
            if i - skip_cnt >= max_pad_length - 2:
                break
            tokens = tokenizer.tokenize(key)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            inputs = [ids]
            inputs = torch.tensor(inputs)
            inputs = inputs.to(device)
            with torch.no_grad():
                _, embs = bert(inputs)
                emb_values.append(embs.cpu().detach().numpy()[0])
        while len(emb_values) < max_pad_length - 2:
            emb_values.append(pad_embs)
        dcmn_embs.append(emb_values)
    return dcmn_embs


def seq_tokenize(input_data, tokenizer, max_seq_length):
    ids = []
    for data in tqdm(input_data):
        words = tokenizer.tokenize(data)
        ids.append(words)

    ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in ids],
                        maxlen=max_seq_length, dtype="long", value=0,
                        truncating="post", padding="post")
    masks = [[float(i != 0.0) for i in ii] for ii in ids]
    return ids, masks


def get_index(srcs):
    tar_indexs = []
    for i, src in enumerate(srcs):
        for j, u in enumerate(src):
            if u == 4:
                tar_indexs.append([0.0] * j + [1.0] + [0.0] * (len(src) - j - 1))
                break
    return tar_indexs


class DataGenerator():
    def __init__(self, max_pad_length=16, max_seq_length=64, cuda=True, emb_size=768, tokenizer = None, seq_tokenizer=None):
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.emb_size = emb_size

        self.abbrs_path = './data/abbrs-all-cased.pkl'
        self.train_txt_path = './data/train(12809).txt'
        self.test_txt_path = './data/test(2030).txt'
        with open(self.abbrs_path, 'rb') as f:
            self.abbrs = pickle.load(f)
        self.train_src_txt, self.train_tar_1_txt, self.train_tar_2_txt = get_train_src_tar_txt(self.train_txt_path)
        self.train_src_txt = self.train_src_txt[:500]
        self.train_tar_1_txt = self.train_tar_1_txt[:500]
        self.train_tar_2_txt = self.train_tar_2_txt[:500]

        self.test_src_txt, self.test_tar_1_txt, self.test_tar_2_txt = get_test_src_tar_txt(self.test_txt_path)

        # generate data
        self.train_seq_srcs = []
        self.train_dcmn_srcs = []
        self.train_dcmn_labels = []
        self.train_keys = []
        self.train_order = []
        self.test_seq_srcs = []
        self.test_dcmn_srcs = []
        self.test_dcmn_labels = []
        self.test_keys = []
        self.test_order = []
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        for i, (src, tar) in enumerate(zip(self.train_src_txt, self.train_tar_1_txt)):
            src = nltk.word_tokenize(src)
            tar = nltk.word_tokenize(tar)
            sentences, labels, srcs, keys, key_ans = get_dcmn_data_from_gt(src, tar, self.abbrs,
                                                                           max_pad_length=max_pad_length,
                                                                           max_dcmn_seq_length=max_seq_length,
                                                                           tokenizer=tokenizer)
            self.train_dcmn_srcs.extend(sentences)
            self.train_dcmn_labels.extend(labels)
            self.train_seq_srcs.extend(srcs)
            self.train_keys.extend(keys)
            self.train_order.append(len(sentences))

        with open('./data/test_mask_step2_2030.pkl', 'rb') as f:
            test_mask_step2 = pickle.load(f)
        test_mask = []

        for src, mask in zip(self.test_src_txt, test_mask_step2):
            src = nltk.word_tokenize(src)
            mask_new = merge_mask(src, mask, tokenizer)
            test_mask.append(mask_new)

        k_a = []
        for i, (src, tar) in enumerate(zip(self.test_src_txt, self.test_tar_1_txt)):
            src = nltk.word_tokenize(src)
            tar = nltk.word_tokenize(tar)
            sentences, labels, src, keys, key_ans = get_dcmn_data_from_gt(src, tar, self.abbrs,
                                                                          max_pad_length=max_pad_length,
                                                                          max_dcmn_seq_length=max_seq_length,
                                                                          tokenizer=tokenizer)
            k_a.append(key_ans)

        for i, (sts, masks, k_a) in enumerate(zip(self.test_src_txt, test_mask, k_a)):
            sts = nltk.word_tokenize(sts)
            sentences, labels, srcs, keys = get_dcmn_data_from_step1(sts, masks, k_a, self.abbrs,
                                                                     max_pad_length=max_pad_length,
                                                                     max_dcmn_seq_length=max_seq_length,
                                                                     tokenizer=tokenizer)
            self.test_order.append(len(sentences))
            self.test_keys.extend(keys)
            self.test_dcmn_srcs.extend(sentences)
            self.test_dcmn_labels.extend(labels)
            self.test_seq_srcs.extend(srcs)

        self.train_seq_srcs, self.train_tar_2_txt = add_sep(train_srcs=self.train_seq_srcs,
                                                            train_tars=self.train_tar_2_txt,
                                                            train_order=self.train_order)

        # bert_model = 'bert-base-cased'
        # bert = BertModel.from_pretrained(bert_model)
        # bert.to(self.device)
        # self.train_embs = get_embs(bert, tokenizer, self.device, self.train_keys, self.abbrs, max_pad_length)
        # self.test_embs = get_embs(bert, tokenizer, self.device, self.test_keys, self.abbrs, max_pad_length)
        # with open('./data/train_embs.pkl', 'wb') as f:
        #     pickle.dump(self.train_embs, f)
        # with open('./data/test_embs.pkl', 'wb') as f:
        #     pickle.dump(self.test_embs, f)

        with open('./data/train_embs.pkl', 'rb') as f:
            self.train_embs = pickle.load(f)
        with open('./data/test_embs.pkl', 'rb') as f:
            self.test_embs = pickle.load(f)

        if seq_tokenizer is None:
            seq_config = Config(16)
            seq_tokenizer = seq_config.tokenizer
        self.train_seq_srcs_ids, self.train_seq_srcs_masks = seq_tokenize(self.train_seq_srcs, seq_tokenizer,
                                                                          max_seq_length)
        self.train_seq_tars_ids, self.train_seq_tars_masks = seq_tokenize(self.train_tar_2_txt, seq_tokenizer,
                                                                          max_seq_length)
        self.test_seq_srcs_ids, self.test_seq_srcs_masks = seq_tokenize(self.test_seq_srcs, seq_tokenizer,
                                                                        max_seq_length)
        self.cudics = pickle.load(open('./data/test_cudics.pkl', 'rb'))
        self.seq_test_tars = pickle.load(open('./data/test_tars.pkl', 'rb'))

        self.train_indices = get_index(self.train_seq_srcs_ids)
        self.test_indices = get_index(self.test_seq_srcs_ids)



    def build_dataset_eval(self):
        token_ids_srcs = self.test_seq_srcs_ids
        seq_len_src = 64
        mask_srcs = self.test_seq_srcs_masks

        cudics = self.cudics
        test_tars = self.seq_test_tars
        test_data = []
        for token_ids_src, mask_src, test_tar, cudic in zip(token_ids_srcs, mask_srcs, test_tars, cudics):
            tars = test_tar
            test_data.append((token_ids_src, int(0), seq_len_src, mask_src, tars, cudic))
        return test_data


if __name__ == '__main__':
    # t = time.time()
    dg = DataGenerator(16)
    # print('done, time cost:{}'.format(time.time()-t))

