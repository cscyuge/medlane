import pickle
import re
import nltk
import time
from transformers import BertTokenizer, BertModel
import torch
from dcmn_seq2seq_v2.models.bert import Config
from keras.preprocessing.sequence import pad_sequences
from dcmn_seq2seq_v2.bleu_eval_new import get_score
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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


def get_one_sample(src_words, key, key_tar, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    if key in abbrs.keys():
        pass
    elif key.upper() in abbrs.keys():
        key = key.upper()
    elif key.lower() in abbrs.keys():
        key = key.lower()

    choices = []

    if key in abbrs.keys() and key_tar is not None:
        temp = [' '.join(src_words), 'what is {} ?'.format(key)]
        label = -1
        skip_cnt = 0
        for index, u in enumerate(abbrs[key]):
            if index - skip_cnt >= max_pad_length - 2:
                break
            if len(u.split(' ')) > 10:
                skip_cnt += 1
                continue
            temp.append(u)
            choices.append(u)
            if is_similar(u, key_tar):
                label = index - skip_cnt
        while len(temp) < max_pad_length:
            temp.append('[PAD]')
            choices.append('[PAD]')

        if len(tokenizer.tokenize(temp[0])) + len(tokenizer.tokenize(temp[1])) + len(
                tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length \
                or label < 0 or label >= max_pad_length - 2:
            return None, None, None
        else:
            return temp, label, choices
    else:
        # return None, None, None, None
        temp = [' '.join(src_words), 'what is {} ?'.format(key), key]
        choices.append(key)
        while len(temp) < max_pad_length:
            temp.append('[PAD]')
            choices.append('[PAD]')
        if len(tokenizer.tokenize(temp[0])) + len(tokenizer.tokenize(temp[1])) + len(
                tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length:
            return None, None, None
        return temp, 0, choices


def get_dcmn_data_from_gt(src_words, tar_words, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    if tar_words[-1] != '.':
        tar_words.append('.')
    i = 0
    j = 0
    sentences = []
    labels = []
    key_choices = []
    seq_src_words = src_words
    indics = []
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
            aft = " ".join(tar_words[j:q])
            for k, word in enumerate(src_words[i:p]):
                temp, label, choices = get_one_sample(src_words, word, aft, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer)
                if temp is not None:
                    sentences.append(temp)
                    labels.append(label)
                    key_choices.append(choices)
                    key_ans[word] = temp[label + 2]
                    seq_src_words[i+k] = '[UNK]'
                    indics.extend([j,q+1])

            i = p
            j = q


    seq_tar_words = []
    for i,word in enumerate(tar_words):
        if i in indics:
            seq_tar_words.append('[MASK]')
        seq_tar_words.append(word)

    return sentences, labels, ' '.join(seq_src_words), key_ans, key_choices, ' '.join(seq_tar_words)


def get_dcmn_data_from_step1(src_words, masks, k_a, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    sentences = []
    seq_src_words = src_words
    labels = []
    key_choices = []
    for i, mask in enumerate(masks):
        if mask == 0:
            continue
        key = src_words[i]
        if key in abbrs.keys():
            pass
        elif key.upper() in abbrs.keys():
            key = key.upper()
        elif key.lower() in abbrs.keys():
            key = key.lower()

        if key in k_a.keys():
            aft = k_a[key]
        elif key in abbrs.keys() and len(abbrs[key]) == 1:
            aft = abbrs[key][0]
        else:
            aft = None
        temp, label, choices = get_one_sample(src_words, key, aft, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer)
        if temp is not None:
            sentences.append(temp)
            labels.append(label)
            key_choices.append(choices)
            seq_src_words[i] = '[UNK]'

    return sentences, labels, ' '.join(seq_src_words), key_choices


def seq_tokenize(input_data, tokenizer, max_seq_length):
    ids = []
    for data in input_data:
        words = tokenizer.tokenize(data)
        ids.append(words)

    ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in ids],
                        maxlen=max_seq_length, dtype="long", value=0,
                        truncating="post", padding="post")
    masks = [[float(i != 0.0) for i in ii] for ii in ids]
    return ids, masks


class DataGenerator:
    def __init__(self, train_batch_size, eval_batch_size, max_pad_length=16, max_seq_length=64,
                 cuda=True, emb_size=768, tokenizer=None, seq_tokenizer=None):

        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.emb_size = emb_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_pad_length = max_pad_length
        self.max_seq_length = max_seq_length

        self.abbrs_path = './data/abbrs-all-cased.pkl'
        self.train_txt_path = './data/train(12809).txt'
        self.test_txt_path = './data/test(2030).txt'
        with open(self.abbrs_path, 'rb') as f:
            self.abbrs = pickle.load(f)
        self.train_src_txt, self.train_tar_1_txt, self.train_tar_2_txt = get_train_src_tar_txt(self.train_txt_path)
        self.train_src_txt = self.train_src_txt
        self.train_tar_1_txt = self.train_tar_1_txt
        self.train_tar_2_txt = self.train_tar_2_txt

        self.test_src_txt, self.test_tar_1_txt, self.test_tar_2_txt = get_test_src_tar_txt(self.test_txt_path)

        # generate data
        self.train_seq_srcs = []
        self.train_seq_tars = []
        self.train_dcmn_srcs = []
        self.train_dcmn_labels = []
        self.train_key_choices = []
        self.test_seq_srcs = []
        self.test_dcmn_srcs = []
        self.test_dcmn_labels = []
        self.test_key_choices = []

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.tokenizer = tokenizer

        self._pad_train_data()

        for i, (src, tar) in enumerate(zip(self.train_src_txt, self.train_tar_1_txt)):
            src = nltk.word_tokenize(src)
            tar = nltk.word_tokenize(tar)
            sentences, labels, _src, key_ans, key_choices, _tar = get_dcmn_data_from_gt(src, tar, self.abbrs,
                                                                                  max_pad_length=max_pad_length,
                                                                                  max_dcmn_seq_length=max_seq_length,
                                                                                  tokenizer=tokenizer)
            if (len(sentences)!=_src.count('[UNK]')):
                print(i, src, len(sentences))
            self.train_dcmn_srcs.extend(sentences)
            self.train_dcmn_labels.extend(labels)
            self.train_seq_srcs.append(_src)
            self.train_seq_tars.append(_tar)
            self.train_key_choices.append(key_choices)

        with open('./data/test_mask_step2_2030.pkl', 'rb') as f:
            test_mask_step2 = pickle.load(f)
        self.test_mask = []

        mask_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        for src, mask in zip(self.test_src_txt, test_mask_step2):
            src = nltk.word_tokenize(src)
            mask_new = merge_mask(src, mask, mask_tokenizer)
            self.test_mask.append(mask_new)

        k_as = []
        for i, (src, tar) in enumerate(zip(self.test_src_txt, self.test_tar_1_txt)):
            src = nltk.word_tokenize(src)
            tar = nltk.word_tokenize(tar)
            sentences, labels, _src, key_ans, _, _tar = get_dcmn_data_from_gt(src, tar, self.abbrs,
                                                                        max_pad_length=max_pad_length,
                                                                        max_dcmn_seq_length=max_seq_length,
                                                                        tokenizer=tokenizer)
            k_as.append(key_ans)

        for i, (sts, masks, k_a) in enumerate(zip(self.test_src_txt, self.test_mask, k_as)):
            sts = nltk.word_tokenize(sts)
            sentences, labels, _src, key_choices = get_dcmn_data_from_step1(sts, masks, k_a, self.abbrs,
                                                                            max_pad_length=max_pad_length,
                                                                            max_dcmn_seq_length=max_seq_length,
                                                                            tokenizer=tokenizer)
            self.test_dcmn_srcs.extend(sentences)
            self.test_dcmn_labels.extend(labels)
            if (len(sentences)!=_src.count('[UNK]')):
                print(i, sts)
            self.test_seq_srcs.append(_src)
            self.test_key_choices.append(key_choices)

        if seq_tokenizer is None:
            seq_config = Config(16)
            seq_tokenizer = seq_config.tokenizer
        self.seq_tokenizer = seq_tokenizer

        for i in range(len(self.train_seq_srcs)):
            self.train_seq_srcs[i] = '[CLS] ' + self.train_seq_srcs[i] + ' [SEP]'
        for i in range(len(self.test_seq_srcs)):
            self.test_seq_srcs[i] = '[CLS]' + self.test_seq_srcs[i] + ' [SEP]'

        # for u in self.train_tar_1_txt:
        #     self.train_seq_tars.append('[CLS] ' + u + ' [SEP]')

        self._train_seq_srcs = self.train_seq_srcs[:]
        self._test_seq_srcs = self.test_seq_srcs[:]

        self.train_seq_srcs_ids, self.train_seq_srcs_masks = seq_tokenize(self.train_seq_srcs, seq_tokenizer,
                                                                          max_seq_length)
        self.train_seq_tars_ids, self.train_seq_tars_masks = seq_tokenize(self.train_seq_tars, seq_tokenizer,
                                                                          max_seq_length)
        self.test_seq_srcs_ids, self.test_seq_srcs_masks = seq_tokenize(self.test_seq_srcs, seq_tokenizer,
                                                                        max_seq_length)
        self.cudics = pickle.load(open('./data/test_cudics.pkl', 'rb'))
        self.seq_test_tars = pickle.load(open('./data/test_tars.pkl', 'rb'))

    def valid(self, results):
        sentences = []
        for src, words in zip(self.test_seq_srcs, results):
            src = src.split('[MASK]')
            words = words.split('[MASK]')
            sts = ''
            for i,u in enumerate(src):
                if i%2==1 and i<len(words):
                    sts += words[i]
                else:
                    sts += u
            sentences.append(sts)

        with open('./result/tmp.out.txt', 'w', encoding='utf-8') as f:
            f.writelines([x.lower() + '\n' for x in sentences])
        bleu, hit, com, ascore = get_score()
        return bleu, hit, com, ascore

    def _pad_train_data(self):
        data_size = 0
        for i, (src, tar) in enumerate(zip(self.train_src_txt, self.train_tar_1_txt)):
            src = nltk.word_tokenize(src)
            tar = nltk.word_tokenize(tar)
            sentences, _, __, ___, ____, _____ = get_dcmn_data_from_gt(src, tar, self.abbrs,
                                                          max_pad_length=self.max_pad_length,
                                                          max_dcmn_seq_length=self.max_seq_length,
                                                          tokenizer=self.tokenizer)
            data_size += len(sentences)

        if data_size % self.train_batch_size == 1:
            self.train_src_txt.append('Labs were significant for wbc .')
            self.train_tar_1_txt.append('Labs were significant for white blood cell .')
            self.train_tar_2_txt.append('Labs were significant for white blood cell .')

    def decode_sentence(self, symbols):
        sentences = []
        for symbol_sen in symbols:
            words = self.seq_tokenizer.convert_ids_to_tokens(symbol_sen)
            temp = ''
            for word in words:
                if word == '[SEP]':
                    break
                if word[0] == '#':
                    word = word[2:]
                    temp += word
                else:
                    temp += ' '
                    temp += word
            sentences.append(temp)
        return sentences

    def push(self, outs):
        p = len(self.train_seq_srcs_ok)
        for out in outs:
            while p < len(self.train_seq_srcs) and self.train_seq_srcs[p].find('[UNK]') == -1:
                self.train_seq_srcs_ok.append(self.train_seq_srcs[p])
                p += 1
                self.k = 0
            if p < len(self.train_seq_srcs):
                self.train_seq_srcs[p] = self.train_seq_srcs[p].replace('[UNK]', '[MASK] '+self.train_key_choices[p][self.k][out]+' [MASK]', 1)
                self.k += 1
        while p < len(self.train_seq_srcs) and self.train_seq_srcs[p].find('[UNK]') == -1:
            self.train_seq_srcs_ok.append(self.train_seq_srcs[p])
            p += 1
            self.k = 0

    def get(self):
        p = len(self.train_seq_srcs_ok)
        if p - self.q >= self.train_batch_size or p == len(self.train_seq_srcs):
            src_ids, src_masks, tar_ids, tar_masks = self.get_train_batch(self.q, self.q + self.train_batch_size)
            self.q += self.train_batch_size
            return src_ids, src_masks, tar_ids, tar_masks
        return None

    def reset(self):
        self.k = 0
        self.q = 0
        self.train_seq_srcs_ok = []
        self.test_seq_srcs_ok = []
        self.train_seq_srcs = self._train_seq_srcs[:]
        self.test_seq_srcs = self._test_seq_srcs[:]

    def get_eval_dataloader(self, outs):
        p = 0
        k = 0
        for out in outs:
            while p < len(self.test_seq_srcs) and self.test_seq_srcs[p].find('[UNK]') == -1:
                self.test_seq_srcs_ok.append(self.test_seq_srcs[p])
                p += 1
                k = 0
            if p < len(self.test_seq_srcs):
                self.test_seq_srcs[p] = self.test_seq_srcs[p].replace('[UNK]', '[MASK] '+self.test_key_choices[p][k][out]+' [MASK]', 1)
                k += 1
        while p < len(self.test_seq_srcs) and self.test_seq_srcs[p].find('[UNK]') == -1:
            self.test_seq_srcs_ok.append(self.test_seq_srcs[p])
            p += 1
            k = 0

        test_srcs = self.test_seq_srcs_ok
        src_ids, src_masks = seq_tokenize(test_srcs, self.seq_tokenizer, self.max_seq_length)
        src_ids = torch.LongTensor(src_ids)
        src_masks = torch.LongTensor(src_masks)
        data = TensorDataset(src_ids, src_masks)
        sampler = SequentialSampler(data)

        dataloader = DataLoader(data, sampler=sampler, batch_size=self.eval_batch_size)
        return dataloader

    def get_train_batch(self, p, q):
        if q > len(self.train_seq_srcs):
            q = len(self.train_seq_srcs)
        train_srcs = self.train_seq_srcs_ok[p:q]
        train_tars = self.train_seq_tars[p:q]
        src_ids, src_masks = seq_tokenize(train_srcs, self.seq_tokenizer, self.max_seq_length)
        tar_ids, tar_masks = seq_tokenize(train_tars, self.seq_tokenizer, self.max_seq_length)
        src_ids = torch.LongTensor(src_ids)
        src_masks = torch.LongTensor(src_masks)
        tar_ids = torch.LongTensor(tar_ids)
        tar_masks = torch.LongTensor(tar_masks)
        return src_ids, src_masks, tar_ids, tar_masks


if __name__ == '__main__':
    from dcmn_seq2seq.config import DCMN_Config
    import dcmn_seq2seq.models.bert as seq_bert

    dcmn_config = DCMN_Config()

    tokenizer = dcmn_config.tokenizer
    seq_config = seq_bert.Config(dcmn_config.seq_batch_size, dcmn_config.no_cuda)
    dg = DataGenerator(train_batch_size=dcmn_config.train_batch_size,
                       eval_batch_size=dcmn_config.eval_batch_size,
                       max_pad_length=dcmn_config.num_choices + 2,
                       max_seq_length=dcmn_config.max_seq_length, cuda=not dcmn_config.no_cuda,
                       tokenizer=dcmn_config.tokenizer, seq_tokenizer=seq_config.tokenizer)
    print(len(dg.train_seq_srcs_masks))
    print(len(dg.train_dcmn_srcs))
    print(len(dg.test_seq_srcs_ids))

    # print(len(dg.train_dcmn_srcs))
    # print('done, time cost:{}'.format(time.time()-t))

    # bleu, hit, com, ascore = get_score()
