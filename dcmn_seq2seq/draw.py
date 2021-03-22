import pickle
import re
import nltk
import time
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertModel,BertConfig,BertPooler
import torch
from dcmn_seq2seq.models.bert import Config
from keras.preprocessing.sequence import pad_sequences

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
            elif sid <=2:
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
    src = ['[CLS]']
    keys = []
    key_ans = {}

    while i < len(src_words):
        if src_words[i] == tar_words[j]:
            src.append(src_words[i])
            i += 1
            j += 1
        else:
            p = i + 1
            q = j + 1

            while p < len(src_words):
                while q < len(tar_words) and tar_words[q] != src_words[p]:
                    q += 1
                if q == len(tar_words):
                    src.append(src_words[p])
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
                    if len(abbrs[pre]) > 1:
                        temp = [' '.join(src_words), 'what is {} ?'.format(pre)]
                        label = -1
                        skip_cnt = 0
                        for index, u in enumerate(abbrs[pre]):
                            if index-skip_cnt>=max_pad_length-2:
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
                                tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length\
                                or label<0 or label >= max_pad_length - 2:
                            src.append(pre)
                        else:
                            sentences.append(temp)
                            labels.append(label)
                            keys.append(pre)
                            src.append('[MASK]')
                            key_ans[pre] = label
                    else:
                        src.append(abbrs[pre][0])
                else:
                    src.append(pre)

            i = p
            j = q
    return sentences, labels, src, keys, key_ans


def get_dcmn_data_from_step1(src_words, masks, k_a, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    sentences = []
    src = ['[CLS]']
    keys = []
    labels = []
    for i, mask in enumerate(masks):
        if mask == 0:
            src.append(src_words[i])
            continue
        key = src_words[i]
        if key in abbrs.keys() and key in k_a.keys() and k_a[key]>=0 and k_a[key]<max_pad_length-2:
            if len(abbrs[key]) > 1:
                temp = [' '.join(src_words), 'what is {} ?'.format(key)]
                label = -1
                if key in k_a.keys():
                    label = k_a[key]
                skip_cnt = 0
                for index, u in enumerate(abbrs[key]):
                    if index-skip_cnt >= max_pad_length-2:
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
                    src.append(key)
                    continue
                sentences.append(temp)
                keys.append(key)
                src.append('[SEP] [MASK] [SEP]')
                labels.append(label)
            else:
                src.append(abbrs[key][0])
        elif key in abbrs.keys() and key not in k_a.keys() and len(abbrs[key]) == 1:
            src.append(abbrs[key][0])
        else:
            src.append(key)

    return sentences, labels, src, keys

def add_sep(train_srcs, train_tars):
    train_src_new = []
    train_tar_new = []

    for src, tar in zip(train_srcs, train_tars):
        src = ' '.join(src)
        src = src.split(' ')
        src_new = src
        if src_new[-1] != '.':
            src_new.append('.')
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

def get_embs(dcmn_keys, abbrs, max_pad_length):
    device = torch.device('cuda')
    bert_model = 'bert-base-cased'
    bert = BertModel.from_pretrained(bert_model)
    bert.to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
    pad_tokens =['[PAD]']
    ids = tokenizer.convert_tokens_to_ids(pad_tokens)
    inputs = [ids]
    inputs = torch.tensor(inputs)
    inputs = inputs.to(device)
    with torch.no_grad():
        _, pad_embs = bert(inputs)
    pad_embs = pad_embs.cpu().detach().numpy()
    dcmn_embs = []
    for keys in tqdm(dcmn_keys):
        key_embs = []
        for key in keys:
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
                    emb_values.append(embs.cpu().detach().numpy())
            while len(emb_values) < max_pad_length-2:
                emb_values.append(pad_embs)
            key_embs.append(emb_values)
        dcmn_embs.append(key_embs)
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


def get_index(srcs, batch_size):
    tar_indexs = []
    for i, src in enumerate(srcs):
        p = 0
        indexs = []
        for j, u in enumerate(src):
            if u == 4:
                indexs.append((i%batch_size,j))
        tar_indexs.append(indexs)
    return tar_indexs


def softmax(x, axis=1):
    row_max = x.max(axis=axis)

    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

class DataGenerator():
    def __init__(self, seq_batch_size, max_pad_length=16, max_seq_length=64, cuda=True):
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.abbrs_path = './data/abbrs-all-cased.pkl'
        self.train_txt_path = './data/train(12809).txt'
        self.test_txt_path = './data/test(2030).txt'
        with open(self.abbrs_path, 'rb') as f:
            self.abbrs = pickle.load(f)
        self.train_src_txt, self.train_tar_1_txt, self.train_tar_2_txt = get_train_src_tar_txt(self.train_txt_path)
        # self.train_src_txt = self.train_src_txt[:500]
        # self.train_tar_1_txt = self.train_tar_1_txt[:500]
        # self.train_tar_2_txt = self.train_tar_2_txt[:500]

        self.test_src_txt, self.test_tar_1_txt, self.test_tar_2_txt = get_test_src_tar_txt(self.test_txt_path)

        # generate data
        self.train_order = []
        self.train_seq_srcs = []
        self.train_dcmn_srcs = []
        self.train_dcmn_labels = []
        self.train_keys = []
        self.test_order = []
        self.test_seq_srcs = []
        self.test_dcmn_srcs = []
        self.test_dcmn_labels = []
        self.test_keys=[]
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        for i, (src, tar) in enumerate(zip(self.train_src_txt, self.train_tar_1_txt)):
            src = nltk.word_tokenize(src)
            tar = nltk.word_tokenize(tar)
            sentences, labels, src, keys, key_ans = get_dcmn_data_from_gt(src, tar, self.abbrs, max_pad_length=max_pad_length,
                                                                    max_dcmn_seq_length=max_seq_length, tokenizer=tokenizer)
            self.train_order.append(len(sentences))
            self.train_dcmn_srcs.extend(sentences)
            self.train_dcmn_labels.extend(labels)
            self.train_seq_srcs.append(src)
            self.train_keys.append(keys)

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
            sentences, labels, src, keys, key_ans = get_dcmn_data_from_gt(src, tar, self.abbrs, max_pad_length=max_pad_length,
                                                                          max_dcmn_seq_length=max_seq_length, tokenizer=tokenizer)
            k_a.append(key_ans)

        for i, (sts, masks, k_a) in enumerate(zip(self.test_src_txt, test_mask, k_a)):
            sts = nltk.word_tokenize(sts)
            sentences, labels, src, keys = get_dcmn_data_from_step1(sts, masks, k_a, self.abbrs, max_pad_length=max_pad_length,
                                                                    max_dcmn_seq_length=max_seq_length, tokenizer=tokenizer)

            self.test_keys.append(keys)
            self.test_order.append(len(sentences))
            self.test_dcmn_srcs.extend(sentences)
            self.test_dcmn_labels.extend(labels)
            self.test_seq_srcs.append(src)


        self.train_seq_srcs, self.train_tar_2_txt = add_sep(train_srcs=self.train_seq_srcs, train_tars=self.train_tar_2_txt)
        self.test_seq_srcs = [' '.join(u) for u in self.test_seq_srcs]

        # self.train_embs = get_embs(self.train_keys, self.abbrs, max_pad_length)
        # self.test_embs = get_embs(self.test_keys, self.abbrs, max_pad_length)
        # with open('./data/train_embs.pkl', 'wb') as f:
        #     pickle.dump(self.train_embs, f)
        # with open('./data/test_embs.pkl', 'wb') as f:
        #     pickle.dump(self.test_embs, f)

        with open('./data/train_embs.pkl', 'rb') as f:
            self.train_embs = pickle.load(f)
        with open('./data/test_embs.pkl', 'rb') as f:
            self.test_embs = pickle.load(f)

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

        self.train_indexes = get_index(self.train_seq_srcs_ids, seq_batch_size)
        self.test_indexes = get_index(self.test_seq_srcs_ids, seq_batch_size)

        self.p_train, self.p_test = 0, 0
        self.q_train, self.q_test = 0, 0
        self.p_emb = 0

        self.seq_train_data = []    # src_id(64), src_mask(64), tar_id(64), tar_mask(64), sum_new_embs(num of keys in src_id), indexes(index of key)
                                    # 0              1            2              3            4                                     5
        for src_id, src_mask, tar_id, tar_mask, indexes in zip(self.train_seq_srcs_ids, self.train_seq_srcs_masks,
                                                               self.train_seq_tars_ids, self.train_seq_tars_masks,
                                                               self.train_indexes):
            self.seq_train_data.append([src_id, src_mask, tar_id, tar_mask, [], indexes])

        self.seq_test_data = []  # src_id(64), src_mask(64), tars, cudics, sum_new_embs(num of keys in src_id),
                                  # 0              1            2     3         4
        for src_id, src_mask, tars, cudic in zip(self.test_seq_srcs_ids, self.test_seq_srcs_masks,
                                                               self.seq_test_tars, self.cudics):
            self.seq_test_data.append([src_id, src_mask, tars, cudic, []])

        self.seq_batch_size = seq_batch_size

    def train_data_to_tensor(self, datas):
        src_ids = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        src_masks = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        tar_ids = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        tar_masks = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        sum_new_embs = [_[4] for _ in datas]
        indexes = [_[5] for _ in datas]
        return src_ids, src_masks, tar_ids, tar_masks, sum_new_embs, indexes


    def update_train(self, batch_scores):
        batch_scores = softmax(batch_scores, axis=1)
        while self.p_train < len(self.train_order) and self.train_order[self.p_train] == 0:
            self.p_train += 1

        for scores in batch_scores:

            new_embs = [score*emb for (score, emb) in zip(scores, self.train_embs[self.p_train][self.p_emb])]
            self.p_emb += 1
            sum_new_emb = np.sum(new_embs, axis=0)
            self.seq_train_data[self.p_train][4].append(sum_new_emb)
            if len(self.seq_train_data[self.p_train][4]) == self.train_order[self.p_train]:
                self.p_train += 1
                while self.p_train < len(self.train_order) and self.train_order[self.p_train] == 0:
                    self.p_train += 1
                self.p_emb = 0

        if self.p_train - self.q_train >= self.seq_batch_size:
            batch_data = self.seq_train_data[self.q_train: self.q_train + self.seq_batch_size]
            self.q_train += self.seq_batch_size
            return self.train_data_to_tensor(batch_data)
        else:
            return None

    def restart_train(self):
        self.p_train = 0
        self.q_train = 0
        self.p_emb = 0
        for i in range(len(self.seq_train_data)):
            self.seq_train_data[i][4] = []

    def update_test(self, batch_scores):
        batch_scores = softmax(batch_scores, axis=1)
        while self.p_test < len(self.test_order) and self.test_order[self.p_test] == 0:
            self.p_test += 1

        for scores in batch_scores:
            new_embs = [score * emb for (score, emb) in zip(scores, self.test_embs[self.p_test][self.q_test])]
            sum_new_emb = np.sum(new_embs, axis=0)
            self.seq_test_data[self.p_test][4].append(sum_new_emb)
            if len(self.seq_test_data[self.p_test][4]) == self.test_order[self.p_test]:
                self.p_test += 1

    def restart_test(self):
        self.p_test = 0
        for i in range(len(self.seq_test_data)):
            self.seq_test_data[i][4] = []


    def get_test_sum_embs(self):
        return [_[4] for _ in self.seq_test_data]

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


    # device = torch.device('cuda')
    # bert_model ='bert-base-cased'
    # bert = BertModel.from_pretrained(bert_model)
    # bert.to(device)
    # tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
    # arr = []
    # key = 'home cup cat dog'
    # tokens = tokenizer.tokenize(key)
    # print(tokens)
    # ids = tokenizer.convert_tokens_to_ids(tokens)
    # inputs = [ids]
    # inputs = torch.tensor(inputs)
    # inputs = inputs.to(device)
    #
    # t = time.time()
    # print(inputs)
    # _, outputs = bert(inputs)
    # outputs = outputs.cpu().detach().numpy()
    # print(np.shape(outputs))
    # print(outputs[0][:10])
    #
    # temp = []
    # for u in tokens:
    #     ids = tokenizer.convert_tokens_to_ids([u])
    #     inputs = [ids]
    #     inputs = torch.tensor(inputs)
    #     inputs = inputs.to(device)
    #     _, outputs = bert(inputs)
    #     outputs = outputs.cpu().detach().numpy()
    #     temp.append(outputs[0][:10])
    # temp = np.asarray(temp)
    # print(temp.sum(axis=0)/4.0)
    #
    # print('time cost:',time.time()-t)
    #
    # print(outputs)







#
# txt_loader = None
#
#
# class DataLoader_DCMN():
#     pass
#
# def parse_mc():
#     data, res = None, None
#     examples = data
#     return examples, res
#
# def read_swag_examples():
#     data,res = parse_mc()
#     examples = data # parse data to examples
#     return examples, res
#
#
#
# for i,batch in enumerate(txt_loader):
#     # batch_size = 16
#     examples, res = read_swag_examples()
#     data_loader_dcmn = DataLoader_DCMN()
#     # train dcmn with data_loader_dcmn
#
#     for i,batch in enumerate(data_loader_dcmn):
#         inputs = batch
#         outputs = dcmn(inputs)
#         data_seq = generator.update(outputs)
#         if out is not None:
#             outputs = seq2seq(data_seq)
#
#             loss.backward()
#
#     for i,batch in enumerate(data_loader_seq_res):
#         # train seq with the res
#         pass
