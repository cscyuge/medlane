import pickle
from nltk.translate.bleu_score import sentence_bleu
#pre_dic = pickle.load(open('data/test_dic.pkl', 'rb'))
import pickle
import random
import os
import re
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from nlgeval import compute_individual_metrics, compute_metrics
import numpy as np

words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

common_words = open('./data/common.txt', 'r').readlines()
common_words = [re.sub('\n', '', w) for w in common_words]
common_words = set(common_words+words)
pro_dic = pickle.load(open('./data/exist_dic_new.pkl', 'rb'))
pro_words = set(pro_dic.keys())

def lemmatize(word, tag):
    if tag.startswith('NN'):
        return wordnet_lemmatizer.lemmatize(word, pos='n')
    elif tag.startswith('VB'):
        return wordnet_lemmatizer.lemmatize(word, pos='v')
    elif tag.startswith('JJ'):
        return wordnet_lemmatizer.lemmatize(word, pos='a')
    elif tag.startswith('R'):
        return wordnet_lemmatizer.lemmatize(word, pos='r')
    else:
        return word

def mark_sentence(sentence):
    sentence = sentence.lower()
    count_pro = 0
    count_unc = 0
    count_total = 0
    sentence = re.sub(r'-?\d+\.?\d*e?-?\d*?', ' num ', sentence)
    words = nltk.word_tokenize(sentence.lower())
    tag = nltk.pos_tag(words)
    for wid, word in enumerate(words):
        word = lemmatize(word, tag[wid][1])
        count_total += 1
        if word in pro_words:
            words[wid] = 'PRO'
            count_pro += 1
        else:
            if word not in common_words and word.isalpha():
                words[wid] = 'UNCOMMON'
                count_unc += 1
            else:
                words[wid] = word
        if words[wid]== 'num':
            words[wid] = 'NUM'
    return count_unc, count_pro, count_total

# def replace_pro(sentence):
#     sentence = sentence.split(' ')
#     tar = ''
#     for wid, word in enumerate(sentence):
#         if word in pre_dic.keys():
#             sentence[wid] = pre_dic[word]
#     for word in sentence:
#         if word != sentence[-1]:
#             tar += (word+' ')
#         else:
#             tar += word
#     return tar

def contact_word_spilt(sentence):
    sentence = re.sub('@@ ', '', sentence)
    #sentence = replace_pro(sentence)
    sentence = sentence.split(' ')
    return sentence

def get_sentence_bleu(candidate, reference):
    score = sentence_bleu(reference, candidate)
    return score


def count_score(candidate, reference):
    avg_score = 0
    for k in range(len(candidate)):
        reference_ = reference[k]
        for m in range(len(reference_)):
            reference_[m] = nltk.word_tokenize(reference_[m])
        candidate[k] = nltk.word_tokenize(candidate[k])
        try:
            tmp = get_sentence_bleu(candidate[k], reference_)
            if tmp < 0.2:
                print(' '.join(reference_[0]))
            avg_score += tmp/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score

def count_hit(candidate, dics):
    avg_score = 0
    for sentence, cdics in zip(candidate, dics):
        max_score = 0
        for cdic in cdics:
            words = sentence
            txt = ''
            for word in words:
                txt += word
                txt += ' '
            count = 0
            for value in cdic.values():
                rs = re.findall(value, txt)
                if len(rs) > 0:
                    count += 1
            if len(cdic) == 0:
                score = 1.0
            else:
                score = count/len(cdic)
            if score > max_score:
                max_score = score
        avg_score += max_score/len(candidate)
    return avg_score

def count_common(candidate):
    avg_score = 0
    for sentence in candidate:
        txt = ''
        for word in sentence:
            txt += word
            txt += ' '
        txt = txt[0:-1]
        unc, pro, count = mark_sentence(txt)
        coms = (count-unc-pro) / (count+1e-3)
        avg_score += coms/len(candidate)
    return avg_score


def count_feature_score(candidates):
    unss = []
    pross = []
    for sentence in candidates:
        unc, pro, count = mark_sentence(sentence)
        uns = unc / count
        pros = pro / count
        unss.append(uns)
        pross.append(pros)
    unss = np.array(unss)
    pross = np.array(pross)
    return 1-unss.mean()-pross.mean()


def get_score():
    results = open('./result/tmp.out.txt', 'r', encoding='utf-8').readlines()
    results = results[1015:]
    sources = open('testdata/test.moses.pro', 'r').readlines()
    sources = [x.replace('\n', '') for x in sources]
    ref = pickle.load(open('testdata/test.cus.pkl', 'rb'))
    dics = pickle.load(open('testdata/test_dic.pkl', 'rb'))
    sen2code = pickle.load(open('data/sen2code.pkl', 'rb'))
    sen2code_new = {}
    for key, value in sen2code.items():
        sen2code_new[key.lower()] = value
    del sen2code
    count = 0
    code_exist = {}
    for source in sources:
        try:
            if sen2code_new[source] not in code_exist.keys():
                code_exist[sen2code_new[source]] = 1
            else:
                code_exist[sen2code_new[source]] += 1
        except:
            count += 1
    # print(count)
    # print(len(code_exist.keys()))
    test_subjects = np.array(results)
    test_targets = np.array(ref)
    test_dics = np.array(dics)
    len_sen = [len(nltk.word_tokenize(x)) for x in sources]
    len_sen = np.array(len_sen)
    # print(len_sen.mean(), len_sen.max(), len_sen.min())
    len_spilt = [(0, 100000)]

    for len_current in len_spilt:
        index = np.where((len_sen >= len_current[0]) & (len_sen < len_current[1]))
        ref = test_targets[index].tolist()
        hyp = test_subjects[index].tolist()
        open('./tmp/hyp.txt', 'w').writelines([x for x in hyp])
        ref0 = [x[0] for x in ref]
        ref1 = [x[1] for x in ref]
        ref2 = [x[2] for x in ref]
        ref3 = [x[3] for x in ref]
        open('./tmp/ref0.txt', 'w').writelines([x + '\n' for x in ref0])
        open('./tmp/ref1.txt', 'w').writelines([x + '\n' for x in ref1])
        open('./tmp/ref2.txt', 'w').writelines([x + '\n' for x in ref2])
        open('./tmp/ref3.txt', 'w').writelines([x + '\n' for x in ref3])

        dics = test_dics[index].tolist()

        metrics_dict = compute_metrics(hypothesis='./tmp/hyp.txt',
                                       references=['./tmp/ref0.txt', './tmp/ref1.txt', './tmp/ref2.txt', './tmp/ref3.txt'],
                                       no_glove=True, no_overlap=False, no_skipthoughts=True)
        # print(metrics_dict)
        hyp = [nltk.word_tokenize(x) for x in hyp]
        hit = count_hit(hyp, dics)
        #     hit=1
        com = count_common(hyp)
        BLEU = (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4
        if BLEU<0.0001:
            BLEU = 0.0001
        if hit<0.0001:
            hit = 0.0001
        if com<0.0001:
            com = 0.0001
        Ascore = (1 + 2.25 + 4) / (4 / BLEU + 2.25 / hit + 1 / com)
        return BLEU, hit, com, Ascore

# ref = pickle.load(open('testdata/test.cus.pkl', 'rb'))
# dics = pickle.load(open('testdata/test_dic.pkl', 'rb'))

# results = open('result/best_save_bert.out.txt', 'r').readlines()
# print(count_score(results, ref))

