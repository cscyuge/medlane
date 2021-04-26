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

def replace_pro(sentence):
    sentence = sentence.split(' ')
    tar = ''
    for wid, word in enumerate(sentence):
        if word in pre_dic.keys():
            sentence[wid] = pre_dic[word]
    for word in sentence:
        if word != sentence[-1]:
            tar += (word+' ')
        else:
            tar += word
    return tar

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
            avg_score += get_sentence_bleu(candidate[k], reference_)/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score

def count_hit(candidate, dics):
    avg_score = 0
    for sentence, cdic in zip(candidate, dics):
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
        avg_score += score/len(candidate)
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



