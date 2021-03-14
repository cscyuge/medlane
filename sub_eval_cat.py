from nlgeval import compute_individual_metrics, compute_metrics
from bleu_eval import count_common, count_hit, count_score
import pickle
import numpy as np
import nltk

results = open('result/best_save_bert.out (1).txt', 'r', encoding='utf-8').readlines()
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
        count+=1
print(count)
print(len(code_exist.keys()))
test_subjects = np.array(results)
test_targets = np.array(ref)
test_dics = np.array(dics)
len_sen = [len(nltk.word_tokenize(x)) for x in sources]
len_sen = np.array(len_sen)
print(len_sen.mean(),len_sen.max() ,len_sen.min())
len_spilt = [(0, 100000)]
for len_current in len_spilt:
    index = np.where((len_sen>=len_current[0])&(len_sen<len_current[1]))
    ref = test_targets[index].tolist()
    hyp = test_subjects[index].tolist()
    open('tmp/hyp.txt', 'w').writelines([x for x in hyp])
    ref0 = [x[0] for x in ref]
    ref1 = [x[1] for x in ref]
    ref2 = [x[2] for x in ref]
    ref3 = [x[3] for x in ref]
    open('tmp/ref0.txt', 'w').writelines([x + '\n' for x in ref0])
    open('tmp/ref1.txt', 'w').writelines([x + '\n' for x in ref1])
    open('tmp/ref2.txt', 'w').writelines([x + '\n' for x in ref2])
    open('tmp/ref3.txt', 'w').writelines([x + '\n' for x in ref3])

    dics = test_dics[index].tolist()
    metrics_dict = compute_metrics(hypothesis='tmp/hyp.txt', references=['tmp/ref0.txt','tmp/ref1.txt','tmp/ref2.txt','tmp/ref3.txt'], no_glove=True, no_overlap=False, no_skipthoughts=True)
    #print(metrics_dict)
    hyp = [nltk.word_tokenize(x) for x in hyp]
    hit = count_hit(hyp, dics)
    com = count_common(hyp)
    BLEU = (metrics_dict['Bleu_1']+metrics_dict['Bleu_2']+metrics_dict['Bleu_3']+metrics_dict['Bleu_4'])/4
    Ascore = (1+2.25+4)/(4/BLEU+2.25/hit+1/com)
    print(BLEU, hit, com, Ascore)
