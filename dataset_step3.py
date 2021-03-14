PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
from tqdm import tqdm
import re
import nltk
import pickle

def build_dataset(config):
    token_ids_srcs = pickle.load(open("./data_tmp/train_src_23.pkl", 'rb'))
    seq_len_src = 64
    mask_srcs = pickle.load(open("./data_tmp/train_src_masks_23.pkl", 'rb'))

    token_ids_tars = pickle.load(open("./data_tmp/train_tar_23.pkl", 'rb'))
    seq_len_tar = 64
    mask_tars = pickle.load(open("./data_tmp/train_tar_masks_23.pkl",'rb'))

    train_data = []
    for token_ids_src, mask_src, token_ids_tar, mask_tar in zip(token_ids_srcs, mask_srcs, token_ids_tars, mask_tars):
        train_data.append((token_ids_src, int(0), seq_len_src, mask_src, token_ids_tar, int(0), seq_len_tar, mask_tar))
    return train_data[:11000]

def build_dataset_eval(config):
    token_ids_srcs = pickle.load(open("./data_tmp/test_src_23.pkl", 'rb'))
    seq_len_src = 64
    mask_srcs = pickle.load(open("./data_tmp/test_src_masks_23.pkl", 'rb'))

    cudics = pickle.load(open('./data_tmp/test_cudics.pkl', 'rb'))
    test_tars = pickle.load(open('./data_tmp/test_tars.pkl', 'rb'))
    test_data = []
    for token_ids_src, mask_src, test_tar, cudic in zip(token_ids_srcs, mask_srcs, test_tars, cudics):
        tars = test_tar
        test_data.append((token_ids_src, int(0), seq_len_src, mask_src, tars, cudic))
    # val_data = test_data[0:len(test_data)//2]
    # test_data = test_data[len(test_data)//2:]
    # return val_data[:100], test_data[:100]
    return test_data, test_data