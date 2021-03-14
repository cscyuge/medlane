# coding: UTF-8
import time
import torch
import pickle
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tqdm import tqdm
from utils import build_iterator, get_time_dif
PAD, CLS = '[PAD]', '[CLS]'

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


def evaluate(config, model, data_set, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    batch_size = 10
    b_len = len(data_set) // batch_size
    data_set = build_iterator(data_set, config)
    embeddings = []
    with torch.no_grad():
        for batch_data, label in data_set:
            outputs = model(batch_data)
            embeddings.append(outputs.cpu().detach().numpy())

    return np.concatenate(embeddings, 0)

def  test(config, model, test_set):
    # test
    #model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    hidden, pooled = evaluate(config, model, test_set, test=True)
    return hidden, pooled


def build_dataset(config, train_data, test_data, id2word):

    def load_dataset(data, pad_size=256):
        contents = []
        data = data.astype(np.int64).tolist()
        for line in tqdm(data):
            for k in range(len(line)):
                line[k] = id2word[line[k]]
            for k in range(len(line)-1, -1, -1):
                if line[k] == '_END_':
                    del line[k]
            line_str = ''
            for word in line:
                line_str += (word)

            token = config.tokenizer.tokenize(line_str)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(0), seq_len, mask))
        return contents
    train = load_dataset(train_data[0:5000], pad_size=config.pad_size)
    test = load_dataset(test_data[0:1000], pad_size=config.pad_size)
    return train, test

if __name__ == '__main__':
    f = open('./data/word2idx_chinese.pickle', 'rb')
    idx2word = {}
    word2idx = pickle.load(f)
    for key, value in word2idx.items():
        idx2word[value] = key
    train_data = np.load('./data/train_data_c.npy')
    test_data = np.load('./data/test_data_c.npy')
    dataset = 'THUCNews'  # 数据集
    model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    train_data, test_data = build_dataset(config, train_data, test_data, idx2word)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    hidden, pooled = test(config, model, train_data)
    print(hidden)
    print(pooled)

