import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import time
import torch
# from pytorch_pretrained.optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle
import numpy as np
from Bert.train_eval import train, init_network
from importlib import import_module
import argparse
from tqdm import tqdm
from Bert.utils import build_iterator, get_time_dif, build_iterator_eval
from Bert.dataset import build_dataset, build_dataset_eval
from Bert.bleu_eval import count_common, count_hit, count_score, get_score
from Bert.Seq2seq_model import Seq2seq, DecoderRNN
import copy

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'

def build(hidden_size, batch_size, max_len, cuda):
    bidirectional = False

    model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config(batch_size, cuda)
    train_data = build_dataset(config)
    train_data = train_data[:-1]
    train_dataloader = build_iterator(train_data, config)
    val_data, test_data = build_dataset_eval(config)
    val_dataloader = build_iterator_eval(val_data, config)
    test_dataloader = build_iterator_eval(test_data, config)

    encoder = x.Model(config).to(config.device)
    decoder = DecoderRNN(len(config.tokenizer.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                         eos_id=config.tokenizer.convert_tokens_to_ids([SEP])[0],
                         sos_id=config.tokenizer.convert_tokens_to_ids([CLS])[0])
    seq2seq = Seq2seq(encoder, decoder)

    if cuda:
        seq2seq.cuda()
    param_optimizer = list(seq2seq.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(len(train_data))
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=config.learning_rate,
    #                      warmup=0.03,
    #                      t_total=len(train_data) * config.num_epochs)

    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.03 * len(train_data) * config.num_epochs),
                                                num_training_steps=len(train_data) * config.num_epochs)  # PyTorch scheduler

    Tensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if cuda:
        seq2seq.cuda()
    Tensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    loss_fun = torch.nn.NLLLoss(reduce=False)
    return seq2seq, optimizer, Tensor, train_dataloader, val_dataloader, test_dataloader, loss_fun, config, scheduler

def decode_sentence(symbols, config):
    sentences = []
    for symbol_sen in symbols:
        words = config.tokenizer.convert_ids_to_tokens(symbol_sen)
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

def remove_mask(srcs, results):
    results_new = []
    for src, tar in zip(srcs, results):
        src = src.split('[MASK]')
        tar = tar.split('[MASK]')
        sts = ''
        for i,u in enumerate(src):
            if i%2==1 and i<len(tar):
                sts += tar[i].strip()+' '
            else:
                sts += u.strip()+' '
        results_new.append(sts)
    return results_new

def eval_set(model, dataloader, config, epoch):
    model.eval()
    results = []
    references = []
    dics = []
    for i, (batch_src, train_tar, train_dic) in enumerate(tqdm(dataloader)):
        decoder_outputs, decoder_hidden, ret_dict = model(batch_src, batch_src[0], 0.0, False)
        symbols = ret_dict['sequence']
        symbols = torch.cat(symbols, 1).data.cpu().numpy()
        sentences = decode_sentence(symbols, config)
        results += sentences
        references += train_tar
        dics += train_dic

    with open('./outs/outs{}.pkl'.format(epoch), 'wb') as f:
        pickle.dump(results, f)

    tmp = []
    for u in results:
        u = u.replace('[MASK] ','')
        u = u.replace(' - ', '-').replace(' . ', '.').replace(' / ', '/')
        tmp.append(u)
    results = tmp


    with open('./result/tmp.out.txt', 'w', encoding='utf-8') as f:
        f.writelines([x.lower() + '\n' for x in results])
    bleu, hit, com, ascore = get_score()

    return results, bleu, hit, com


def valid(model, optimizer, Tensor, train_dataloader, val_dataloader, test_dataloader, loss_fun, config,scheduler):
    val_results, bleu, hit, com = eval_set(model, val_dataloader, config, epoch=0)
    print(val_results[0:5])
    print('BLEU:%f, HIT:%f, COMMON:%f' %(bleu, hit, com))

    return bleu, hit, com

def start_exp():
    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    seq2seq, optimizer, Tensor, train_dataloader, val_set, test_set, loss_fun, config ,scheduler= build(768, 1, 50, True)
    save_file_best = torch.load('./cache/best_save.data', map_location=torch.device('cuda:2'))

    seq2seq.load_state_dict(save_file_best['para'])
    bleu, hit, com = valid(seq2seq, optimizer, Tensor, train_dataloader, val_set, test_set, loss_fun, config,scheduler)


if __name__ == '__main__':
    start_exp()


