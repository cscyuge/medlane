# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time
import logging
import os
import random
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from dcmn_seq2seq.dcmn import BertForMultipleChoiceWithMatch
from dcmn_seq2seq.preprocess import get_dataloader
from dcmn_seq2seq.train import train_valid
from dcmn_seq2seq.config import DCMN_Config

from dcmn_seq2seq.Seq2seq import DecoderRNN, Seq2seq, SEP, CLS
from dcmn_seq2seq.draw import DataGenerator
import dcmn_seq2seq.models.bert as seq_bert
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '3'


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dcmn():
    dcmn_config = DCMN_Config()

    output_eval_file = os.path.join(dcmn_config.output_dir, dcmn_config.output_file)

    if os.path.exists(output_eval_file) and dcmn_config.output_file != 'output_test.txt':
        raise ValueError("Output file ({}) already exists and is not empty.".format(output_eval_file))
    with open(output_eval_file, "w") as writer:
        writer.write("***** Eval results Epoch  %s *****\t\n" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        dic = str([(name, value) for name, value in vars(dcmn_config).items()])
        writer.write("%s\t\n" % dic)


    random.seed(dcmn_config.seed)
    np.random.seed(dcmn_config.seed)
    torch.manual_seed(dcmn_config.seed)
    if dcmn_config.n_gpu > 0:
        torch.cuda.manual_seed_all(dcmn_config.seed)

    tokenizer = dcmn_config.tokenizer
    seq_config = seq_bert.Config(dcmn_config.seq_batch_size, dcmn_config.no_cuda)
    dg = DataGenerator(train_batch_size=dcmn_config.train_batch_size,
                       eval_batch_size=dcmn_config.eval_batch_size,
                       max_pad_length=dcmn_config.num_choices+2,
                       max_seq_length = dcmn_config.max_seq_length, cuda=not dcmn_config.no_cuda,
                       tokenizer = dcmn_config.tokenizer, seq_tokenizer=seq_config.tokenizer)


    train_dataloader, train_examples_size = get_dataloader(data_dir=dcmn_config.data_dir, data_file=dcmn_config.train_file,
                                                           num_choices=dcmn_config.num_choices,
                                                           tokenizer=tokenizer, max_seq_length=dcmn_config.max_seq_length,
                                                           batch_size=dcmn_config.train_batch_size,
                                                           dg=dg, is_training=True)
    num_train_steps = int(
        train_examples_size / dcmn_config.train_batch_size / dcmn_config.gradient_accumulation_steps * dcmn_config.num_train_epochs)
    t_total = num_train_steps
    dcmn_config.t_total = t_total

    eval_dataloader, eval_examples_size = get_dataloader(data_dir=dcmn_config.data_dir, data_file=dcmn_config.test_file,
                                                         num_choices=dcmn_config.num_choices,
                                                         tokenizer=tokenizer, max_seq_length=dcmn_config.max_seq_length,
                                                         batch_size=dcmn_config.eval_batch_size,
                                                         dg=dg, is_training=False)


    model = BertForMultipleChoiceWithMatch.from_pretrained(dcmn_config.bert_model, num_choices=dcmn_config.num_choices)
    model.to(dcmn_config.device)

    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]


    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=dcmn_config.learning_rate,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(dcmn_config.warmup_proportion*dcmn_config.t_total),
                                                num_training_steps=dcmn_config.t_total)  # PyTorch scheduler

    loss_fun = torch.nn.CrossEntropyLoss()

    return model, dcmn_config, train_dataloader, eval_dataloader, \
           optimizer, scheduler, loss_fun, dg , seq_config


def build_seq2seq(config, hidden_size, max_len, no_cuda, dg):
    bidirectional = False

    config.hidden_size = hidden_size

    encoder = seq_bert.Model(config).to(config.device)
    decoder = DecoderRNN(len(config.tokenizer.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                         eos_id=config.tokenizer.convert_tokens_to_ids([SEP])[0],
                         sos_id=config.tokenizer.convert_tokens_to_ids([CLS])[0])

    decoder = decoder.to(config.device)

    seq2seq = Seq2seq(encoder, decoder)
    if not no_cuda:
        seq2seq.cuda()
    param_optimizer = list(seq2seq.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    0.03 * len(dg.train_src_txt) * config.num_epochs),
                                                num_training_steps=len(dg.train_src_txt) * config.num_epochs)  # PyTorch scheduler

    if not no_cuda:
        seq2seq.cuda()
    loss_fun = torch.nn.NLLLoss(reduce=False)
    return seq2seq, optimizer, scheduler, loss_fun

def main():
    dcmn, dcmn_config, train_dataloader, eval_dataloader, \
        dcmn_optimizer, dcmn_scheduler, dcmn_loss_fun, dg, seq_config = build_dcmn()

    seq2seq, seq_optimizer, seq_scheduler, seq_loss_fun = build_seq2seq(seq_config, 768, dcmn_config.max_seq_length, dcmn_config.no_cuda, dg)
    train_valid(dcmn, dcmn_config, train_dataloader, eval_dataloader, dcmn_optimizer, dcmn_scheduler, dcmn_loss_fun,
                seq2seq,seq_config, seq_optimizer, seq_scheduler, seq_loss_fun, dg)



if __name__ == '__main__':
    main()