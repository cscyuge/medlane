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

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from dcmn_pytorch.model import BertForMultipleChoiceWithMatch

from dcmn_pytorch.utils import  get_arg_parser
from dcmn_pytorch.preprocess import get_dataloader
from dcmn_pytorch.train import train_valid

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
TASK_NAME = 'mctest' #'roc' 'coin' 'race' 'mctest'
NUM_CHOICE = 4


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    logger.info(args)
    output_eval_file = os.path.join(args.output_dir, args.output_file)

    if os.path.exists(output_eval_file) and args.output_file != 'output_test.txt':
        raise ValueError("Output file ({}) already exists and is not empty.".format(output_eval_file))
    with open(output_eval_file, "w") as writer:
        writer.write("***** Eval results Epoch  %s *****\t\n" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        writer.write("%s\t\n" % args)

    args.model_name = args.output_file[:-4] + ".bin"
    # print(args.model_name)

    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = args.n_gpu
    else:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
        n_gpu = args.n_gpu
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_dataloader, train_examples_size  = get_dataloader(data_dir=args.data_dir, data_file=args.train_file, num_choices=args.num_choices,
                                      tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=args.train_batch_size)
    num_train_steps = int(
        train_examples_size / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    t_total = num_train_steps

    eval_dataloader, eval_examples_size = get_dataloader(data_dir=args.data_dir, data_file=args.test_file,
                                                           num_choices=args.num_choices,
                                                           tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                                           batch_size=args.eval_batch_size)

    model = BertForMultipleChoiceWithMatch.from_pretrained(args.bert_model,
                                                           cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
                                                           num_choices=args.num_choices)

    model.to(device)

    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    train_valid(model, args.num_train_epochs, train_dataloader, eval_dataloader, device,
                optimizer, t_total, args.gradient_accumulation_steps, args.warmup_proportion, args.learning_rate,
                args.output_dir, args.output_file, args.model_name)

if __name__ == '__main__':
    main()