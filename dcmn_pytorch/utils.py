import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import pickle
import argparse



def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)
    # mask = None
    if mask is None:
        result = softmax(vector, dim=-1)
    else:
        result = softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i]+1]
        doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
        ques_option_seq_output[i, :ques_len[i]+option_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 1]
        # print(option_len[i],len(sequence_output[i]),doc_len[i] , ques_len[i])
        option_seq_output[i, :option_len[i]] = sequence_output[i,
                                                 doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                   i] + 2]
        
    return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


def parse_mc(input_file, answer_file, max_pad_length):

    with open(input_file, 'rb') as f:
        sentences = pickle.load(f)
    with open(answer_file, 'rb') as f:
        labels = pickle.load(f)

    for i in range(len(sentences)):
        u = sentences[i]
        while len(u) < max_pad_length:
            u.append('[PAD] [PAD]')
        if len(u) > max_pad_length:
            # print(u)
            u = u[:max_pad_length]
        sentences[i] = tuple(u)
        if labels[i]>=max_pad_length-2:
            labels[i] = -1
        if labels[i]<0:
            labels[i] = -1
    _sentences = []
    _labels = []
    for i in range(len(sentences)):
        if len(sentences[i][0].split(' ')) + len(sentences[i][1].split(' '))+ len(sentences[i][2].split(' ')) <50\
                and labels[i]>=0:
            _sentences.append(sentences[i])
            _labels.append(labels[i])
    sentences = _sentences
    labels= _labels
    # sentences = sentences[:500]
    # labels = labels[:500]

    q_id = [i+1 for i in range(len(labels))]
    article = [u[0] for u in sentences]
    question = [u[1] for u in sentences]
    cts = []
    for i in range(max_pad_length-2):
        cts.append([u[i+2] for u in sentences])
    y = labels

    return article, question, cts, y, q_id


def get_arg_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        # default='data/openbookqa',
                        default='./data',
                        # default='data/race',
                        type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert-base-cased',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_dir",
                        default='mctest_output',
                        # default='race_output',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--output_file",
                        default='output_test.txt',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_file",
                        # default='train_dev.json',
                        # default='test_dev.json',
                        default='train_sentences.pkl',
                        type=str)
    parser.add_argument("--test_file",
                        # default='test.json',
                        # default='test_dev.json',
                        default='dev_sentences.pkl',
                        type=str)
    parser.add_argument("--max_seq_length",
                        default=128,
                        # default=128,
                        type=int)
    parser.add_argument("--train_batch_size",
                        default=2,
                        # default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        # default=12,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_choices",
                        default=14,
                        # default=12,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")


    parser.add_argument("--model_name",
                        default='output_test.bin',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument('--n_gpu',
                        type=int, default=1,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--gpu_id',
                        type=int, default=0,
                        help='gpu id')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")

    parser.add_argument('--loss_scale',
                        type=float, default=4,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    return parser