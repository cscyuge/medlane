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
from Bert.bleu_eval import count_common, count_hit, count_score
import copy
PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device



class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.hidden_change = nn.Linear(768, self.hidden_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                          function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        encoder_hidden = torch.tanh(self.hidden_change(encoder_hidden)).unsqueeze(dim=0)
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(0)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(0)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_src, batch_tar=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(batch_src)
        # print(encoder_outputs,encoder_hidden)
        target_variable = batch_tar
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result


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


def eval_set(model, dataloader, config):
    model.eval()
    results = []
    references = []
    dics = []
    tot_loss = 0
    for i, (batch_src, train_tar, train_dic) in enumerate(dataloader):
        decoder_outputs, decoder_hidden, ret_dict = model(batch_src, None, 0.0)
        symbols = ret_dict['sequence']
        symbols = torch.cat(symbols, 1).data.cpu().numpy()
        sentences = decode_sentence(symbols, config)
        results += sentences
        references += train_tar
        dics += train_dic

    tmp = copy.deepcopy(references)
    bleu = count_score(results, tmp)
    del tmp
    hit = count_hit(results, dics)
    com = count_common(results)
    sentences = []
    for words in results:
        tmp = ''
        for word in words:
            tmp += word
            tmp += ' '
        sentences.append(tmp)
    model.train()
    return sentences, bleu, hit, com


def train(model, optimizer, Tensor, train_dataloader, val_dataloader, test_dataloader, loss_fun, config,scheduler):
    model.train()
    #training steps
    max_bleu = -99999
    save_file = {}
    for e in range(config.num_epochs):
        tot_loss = 0
        tr_step = 0
        for i, (batch_src, batch_tar) in enumerate(tqdm(train_dataloader)):
            decoder_outputs, decoder_hidden, ret_dict = model(batch_src, batch_tar[0], 0.5)
            optimizer.zero_grad()
            target = batch_tar[0][:, 1:].reshape(-1)
            mask = batch_tar[2][:, 1:].reshape(-1).float()
            logit = torch.stack(decoder_outputs, 1).view(target.shape[0], -1)
            loss = (loss_fun(input=logit, target=target) * mask).sum() / mask.sum()
            tot_loss += loss.item()
            tr_step += 1
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 50 == 0:
                print('train loss:%f' %loss.item())
        #validation steps
        if e >= 0:
            val_results, bleu, hit, com = eval_set(model, val_dataloader, config)
            print(val_results[0:5])
            # print('BLEU:%f, HIT:%f, COMMON:%f' %(bleu, hit, com))
            if bleu > max_bleu:
                max_bleu = bleu
                save_file['epoch'] = e + 1
                save_file['para'] = model.state_dict()
                save_file['best_bleu'] = bleu
                save_file['best_hit'] = hit
                save_file['best_common'] = com
                torch.save(save_file, './cache/best_save.data')
            if bleu < max_bleu - 0.6:
                print('Early Stop')
                break
            print(save_file['epoch'] - 1)

            result = {'tr_loss': tot_loss/tr_step,
                      'BLUE': bleu,
                      'HIT': hit,
                      'COMMON': com,
                      }
            print(result)
            with open('./result/results.txt', "a") as writer:
                writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (
                    e, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                for key in sorted(result.keys()):
                    writer.write("%s = %s\t" % (key, str(result[key])))
                writer.write("\t\n")


    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val BLEU:%f, HIT:%f, COMMON:%f' %(save_file_best['best_bleu'], save_file_best['best_hit'], save_file_best['best_common']))
    model.load_state_dict(save_file_best['para'])
    test_results, bleu, hit, com = eval_set(model, test_dataloader, config)
    print('Test BLEU:%f, HIT:%f, COMMON:%f' % (bleu, hit, com))
    with open('./result/best_save_bert.out.txt', 'w') as f:
        f.writelines([x + '\n' for x in test_results])
    return bleu, hit, com
def start_exp():
    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    ans = []
    for t in range(1):
        seq2seq, optimizer, Tensor, train_dataloader, val_set, test_set, loss_fun, config ,scheduler= build(768, 16, 50, True)
        bleu, hit, com = train(seq2seq, optimizer, Tensor, train_dataloader, val_set, test_set, loss_fun, config,scheduler)
        ans.append([bleu, hit, com])
    ans = np.array(ans)
    rs_name = './result/bert_%d_Att.txt' % 786
    with open(rs_name, 'w') as f:
        f.write(str(ans))
        f.write(str(ans.mean(0))+'\n')
        f.write(str(ans.std(0))+'\n')

if __name__ == '__main__':
    start_exp()


