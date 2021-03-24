from dcmn_seq2seq.utils import parse_mc, select_field
import logging
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""

    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 endings,
                 key_embs,
                 src_ids,
                 src_masks,
                 indices,
                 tar_ids=None,
                 tar_masks=None,
                 tars=None,
                 cudic=None,
                 label=None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = endings
        self.key_embs = key_embs
        self.src_ids = src_ids
        self.src_masks = src_masks
        self.indices = indices
        self.tar_ids = tar_ids
        self.tar_masks = tar_masks
        self.tars = tars
        self.cudic = cudic
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"endings: {self.endings}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label,
                 key_embs,
                 src_ids,
                 src_masks,
                 indices,
                 tar_ids=None,
                 tar_masks=None,
                 tars=None,
                 cudic=None
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len,
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label
        self.key_embs = key_embs
        self.src_ids = src_ids
        self.src_masks = src_masks
        self.indices = indices
        self.tar_ids = tar_ids
        self.tar_masks = tar_masks
        self.tars = tars
        self.cudic = cudic
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    pop_label = True
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(1)
        else:
            tokens_b.pop(1)


def read_swag_examples(input_file, max_pad_length, dg):
    answer_file = input_file.replace('sentences', 'labels')
    article, question, cts, key_embs, y, q_id, \
        src_ids, src_masks, indices, tar_ids, tar_masks, tars, cudics  = parse_mc(input_file, answer_file, max_pad_length, dg)

    examples = [
        SwagExample(
            swag_id=s6,
            context_sentence=s1,
            start_ending=s2,  # in the swag dataset, the
            # common beginning of each
            # choice is stored in "sent2".
            endings=s3,
            key_embs=s4,
            label=s5,
            src_ids=s7,
            src_masks=s8,
            indices=s9,
            tar_ids=s10,
            tar_masks=s11,
            tars=s12,
            cudic=s13,
        ) for i, (s1, s2, *s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13) in
                enumerate(zip(article, question, *cts, key_embs, y, q_id, src_ids, src_masks, indices, tar_ids, tar_masks, tars, cudics))
    ]

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, dg):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    ool = 0
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]  # + start_ending_tokens

            ending_token = tokenizer.tokenize(ending)
            option_len = len(ending_token)
            ques_len = len(start_ending_tokens)

            ending_tokens = start_ending_tokens + ending_token

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            doc_len = len(context_tokens_choice)
            if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                ques_len = len(ending_tokens) - option_len
            if ques_len<=0:
                print(len(ending_tokens), option_len)
                print(example)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert (doc_len + ques_len + option_len) <= max_seq_length
            if (doc_len + ques_len + option_len) > max_seq_length:
                print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                assert (doc_len + ques_len + option_len) <= max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))

        label = example.label
        if example_index < 0:
            logger.info("*** Example ***")
            logger.info(f"swag_id: {example.swag_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id=example.swag_id,
                choices_features=choices_features,
                label=label,
                key_embs=example.key_embs,
                src_ids=example.src_ids,
                src_masks=example.src_masks,
                indices=example.indices,
                tar_ids=example.tar_ids,
                tar_masks=example.tar_masks,
                tars=example.tars,
                cudic=example.cudic
            )
        )

    return features


def get_dataloader(data_dir, data_file, num_choices, tokenizer, max_seq_length, batch_size, dg, is_training):
    examples = read_swag_examples(os.path.join(data_dir, data_file), max_pad_length=num_choices + 2, dg=dg)

    features = convert_examples_to_features(
        examples, tokenizer, max_seq_length, True, dg)
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_doc_len = torch.tensor(select_field(features, 'doc_len'), dtype=torch.long)
    all_ques_len = torch.tensor(select_field(features, 'ques_len'), dtype=torch.long)
    all_option_len = torch.tensor(select_field(features, 'option_len'), dtype=torch.long)
    all_key_embs = torch.tensor([f.key_embs for f in features], dtype=torch.float)
    # all_key_embs = all_key_embs.squeeze()
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    all_indices = torch.tensor([f.indices for f in features], dtype=torch.float)
    all_src_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
    all_src_masks = torch.tensor([f.src_masks for f in features], dtype=torch.long)

    if is_training:
        all_tar_ids = torch.tensor([f.tar_ids for f in features], dtype=torch.long)
        all_tar_masks = torch.tensor([f.tar_masks for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                             all_option_len, all_key_embs, all_src_ids, all_src_masks, all_indices, all_tar_ids,
                             all_tar_masks)

        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader, len(examples)
    else:
        all_tars = [f.tars for f in features]
        all_cudic = [f.cudic for f in features]
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                                   all_option_len, all_key_embs, all_src_ids, all_src_masks, all_indices)

        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader, len(examples), all_tars, all_cudic
