# !/usr/bin/env python3
import logging
import mxnet as mx
from mxnet import nd, gluon
import multiprocessing as mp
import gluonnlp as nlp
import numpy as np
from collections import defaultdict
from bert_embedding import BertEmbedding
import pickle


logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.Logger(__name__)
logger.setLevel(logging.WARNING)

def to_dataset(samples, labels, ctx=mx.gpu(), batch_size=64, max_seq_length=25):
    '''
    this function will use BertEmbedding to get each fields' embeddings
    and load the given labels, put them together into a dataset
    '''
    bertembedding = BertEmbedding(ctx=mx.gpu(), batch_size=batch_size, max_seq_length=max_seq_length)
    logger.info('Construct bert embedding for sentences')
    embs = []
    from tqdm import tqdm
    for sample in tqdm(samples):
        tokens_embs = bertembedding.embedding(sample)
        embs.append([np.asarray(token_emb[1]) for token_emb in tokens_embs])

    if labels:
        dataset = [[*obs_hyp, label] for obs_hyp, label in zip(embs, labels)]
    else:
        dataset = embs
    return dataset

def get_length(dataset):
    '''
    lengths used for batch sampler, we will use the first field of each row
    for now, i.e., obs1
    '''
    return [row[0].shape[0] for row in dataset]


def to_dataloader(dataset, batch_size=64, num_buckets=10, bucket_ratio=.5):
    '''
    this function will sample the dataset to dataloader
    '''
    pads = [nlp.data.batchify.Pad(axis=0, pad_val=0) for _ in range(len(dataset[0])-1)]

    batchify_fn = nlp.data.batchify.Tuple(
        *pads,                      # for observations and hypotheses
        nlp.data.batchify.Stack()   # for labels
    )

    lengths = get_length(dataset)
    print('Build batch_sampler')
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        lengths=lengths, batch_size=batch_size, num_buckets=num_buckets,
        ratio=bucket_ratio, shuffle=True
    )
    print(batch_sampler.stats())

    dataloader = gluon.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn,
        # num_workers=_get_threads()
    )
    
    return dataloader


def get_dataloader(sts, labels=None, keys=['obs1', 'obs2', 'hyp1', 'hyp2'], \
                   batch_size=64, num_buckets=10, bucket_ratio=.5, \
                   ctx=mx.gpu(), max_seq_length=25, max_pad_length=4, sample_num=None, dataset_load_path=None,dataset_save_path=None):
    '''
    this function will use the helpers above, take sentence file path,
    label file path, and batch_size, num_buckets, bucket_ratio, to
    get the dataloader for model to us. sample_num controls how many
    samples in dataset the model will use, defualt to None, e.g., use all
    '''
    if dataset_load_path:
        with open(dataset_load_path,'rb') as f:
            dataset = pickle.load(f)
        dataset = dataset[:sample_num]
        dataloader = to_dataloader(dataset=dataset, batch_size=batch_size, \
                                   num_buckets=num_buckets, bucket_ratio=bucket_ratio)

    elif labels:
        with open(sts,'rb') as f:
            sentences = pickle.load(f)
        with open(labels,'rb') as f:
            labels = pickle.load(f)

        sentences = sentences[:sample_num]
        labels = labels[:sample_num]

        try:
            assert(len(sentences)==len(labels))
        except:
            logger.error('Sample sentence length does not equal to label\'s length!')
            exit(-1)

        for i in range(len(sentences)):
            u = sentences[i]
            while len(u) < max_pad_length:
                u.append('[PAD] [PAD]')
            if len(u) > max_pad_length:
#                 print(u)
                u = u[:max_pad_length]
            sentences[i] = tuple(u)

        dataset = to_dataset(sentences, labels, ctx=ctx, batch_size=batch_size, \
                             max_seq_length=max_seq_length)

        with open(dataset_save_path,'wb') as f:
            pickle.dump(dataset, f)

        dataloader = to_dataloader(dataset=dataset, batch_size=batch_size, \
                                   num_buckets=num_buckets, bucket_ratio=bucket_ratio)
    else:
        dataset = to_dataset(sts, labels, ctx=ctx, batch_size=batch_size, \
                             max_seq_length=max_seq_length)
        dataloader = []
        for sample in dataset:
            batch = []
            for emb in sample:
                batch.append(nd.array(emb.reshape(1, *emb.shape)))
            dataloader.append(batch)

    return dataloader
