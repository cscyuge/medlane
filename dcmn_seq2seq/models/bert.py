# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


class Config(object):

    """配置参数"""
    def __init__(self, batch_size, no_cuda=False):
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')   # 设备
        self.require_improvement = 10000000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 30                                            # epoch数
        self.batch_size = batch_size                                       # mini-batch大小
        # self.pad_size = 64                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 8.264744570615555e-06

        # self.learning_rate = 5e-5 / 5.0                                       # 学习率
        self.bert_path = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        # self.bert_path = 'bert-base-cased'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

        # self.warmup_proportion = 0.03
        self.warmup_proportion = 0.0


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        bert_output = self.bert(context, attention_mask=mask)
        state = bert_output[0]
        pooled = bert_output[1]
        return state, pooled
