import os
import time
import torch
from transformers import BertTokenizer


class DCMN_Config():

    def __init__(self):
        # The input data dir. Should contain the .csv files (or other data files) for the task.
        self.data_dir = './data'

        # Bert pre-trained model selected in the list: bert-base-uncased,
        # bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.
        self.bert_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

        # Set this flag if you are using an uncased model.
        self.do_lower_case = False

        # The output directory where the model checkpoints will be written.
        self.output_dir = 'mctest_output'
        self.output_file = 'output_test.txt'

        self.train_file = 'train_sentences.pkl'
        self.test_file = 'dev_sentences.pkl'

        self.max_seq_length = 64
        self.train_batch_size = 4
        self.eval_batch_size = 4
        self.seq_batch_size = 4
        self.num_choices = 8
        self.learning_rate = 1e-5/2.0
        self.num_train_epochs = 30.0
        self.model_name = 'output_test.bin'
        self.n_gpu = 1
        self.gpu_id = 0

        # Number of updates steps to accumulate before performing a backward/update pass.
        self.gradient_accumulation_steps = 1

        # Proportion of training to perform linear learning rate warmup for.
        # E.g., 0.1 = 10%% of training.
        self.warmup_proportion = 0.1

        # Whether not to use CUDA when available
        self.no_cuda = False

        # random seed for initialization
        self.seed = 42

        # Whether to perform optimization and keep the optimizer averages on CPU
        self.optimize_on_cpu = False

        # Loss scaling, positive power of 2 values can improve fp16 convergence.
        self.loss_scale = 4

        output_eval_file = os.path.join(self.output_dir, self.output_file)

        if os.path.exists(output_eval_file) and self.output_file != 'output_test.txt':
            raise ValueError("Output file ({}) already exists and is not empty.".format(output_eval_file))
        with open(output_eval_file, "w") as writer:
            writer.write("***** Eval results Epoch  %s *****\t\n" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            dic = str([(name, value) for name,value in vars(self).items()])
            writer.write("%s\t\n" % dic)

        if self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        else:
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device("cuda", self.gpu_id)

        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.gradient_accumulation_steps))

        self.train_batch_size = int(self.train_batch_size / self.gradient_accumulation_steps)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        self.t_total = 0
        self.hidden_size = 768



class Seq2seq_Config():

    def __init__(self):
        pass



if __name__ == '__main__':
    config = DCMN_Config()
