import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from Seq2seq23 import start_exp
import torch

def main():
    start_exp()

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    print(torch.cuda.is_available())
    main()