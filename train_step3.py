from Seq2seq import start_exp
import torch

def main():
    start_exp()

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    print(torch.cuda.is_available())
    main()