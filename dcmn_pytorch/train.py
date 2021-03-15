import logging
import numpy as np
import torch
import os
import time
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def train(model, dataloader, device, optimizer, global_step, t_total,
          gradient_accumulation_steps, warmup_proportion, learning_rate):
    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len = batch
        loss = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len, label_ids)

        # if gradient_accumulation_steps > 1:
        #     loss = loss / gradient_accumulation_steps
        # tr_loss += loss.item()
        # nb_tr_examples += input_ids.size(0)
        # nb_tr_steps += 1
        #
        # loss.backward()
        # if (step + 1) % gradient_accumulation_steps == 0:
        #     # modify learning rate with special warm up BERT uses
        #     lr_this_step = learning_rate * warmup_linear(global_step / t_total, warmup_proportion)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_this_step
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     global_step += 1

    # logger.info("lr = %f", lr_this_step)
    # lr_pre = lr_this_step
    lr_pre = 0

    return tr_loss, nb_tr_steps, global_step, lr_pre


def valid(model, dataloader, device):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len in tqdm(
            dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        all_doc_len = all_doc_len.to(device)
        all_ques_len = all_ques_len.to(device)
        all_option_len = all_option_len.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len,
                                  label_ids)
            logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy


def train_valid(model, num_train_epochs, train_dataloader, eval_dataloader, device,
                optimizer, t_total, gradient_accumulation_steps, warmup_proportion, learning_rate,
                output_dir, output_file, model_name):
    global_step = 0
    best_accuracy = 0

    for epoch in range(int(num_train_epochs)):
        logger.info("**** Epoch {} *****".format(epoch))

        tr_loss, nb_tr_steps, global_step, lr_pre = train(model, train_dataloader, device, optimizer, global_step,
                                                          t_total, gradient_accumulation_steps, warmup_proportion,
                                                          learning_rate)

        # eval_loss, eval_accuracy = valid(model, eval_dataloader, device)
        # if eval_accuracy > best_accuracy:
        #     logger.info("**** Saving model.... *****")
        #     best_accuracy = eval_accuracy
        #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #     output_model_file = os.path.join(output_dir, model_name)
        #     torch.save(model_to_save.state_dict(), output_model_file)
        #
        # result = {'eval_loss': eval_loss,
        #           'best_accuracy': best_accuracy,
        #           'eval_accuracy': eval_accuracy,
        #           'global_step': global_step,
        #           'lr_now': lr_pre,
        #           'loss': tr_loss / nb_tr_steps}
        # output_eval_file = os.path.join(output_dir, output_file)
        # with open(output_eval_file, "a") as writer:
        #     logger.info("***** Eval results *****")
        #     writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (
        #         epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\t" % (key, str(result[key])))
        #     writer.write("\t\n")
