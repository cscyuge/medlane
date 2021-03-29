import logging
import numpy as np
import torch
import os
import time
from tqdm import tqdm
from torch.nn.functional import softmax

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


def train(model, dataloader, device, optimizer, dcmn_scheduler, global_step,
          gradient_accumulation_steps, loss_fun, dcmn_config,
          seq2seq, seq_config, seq_optimizer, seq_scheduler, seq_loss_fun):
    model.train()
    seq2seq.train()

    tr_dcmn_loss = 0
    tr_seq_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len, key_embs, \
        src_ids, src_masks, indices, tar_ids, tar_masks = batch

        outputs = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len)
        loss = loss_fun(outputs, label_ids)

        batch_scores = softmax(outputs, dim=1)
        batch_scores = batch_scores.unsqueeze(-1).expand(dcmn_config.train_batch_size, dcmn_config.num_choices,
                                                         seq_config.hidden_size)
        sum_embs = torch.sum(key_embs * batch_scores, dim=1)

        decoder_outputs, decoder_hidden, ret_dict = seq2seq([src_ids,src_masks], indices, sum_embs, tar_ids, 0.5)
        target = tar_ids[:, 1:].reshape(-1)
        mask = tar_masks[:, 1:].reshape(-1).float()
        logit = torch.stack(decoder_outputs, 1).view(target.shape[0], -1)

        seq_loss = (seq_loss_fun(input=logit, target=target) * mask).sum() / mask.sum()
        tr_dcmn_loss += loss.item()
        tr_seq_loss += seq_loss.item()

        if step % 500 == 0:
            print('train dcmn loss:{:.6f}, train seq2seq loss:{:.6f}'.format(loss.item(), seq_loss.item()))

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        total_loss = seq_loss + loss
        total_loss.backward()

        seq_optimizer.step()
        seq_scheduler.step()
        seq_optimizer.zero_grad()

        # modify learning rate with special warm up BERT uses
        # lr_this_step = learning_rate * warmup_linear(global_step / t_total, warmup_proportion)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_this_step
        optimizer.step()
        dcmn_scheduler.step()
        optimizer.zero_grad()
        global_step += 1


    # logger.info("lr = %f", lr_this_step)

    return tr_dcmn_loss, tr_seq_loss, nb_tr_steps, global_step, dcmn_scheduler.get_last_lr(), seq_scheduler.get_last_lr()


def valid(dcmn, dataloader, device, loss_fun, dcmn_config,
          seq2seq, seq_config):
    dcmn.eval()
    seq2seq.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    outs = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len, \
        key_embs, src_ids, src_masks, indices = batch

        with torch.no_grad():
            logits = dcmn(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)
            tmp_eval_loss = loss_fun(logits, label_ids)
            batch_scores = softmax(logits, dim=1)
            batch_scores = batch_scores.unsqueeze(-1).expand(dcmn_config.eval_batch_size, dcmn_config.num_choices,
                                                             seq_config.hidden_size)
            sum_embs = torch.sum(key_embs * batch_scores, dim=1)
            batch_src = [src_ids, src_masks]
            decoder_outputs, decoder_hidden, ret_dict = seq2seq(batch_src, indices, sum_embs, None, 0.0)
            symbols = ret_dict['sequence']
            symbols = torch.cat(symbols, 1).data.cpu().numpy()
            for u in symbols:
                outs.append(u)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy, outs


def train_valid(dcmn, dcmn_config, train_dataloader, eval_dataloader,
                dcmn_optimizer, dcmn_scheduler, dcmn_loss_fun,
                seq2seq, seq_config, seq_optimizer, seq_scheduler, seq_loss_fun, dg):
    num_train_epochs = dcmn_config.num_train_epochs
    device = dcmn_config.device
    gradient_accumulation_steps = dcmn_config.gradient_accumulation_steps
    output_dir = dcmn_config.output_dir
    output_file = dcmn_config.output_file
    model_name = dcmn_config.model_name

    global_step = 0
    best_accuracy = 0
    save_file = {}
    best_bleu = 0

    for epoch in range(int(num_train_epochs)):
        logger.info("**** Epoch {} *****".format(epoch))

        tr_dcmn_loss, tr_seq_loss, nb_tr_steps, global_step, lr_pre, seq_lr_pre = train(dcmn, train_dataloader, device, dcmn_optimizer,
                                                                           dcmn_scheduler, global_step,
                                                                           gradient_accumulation_steps, dcmn_loss_fun, dcmn_config,
                                                                           seq2seq, seq_config, seq_optimizer, seq_scheduler,
                                                                           seq_loss_fun)

        eval_loss, eval_accuracy, outs = valid(dcmn, eval_dataloader, device, dcmn_loss_fun, dcmn_config,
                                               seq2seq, seq_config)

        import pickle
        with open('./outs/outs{}.pkl'.format(epoch), 'wb') as f:
            pickle.dump(outs, f)
        val_results, bleu, hit, com, ascore = dg.valid(outs)


        if eval_accuracy > best_accuracy:
            logger.info("**** Saving best dcmn model.... *****")
            best_accuracy = eval_accuracy
            model_to_save = dcmn.module if hasattr(dcmn, 'module') else dcmn  # Only save the model it-self
            output_model_file = os.path.join(output_dir, model_name)
            torch.save(model_to_save.state_dict(), output_model_file)

        if bleu > best_bleu:
            logger.info("**** Saving best dcmn+seq2seq model.... *****")
            best_bleu = bleu
            model_to_save = dcmn.module if hasattr(dcmn, 'module') else dcmn  # Only save the model it-self
            save_file['epoch'] = epoch + 1
            save_file['seq_para'] = seq2seq.state_dict()
            save_file['dcmn_para'] = model_to_save.state_dict()
            save_file['best_bleu'] = bleu
            save_file['best_hit'] = hit
            save_file['best_common'] = com
            save_file['best_ascore'] = ascore
            torch.save(save_file, './cache/best_save.data')
            with open('./result/best_save_bert.out.txt', 'w', encoding='utf-8') as f:
                f.writelines([x.lower() + '\n' for x in val_results])

        result = {'eval_loss': eval_loss,
                  'best_accuracy': best_accuracy,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'dcmn_lr_now': lr_pre,
                  'seq_lr_now':seq_lr_pre,
                  'tr_dcmn_loss': tr_dcmn_loss / nb_tr_steps,
                  'tr_seq_loss' : tr_seq_loss / nb_tr_steps,
                  'BLUE': bleu,
                  'HIT': hit,
                  'COMMON': com,
                  'ASCORE': ascore}

        output_eval_file = os.path.join(output_dir, output_file)
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (
                epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\t" % (key, str(result[key])))
            writer.write("\t\n")

