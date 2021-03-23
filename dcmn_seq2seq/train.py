import logging
import numpy as np
import torch
import os
import time
from tqdm import tqdm
from dcmn_seq2seq.utils import decode_sentence
from dcmn_seq2seq.bleu_eval_new import get_score

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
          gradient_accumulation_steps, warmup_proportion, learning_rate, loss_fun, dcmn_config,
          seq2seq, seq_config, seq_optimizer, seq_loss_fun, dg):
    model.train()
    seq2seq.train()

    tr_loss = 0
    pre_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len, key_embs = batch
        outputs = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len)

        loss = loss_fun(outputs, label_ids)
        seq_batch_input, sum_embs = dg.update_train(outputs, key_embs, dcmn_config=dcmn_config)
        if seq_batch_input is not None:
            src_ids, src_masks, tar_ids, tar_masks, indexes = seq_batch_input
            batch_src = [src_ids, 0, src_masks]
            batch_tar = [tar_ids, 0, tar_masks]
            decoder_outputs, decoder_hidden, ret_dict = seq2seq(batch_src, indexes, sum_embs, batch_tar[0], 0.5)
            seq_optimizer.zero_grad()
            target = batch_tar[0][:, 1:].reshape(-1)
            mask = batch_tar[2][:, 1:].reshape(-1).float()
            logit = torch.stack(decoder_outputs, 1).view(target.shape[0], -1)
            seq_loss = (seq_loss_fun(input=logit, target=target) * mask).sum() / mask.sum() + pre_loss*5
            pre_loss = 0
            seq_loss.backward()
            seq_optimizer.step()

        # if gradient_accumulation_steps > 1:
        #     loss = loss / gradient_accumulation_steps
        tr_loss += loss.item()
        pre_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1


        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate * warmup_linear(global_step / t_total, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    # logger.info("lr = %f", lr_this_step)
    lr_pre = lr_this_step

    return tr_loss, nb_tr_steps, global_step, lr_pre


def valid(model, dataloader, device, loss_fun, dg):
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
            logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)
            dg.update_test(logits.cpu().detach().numpy())
            tmp_eval_loss = loss_fun(logits, label_ids)

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


def eval_set(model, dataloader, config, test_sum_embs, test_index, test_qs):
    p = 0
    model.eval()
    results = []
    references = []
    dics = []
    sep_count = 0

    for i, (batch_src, train_tar, train_dic) in tqdm(enumerate(dataloader)):
        replace_embs = test_sum_embs[test_qs[i]: test_qs[i+1]]
        decoder_outputs, decoder_hidden, ret_dict = model(batch_src,test_index[i*config.test_batch_size: (i+1)*config.test_batch_size], replace_embs, None, 0.0)
        symbols = ret_dict['sequence']
        symbols = torch.cat(symbols, 1).data.cpu().numpy()
        sentences, sep = decode_sentence(batch_src, symbols, config)
        sep_count += sep
        results += sentences
        references += train_tar
        dics += train_dic

    # print("sep:%d" % (sep_count))
    sentences = results

    with open('./result/tmp.out.txt', 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in sentences])
    bleu, hit ,com, ascore = get_score()
    return sentences, bleu, hit, com, ascore


def train_valid(dcmn, dcmn_config, train_dataloader, eval_dataloader, dcmn_optimizer, dcmn_loss_fun,
                seq2seq, seq_config, seq_optimizer, seq_loss_fun, test_dataloader,dg):
    num_train_epochs = dcmn_config.num_train_epochs
    device = dcmn_config.device
    t_total = dcmn_config.t_total
    gradient_accumulation_steps = dcmn_config.gradient_accumulation_steps
    warmup_proportion = dcmn_config.warmup_proportion
    learning_rate = dcmn_config.learning_rate
    output_dir = dcmn_config.output_dir
    output_file = dcmn_config.output_file
    model_name = dcmn_config.model_name

    global_step = 0
    best_accuracy = 0
    save_file = {}
    max_bleu = 0

    for epoch in range(int(num_train_epochs)):
        logger.info("**** Epoch {} *****".format(epoch))

        tr_loss, nb_tr_steps, global_step, lr_pre = train(dcmn, train_dataloader, device, dcmn_optimizer, global_step,
                                                          t_total, gradient_accumulation_steps, warmup_proportion,
                                                          learning_rate, dcmn_loss_fun, dcmn_config,
                                                          seq2seq, seq_config, seq_optimizer, seq_loss_fun, dg)
        dg.restart_train()
        eval_loss, eval_accuracy = valid(dcmn, eval_dataloader, device, dcmn_loss_fun, dg)
        if dg.q_test_emb>0:
            dg.test_qs.append(dg.q_test_emb)
        if epoch >= 0:
            val_results, bleu, hit, com, ascore = eval_set(seq2seq, test_dataloader, seq_config, dg.test_sum_embs, dg.test_indexes, dg.test_qs)
            # validation steps

            print(val_results[0:5])
            print('BLEU:%f, HIT:%f, COMMON:%f, ASCORE:%f' % (bleu, hit, com, ascore))
            if bleu > max_bleu:
                max_bleu = bleu
                save_file['epoch'] = epoch + 1
                save_file['para'] = seq2seq.state_dict()
                save_file['best_bleu'] = bleu
                save_file['best_hit'] = hit
                save_file['best_common'] = com
                torch.save(save_file, './cache/best_save.data')
        if epoch < num_train_epochs-1:
            dg.restart_test()

        if eval_accuracy > best_accuracy:
            logger.info("**** Saving model.... *****")
            best_accuracy = eval_accuracy
            model_to_save = dcmn.module if hasattr(dcmn, 'module') else dcmn  # Only save the model it-self
            output_model_file = os.path.join(output_dir, model_name)
            torch.save(model_to_save.state_dict(), output_model_file)

        result = {'eval_loss': eval_loss,
                  'best_accuracy': best_accuracy,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'lr_now': lr_pre,
                  'loss': tr_loss / nb_tr_steps}
        output_eval_file = os.path.join(output_dir, output_file)
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (
                epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\t" % (key, str(result[key])))
            writer.write("\t\n")

    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val BLEU:%f, HIT:%f, COMMON:%f' % (
        save_file_best['best_bleu'], save_file_best['best_hit'], save_file_best['best_common']))
    seq2seq.load_state_dict(save_file_best['para'])
    test_results, bleu, hit, com, ascore = eval_set(seq2seq, test_dataloader, seq_config, dg.test_sum_embs, dg.test_indexes, dg.test_qs)
    print('Test BLEU:%f, HIT:%f, COMMON:%f, ASCORE:%f' % (bleu, hit, com, ascore))
    with open('./result/best_save_bert.out.txt', 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in test_results])


