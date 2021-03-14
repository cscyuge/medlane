# !/usr/bin/env python3
# helpers for training the model
import mxnet as mx
from mxnet import autograd, nd
import logging, time
import numpy as np

from tqdm import tqdm

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.Logger(__name__)
logger.setLevel(logging.WARNING)


def calculate_loss(inputs, labels, model, loss_func, loss_name='sce', class_weight=None):
    '''
    this function will calculate loss, we use softmax cross entropy for now,
    possibly extend to weighted version
    '''
    preds = model(inputs)
    if loss_name == 'sce':
        l = loss_func(preds, labels)
    else:
        logger.error('Loss function %s not implemented' % loss_name)
        raise NotImplementedError
    return preds, l

train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []


def one_epoch(dataloader, model, loss_func, trainer, ctx, is_train, epoch,
              class_weight=None, loss_name='sce'):
    '''
    this function trains model for one epoch if `is_train` is True
    also calculates loss/metrics whether in training or dev
    '''
    loss_val = 0.
    metric = mx.metric.Accuracy()
    preds = []
    for n_batch, iters in tqdm(enumerate(dataloader)):
        *inputs, labels = [item.as_in_context(ctx) for item in iters]

        if is_train:
            with autograd.record():
                batch_pred, l = calculate_loss(inputs, labels, model, loss_func, loss_name, class_weight)

            # backward calculate
            l.backward()

            # update parmas
            trainer.step(labels.shape[0])

        else:
            batch_pred, l = calculate_loss(inputs, labels, model, loss_func, loss_name, class_weight)

        # keep result for metric
        batch_pred = nd.softmax(batch_pred, axis=1)
        preds.extend(batch_pred)
        batch_true = labels.reshape(-1)
        metric.update(preds=batch_pred, labels=batch_true)

        batch_loss = l.mean().asscalar()
        loss_val += batch_loss

    # metric
    loss_val /= n_batch + 1

    if is_train:
        acc = metric.get()
        train_accuracy.append(acc[1])
        print(acc)
        train_loss.append(loss_val)
        print('epoch %d, learning_rate %.5f \n\t train_loss %.4f, %s: %.4f' %
             (epoch, trainer.learning_rate, loss_val, *acc))
        # train_curve.append((acc, F1))
        # declay lr
        if epoch % 2 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.9)
    else:
        acc = metric.get()
        print(acc)
        val_accuracy.append(acc[1])
        val_loss.append(loss_val)
        print('\t valid_loss %.4f, %s: %.4f' % (loss_val, *acc))
        # valid_curve.append((acc, F1))
   
    return preds, loss_val

def train_valid(dataloader_train, dataloader_test, model, loss_func, trainer, \
                num_epoch, ctx, class_weight=None, loss_name='sce'):
    '''
    wrapper for training and "test" the model
    '''
    for epoch in range(1, num_epoch+1):
        start = time.time()
        # train
        is_train = True
        one_epoch(dataloader_train, model, loss_func, trainer, ctx, is_train, \
                  epoch, class_weight, loss_name)

        # valid
        is_train = False
        one_epoch(dataloader_test, model, loss_func, trainer, ctx, is_train, \
                  epoch, class_weight, loss_name)
        end = time.time()
        print('time %.2f sec' % (end-start))
        print("*"*100)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss():
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(train_loss, 'b-o', label="training loss")
    plt.plot(val_loss, 'r-o', label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_acc():
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(train_accuracy, 'b-o', label="training accuracy")
    plt.plot(val_accuracy, 'r-o', label="validation accuracy")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()