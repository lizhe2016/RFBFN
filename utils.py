# -*- coding : utf-8 -*-
import logging
import random

import torch
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def get_batch(samples, batch_size, is_single_batch):
    list_samples_batches = []
    if not is_single_batch:
        for i in range(0, len(samples), batch_size):
            list_samples_batches.append(samples[i: i + batch_size])
    else:
        to_single_batch = []
        for i in range(0, len(samples)):
            if len(samples[i]['id']) > 350:
                to_single_batch.append(i)

        for i in to_single_batch:
            list_samples_batches.append([samples[i]])
        samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

        for i in range(0, len(samples), batch_size):
            list_samples_batches.append(samples[i: i + batch_size])
    return list_samples_batches


def save_model(model, model_save_path, epoch):
    save_model_path = os.path.join(model_save_path, "epoch_" + str(epoch) + "_save_all_model.pkl")
    logger.info('Saving model to %s...' % save_model_path)
    torch.save(model.state_dict(), save_model_path)
    # model = Model()
    # model.load_state_dict(torch.load('\parameter.pkl'))

def load_state_dict_for_model(model, path):
    state_dict = torch.load(path)
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_decay(optimizer, epoch, decay_rate):
    # lr = init_lr * ((1 - decay_rate) ** epoch)
    if epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (1 - decay_rate)
            # print(param_group['lr'])
    return optimizer


def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
