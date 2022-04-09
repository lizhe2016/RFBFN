# -*- coding : utf-8 -*-
import torch
import gc
import torch.autograd as autograd
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from RFBFN_model import RFBFN_Model
from RFBFN_evaluate import eval_for_RFBFN
from RFBFN_config import ArgumentParser
import pickle
from utils import get_batch, save_model, set_seed, lr_decay

import os
import sys
import logging
from tqdm import tqdm

config_for_model = ArgumentParser()

if config_for_model.gpu_setting != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = config_for_model.gpu_setting

logging.basicConfig(filename=os.path.join(config_for_model.log_path, "info.log"),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


if __name__ == '__main__':
    set_seed(config_for_model.seed)

    logger.addHandler(logging.FileHandler(os.path.join(config_for_model.log_path, "train.log"), 'w'))
    logger.info(sys.argv)
    logger.info(config_for_model)

    model = RFBFN_Model(config_for_model)

    train_data = pickle.load(open(config_for_model.save_path + config_for_model.task + "_all_" + "train" + "_samples.pkl",
                                  "rb"))
    train_iter = get_batch(train_data, config_for_model.batch_size, config_for_model.is_single_batch)

    test_data = pickle.load(open(config_for_model.save_path + config_for_model.task + "_all_" + "test" + "_samples.pkl",
                                 "rb"))
    test_iter = get_batch(test_data, config_for_model.batch_size, config_for_model.is_single_batch)

    dev_data = pickle.load(open(config_for_model.save_path + config_for_model.task + "_all_" + "dev" + "_samples.pkl",
                                "rb"))

    dev_iter = get_batch(dev_data, config_for_model.batch_size, config_for_model.is_single_batch)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_params = [
        # encoder
        {
            'params': [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and ".decoder" not in n and "bert" in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_bert_encoder
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and ".decoder" not in n and "bert" in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_bert_encoder
        },

        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and
                       ".decoder" not in n and "bert" not in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_NER_encoder
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and ".decoder" not in n and "bert" not in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_NER_encoder
        },

        # BFT decoder
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and ".decoder_for_BF" in n and "crossattention" not in n and "span" not in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_bert_decoder_in_BF
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and ".decoder_for_BF" in n and "crossattention" not in n and "span" not in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_bert_decoder_in_BF
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and "crossattention" in n and ".decoder_for_BF" in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_crossattention_in_BF
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and "crossattention" in n and ".decoder_for_BF" in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_crossattention_in_BF
        },
        {
            'params': [p for n, p in model.named_parameters() if not
                       any(nd in n for nd in no_decay) and "span" in n and ".decoder_for_BF" in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_RE_decoder_in_BF
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and "span" in n and ".decoder_for_BF" in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_RE_decoder_in_BF
        },

        # SPN decoder
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and ".decoder_for_RD" in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_in_RD
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and ".decoder_for_RD" in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_in_RD
        }
    ]

    optimizer = AdamW(grouped_params)
    if config_for_model.lr_mode == "2":
        step_total = len(train_iter) * config_for_model.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(step_total * config_for_model.warmup_proportion),
                                                    step_total)

    train_loss = 0
    best_f1 = 0
    best_f1_rel = 0
    global_step = 0
    train_examples = 0
    best_result_epoch = 0
    best_result_epoch_rel = 0
    for epoch in range(config_for_model.epoch):
        model.train()
        # optimizer.zero_grad()
        if config_for_model.lr_mode == "1":
            optimizer = lr_decay(optimizer, epoch, config_for_model.lr_decay)
        logger.info('=== Epoch %d ===' % epoch)

        for data in tqdm(train_iter):
            global_step += 1
            train_examples += len(data)

            output_dict = model(data)
            loss = output_dict["loss_total"].mean()
            loss.backward()

            train_loss += loss.item()
            if config_for_model.max_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_for_model.max_grad_norm)

            optimizer.step()

            if config_for_model.lr_mode == "2":
                scheduler.step()

            optimizer.zero_grad()

            if global_step % config_for_model.print_loss_step == 0 and global_step != 0:
                logger.info('loss=%.5f' % (train_loss / train_examples))
                train_loss = 0
                train_examples = 0
        # gc.collect()
        # torch.cuda.empty_cache()
        F1_triple, rel_F1_triple = eval_for_RFBFN(model, test_iter, config_for_model)
        if F1_triple > best_f1:
            best_f1 = F1_triple
            best_result_epoch = epoch
            logger.info('!!! Best eval (epoch=%d) on test data: %.5f' % (epoch, F1_triple))
            save_model(model, config_for_model.model_save_path, epoch)
        if rel_F1_triple > best_f1_rel and not os.path.exists(os.path.join(config_for_model.model_save_path,
                                                              "epoch_" + str(epoch) + "_save_all_model.pkl")):
            best_f1_rel = rel_F1_triple
            best_result_epoch_rel = epoch
            logger.info('!!! Best REL eval (epoch=%d) on test data: %.5f' % (epoch, best_f1_rel))
            save_model(model, config_for_model.model_save_path, epoch)
        # gc.collect()
        # torch.cuda.empty_cache()
    logger.info('!!! Best eval (epoch=%d) on test data: %.5f' % (best_result_epoch, best_f1))
    logger.info('!!! Best REL eval (epoch=%d) on test data: %.5f' % (best_result_epoch_rel, best_f1_rel))
    pickle.dump(best_result_epoch, open("./pred_result/f1_save.pkl", "wb"))
    pickle.dump(best_result_epoch_rel, open("./pred_result/rel_f1_save.pkl", "wb"))


