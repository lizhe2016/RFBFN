# -*- coding : utf-8 -*-
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from RFBFN_model import RFBFN_Model_SELECT, RFBFN_Model
from RFBFN_config import ArgumentParser
import pickle
from utils import get_batch, set_seed, lr_decay, save_model, load_state_dict_for_model

import os
import sys
import logging
from tqdm import tqdm

config_for_model = ArgumentParser()

if config_for_model.gpu_setting != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = config_for_model.gpu_setting

logging.basicConfig(filename=os.path.join(config_for_model.log_path, "select_info.log"),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


if __name__ == '__main__':
    set_seed(config_for_model.seed)

    logger.addHandler(logging.FileHandler(os.path.join(config_for_model.log_path, "select_train.log"), 'w'))
    logger.info(sys.argv)
    logger.info(config_for_model)

    model_origin = RFBFN_Model(config_for_model)
    best_epoch_rel = pickle.load(open("./pred_result/f1_save.pkl", "rb"))
    best_epoch_path_rel = os.path.join("./save_model/", "epoch_" + str(best_epoch_rel) + "_save_all_model.pkl")

    model_origin = load_state_dict_for_model(model_origin, best_epoch_path_rel)

    model = RFBFN_Model_SELECT(config_for_model)

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
                       not any(nd in n for nd in no_decay) and "bert" in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_bert_encoder
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and "bert" in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_bert_encoder
        },

        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and
                       "bert" not in n],
            'weight_decay': config_for_model.weight_decay,
            'lr': config_for_model.learning_rate_for_SELECT_encoder
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and "bert" not in n],
            'weight_decay': 0.0,
            'lr': config_for_model.learning_rate_for_SELECT_encoder
        }
    ]

    optimizer = AdamW(grouped_params)
    if config_for_model.lr_mode == "2":
        step_total = len(train_iter) * config_for_model.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(step_total * config_for_model.warmup_proportion),
                                                    step_total)

    train_loss = 0
    best_f1 = 0
    global_step = 0
    train_examples = 0
    best_result_epoch = 0

    for epoch in range(config_for_model.epoch_select):
        model_origin.eval()
        model.train()
        # optimizer.zero_grad()
        if config_for_model.lr_mode == "1":
            optimizer = lr_decay(optimizer, epoch, config_for_model.lr_decay)
        logger.info('=== Epoch %d ===' % epoch)

        for data in tqdm(train_iter):
            global_step += 1
            train_examples += len(data)
            with torch.no_grad():
                output_dict_eval = model_origin(data)
                class_logits = output_dict_eval["class_logits"]

            output_dict = model(data, class_logits)
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
        save_model(model, config_for_model, epoch)






