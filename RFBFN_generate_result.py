# -*- coding : utf-8 -*-
import torch
import json
from RFBFN_model import RFBFN_Model, RFBFN_Model_SELECT
from RFBFN_evaluate import calculate_result_generate, calculate_result_generate_select, eval_for_RFBFN, model_eval, \
    get_performance_triplets
from RFBFN_config import ArgumentParser
import pickle
from utils import get_batch, save_model, set_seed, lr_decay, load_state_dict_for_model, get_labelmap

from data_preprocess import nyt_rel_labels_constant, webnlg_rel_labels_constant, \
    webnlg_rel_labels_constant_no_star
import os
import sys
import logging
from tqdm import tqdm

config_for_model = ArgumentParser()
data_config = ArgumentParser()

if config_for_model.gpu_setting != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = config_for_model.gpu_setting

logging.basicConfig(filename=os.path.join("./pred_result/generate_result/",
                                          "generate_info.log"),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def generate_json():
    if config_for_model.task == "webnlg":
        test_data_R = pickle.load(open("./data_preprocess/processed_data/webnlg_all_test_samples.pkl", "rb"))
        if config_for_model.star == 1:
            with open("./data_preprocess/data/webnlg_star/test.json") as f:
                lines = f.readlines()
                origin_data = [eval(ele) for ele in lines]
        else:
            with open("./data_preprocess/data/webnlg/test.json") as f:
                lines = f.readlines()
                origin_data = [eval(ele) for ele in lines]
        origin_iter_for_SPN = get_batch(origin_data, config_for_model.batch_size, config_for_model.is_single_batch)
    else:
        test_data_R = pickle.load(open("./data_preprocess/processed_data/nyt_all_test_samples.pkl", "rb"))
        if config_for_model.star == 1:
            with open("./data_preprocess/data/nyt_star/test.json") as f:
                lines = f.readlines()
                origin_data = [eval(ele) for ele in lines]
        else:
            with open("./data_preprocess/data/nyt/test.json") as f:
                lines = f.readlines()
                origin_data = [eval(ele) for ele in lines]
        origin_iter_for_SPN = get_batch(origin_data, config_for_model.batch_size, config_for_model.is_single_batch)

    test_data_R_single = []
    doc_p = "sentence-1"
    for d in test_data_R:
        if d["doc_key"] == doc_p:
            continue
        doc_p = d["doc_key"]
        test_data_R_single.append(d)
    test_iter = get_batch(test_data_R_single, config_for_model.batch_size, config_for_model.is_single_batch)

    model = RFBFN_Model(config_for_model)

    best_epoch_rel = pickle.load(open("./pred_result/f1_save.pkl", "rb"))
    best_epoch_path_rel = os.path.join("./save_model/", "epoch_" + str(best_epoch_rel) + "_save_all_model.pkl")

    model = load_state_dict_for_model(model, best_epoch_path_rel)

    model.eval()
    if config_for_model.task == "nyt":
        rel_label2id, rel_id2label = get_labelmap(nyt_rel_labels_constant)
    else:
        if config_for_model.star == 1:
            rel_label2id, rel_id2label = get_labelmap(webnlg_rel_labels_constant)
        else:
            rel_label2id, rel_id2label = get_labelmap(webnlg_rel_labels_constant_no_star)

    gold_triplets = {}
    with torch.no_grad():
        for data_list_eval, origin_list in tqdm(zip(test_iter, origin_iter_for_SPN),
                                                total=len(test_iter)):
            output_dict_eval = model(data_list_eval)

            predicted_relation_labels, predicted_head, predicted_tail, predicted_entity_labels, \
            predicted_relation_labels_select = calculate_result_generate(output_dict_eval)

            for i, sample in enumerate(origin_list):
                doc_key = data_list_eval[i]["doc_key"]
                pred_sample = {}
                pred_relations_in_current_sample = []
                pred_sample["sentText"] = sample["sentText"]
                pred_sample["relationMentions"] = []
                for triplet_id in range(config_for_model.queries_num_for_RD):
                    pred_relation_id = predicted_relation_labels[i][triplet_id]
                    if pred_relation_id != 0:
                        pred_relations_in_current_sample.append(pred_relation_id)
                # save json for BFT
                pred_relations_in_current_sample = list(set(pred_relations_in_current_sample))
                for each_pred_rel in pred_relations_in_current_sample:
                    if each_pred_rel in [rel_label2id[each_gold_rel["label"]]
                                         for each_gold_rel in sample["relationMentions"]]:
                        row_each = {"label": rel_id2label[each_pred_rel], "em1Text": "", "em2Text": ""}
                        pred_sample["relationMentions"].append(row_each)
                with open("./pred_result/generate_result/pred_json_for_BF.json",
                          "a") as f:
                    f.write(json.dumps(pred_sample))
                    f.write("\r")

                sample_relation_id = data_list_eval[i]["relation_id"].cpu().numpy()
                sample_head_entity_pos = data_list_eval[i]["head_entity_pos"].cpu().numpy()
                sample_tail_entity_pos = data_list_eval[i]["tail_entity_pos"].cpu().numpy()
                if doc_key not in gold_triplets.keys():
                    gold_triplets[doc_key] = []
                for gold_triplet_id in range(len(sample_relation_id)):
                    gold_relation_id = sample_relation_id[gold_triplet_id]
                    gold_head = sample_head_entity_pos[gold_triplet_id]
                    gold_tail = sample_tail_entity_pos[gold_triplet_id]
                    gold_triplet = (gold_relation_id, gold_head, gold_tail)
                    if gold_triplet not in gold_triplets[doc_key]:
                        gold_triplets[doc_key].append(gold_triplet)
    pickle.dump(gold_triplets, open("./pred_result/generate_result/gold_test_triplets.pkl", "wb"))


def get_all_performance():
    model = RFBFN_Model(config_for_model)
    best_epoch = pickle.load(open("./pred_result/rel_f1_save.pkl", "rb"))
    best_epoch_path = os.path.join("./save_model/", "epoch_" + str(best_epoch) + "_save_all_model.pkl")
    model = load_state_dict_for_model(model, best_epoch_path)

    if config_for_model.task == "webnlg":
        test_data_pred = pickle.load(open("./pred_result/pred_data/webnlg_all_test_samples.pkl", "rb"))
        test_iter = get_batch(test_data_pred, config_for_model.batch_size, config_for_model.is_single_batch)
    else:
        test_data_pred = pickle.load(open("./pred_result/pred_data/nyt_all_test_samples.pkl", "rb"))
        test_iter = get_batch(test_data_pred, config_for_model.batch_size, config_for_model.is_single_batch)

    gold_triplets = pickle.load(open("./pred_result/generate_result/gold_test_triplets.pkl", "rb"))
    logger.info('Evaluating...')
    _, pred_triplets_in_BF, _, _, _, _ = \
        model_eval(model, test_iter, config_for_model)
    F1_BFN, _ = get_performance_triplets(pred_triplets_in_BF, gold_triplets, log_choose=1)


if __name__ == '__main__':
    set_seed(config_for_model.seed)
    logger.addHandler(logging.FileHandler(os.path.join("./pred_result/generate_result/",
                                                       "SPN_generate_test.log"), 'w'))
    logger.info(sys.argv)
    logger.info(config_for_model)

    if config_for_model.generate_step == "1":
        generate_json()
    else:
        get_all_performance()

