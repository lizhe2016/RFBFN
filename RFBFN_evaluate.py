# -*- coding : utf-8 -*-
import torch
from tqdm import tqdm
import logging
logger = logging.getLogger('root')


def eval_for_RFBFN(model, eval_iter, config_for_model):
    logger.info('Evaluating...')
    pred_triplets_in_RD, pred_triplets_in_BF, gold_triplets, right_entity, gold_total, pred_total = \
        model_eval(model, eval_iter, config_for_model)
    _, rel_F1_triple = get_performance_triplets(pred_triplets_in_RD, gold_triplets, log_choose=0)
    F1_BF, _ = get_performance_triplets(pred_triplets_in_BF, gold_triplets, log_choose=1)

    return F1_BF, rel_F1_triple


def calculate_result(output_dict_eval):
    # logits_for_NER_eval.shape = [bz, span_len, num_ner_label] ==> [bz, real_span_len]
    logits_for_NER_eval = output_dict_eval["logits_for_NER"]

    logits_for_triple_eval = output_dict_eval["logits_for_answer"]

    class_logits_eval = output_dict_eval["class_logits"]
    head_logits_eval = output_dict_eval["head_logits"]
    tail_logits_eval = output_dict_eval["tail_logits"]

    _, predicted_relation_labels = class_logits_eval.max(2)
    _, predicted_head = head_logits_eval.max(2)
    _, predicted_tail = tail_logits_eval.max(2)

    predicted_relation_labels = predicted_relation_labels.cpu().numpy()
    predicted_head = predicted_head.cpu().numpy()
    predicted_tail = predicted_tail.cpu().numpy()

    _, predicted_entity_labels = logits_for_NER_eval.max(2)
    _, predicted_answers = logits_for_triple_eval.max(2)

    # predicted_label.shape = [bz, span_len]
    predicted_entity_labels = predicted_entity_labels.cpu().numpy()

    predicted_answers = predicted_answers.cpu().numpy()
    return predicted_relation_labels, predicted_head, predicted_tail, predicted_entity_labels, predicted_answers


def calculate_result_generate(output_dict_eval):
    # logits_for_NER_eval.shape = [bz, span_len, num_ner_label] ==> [bz, real_span_len]
    logits_for_NER_eval = output_dict_eval["logits_for_NER"]
    _, predicted_entity_labels = logits_for_NER_eval.max(2)
    predicted_entity_labels = predicted_entity_labels.cpu().numpy()

    class_logits_eval = output_dict_eval["class_logits"]
    head_logits_eval = output_dict_eval["head_logits"]
    tail_logits_eval = output_dict_eval["tail_logits"]

    _, predicted_relation_labels = class_logits_eval.softmax(-1).max(2)

    _, predicted_head = head_logits_eval.max(2)
    _, predicted_tail = tail_logits_eval.max(2)

    predicted_relation_labels = predicted_relation_labels.cpu().numpy()
    predicted_head = predicted_head.cpu().numpy()
    predicted_tail = predicted_tail.cpu().numpy()
    predicted_relation_labels_select = None
    return predicted_relation_labels, predicted_head, predicted_tail, predicted_entity_labels, \
           predicted_relation_labels_select



def model_eval(model, eval_iter, config_for_model):
    model.eval()

    right_entity = 0
    gold_total = 0
    pred_total = 0

    gold_triplets = {}
    pred_triplets_in_RD = {}
    pred_triplets_in_BF = {}

    with torch.no_grad():
        for data_list_eval in tqdm(eval_iter):
            output_dict_eval = model(data_list_eval)
            predicted_relation_labels, predicted_head, predicted_tail, predicted_entity_labels, predicted_answers = \
                calculate_result(output_dict_eval)

            for i, sample in enumerate(data_list_eval):
                # len(sample["span_labels"]) can be replaced by span_mask which means all the possible spans num
                span_labels = sample["span_labels"].cpu().numpy()
                for j in range(len(sample["span_labels"])):
                    gold_entity_label = span_labels[j]
                    pred_entity_label = predicted_entity_labels[i][j]
                    if pred_entity_label != 0 and gold_entity_label == pred_entity_label:
                        right_entity += 1
                    if pred_entity_label != 0:
                        pred_total += 1
                    if gold_entity_label != 0:
                        gold_total += 1

                sample_relation_id = sample["relation_id"].cpu().numpy()
                sample_head_entity_pos = sample["head_entity_pos"].cpu().numpy()
                sample_tail_entity_pos = sample["tail_entity_pos"].cpu().numpy()
                if sample["doc_key"] not in gold_triplets.keys():
                    gold_triplets[sample["doc_key"]] = []
                for gold_triplet_id in range(len(sample["relation_id"])):
                    gold_relation_id = sample_relation_id[gold_triplet_id]
                    gold_head = sample_head_entity_pos[gold_triplet_id]
                    gold_tail = sample_tail_entity_pos[gold_triplet_id]
                    gold_triplet = (gold_relation_id, gold_head, gold_tail)
                    if gold_triplet not in gold_triplets[sample["doc_key"]]:
                        gold_triplets[sample["doc_key"]].append(gold_triplet)

                if sample["doc_key"] not in pred_triplets_in_RD.keys():
                    pred_triplets_in_RD[sample["doc_key"]] = []
                for triplet_id in range(config_for_model.queries_num_for_RD):
                    pred_relation_id = predicted_relation_labels[i][triplet_id]
                    pred_head = predicted_head[i][triplet_id]
                    pred_tail = predicted_tail[i][triplet_id]
                    if pred_relation_id != 0:
                        pred_triplet = (pred_relation_id, pred_head, pred_tail)
                        if pred_triplet not in pred_triplets_in_RD[sample["doc_key"]]:
                            pred_triplets_in_RD[sample["doc_key"]].append(pred_triplet)

                template_relation_id = sample["template_relation_id"].cpu().numpy()
                if sample["doc_key"] not in pred_triplets_in_BF.keys():
                    pred_triplets_in_BF[sample["doc_key"]] = []
                pred_answer_set = get_answer_pair(predicted_answers[i])
                if len(pred_answer_set) != 0:
                    for a in pred_answer_set:
                        pred_triplet_in_BF = (template_relation_id[0], a[0] - 1, a[1] - 1)
                        if pred_triplet_in_BF not in pred_triplets_in_BF[sample["doc_key"]]:
                            pred_triplets_in_BF[sample["doc_key"]].append(pred_triplet_in_BF)
                pred_answer_set.clear()
    return pred_triplets_in_RD, pred_triplets_in_BF, gold_triplets, right_entity, gold_total, pred_total


def get_performance_triplets(final_pred_triplets, gold_triplets, log_choose):
    rel_right = 0
    entity_pair_right = 0

    right_triples = 0
    pred_total_triple = 0
    gold_total_triple = 0
    for key_p in final_pred_triplets.keys():
        right_triples += len(set(final_pred_triplets[key_p]) & set(gold_triplets[key_p]))
        pred_total_triple += len(set(final_pred_triplets[key_p]))
        gold_total_triple += len(set(gold_triplets[key_p]))
        for tri in list(set(final_pred_triplets[key_p])):
            if tri[0] in [g[0] for g in gold_triplets[key_p]]:
                rel_right = rel_right + 1
            if tri[1:] in [g[1:] for g in gold_triplets[key_p]]:
                entity_pair_right = entity_pair_right + 1
    precision, recall, F1_triple = F1_score(right_triples, pred_total_triple, gold_total_triple)
    if log_choose:
        logger.info('precision_triple: %.5f, recall_triple: %.5f, F1_triple: %.5f' % (precision, recall, F1_triple))
    rel_precision, rel_recall, rel_F1_triple = F1_score(rel_right, pred_total_triple, gold_total_triple)
    return F1_triple, rel_F1_triple


def F1_score(right_num, pred_total, gold_total):
    if pred_total != 0:
        precision = right_num / pred_total
    else:
        precision = -1

    recall = right_num / gold_total
    if precision == 0 or recall == 0 or precision == -1:
        F1_triple = -1
    else:
        F1_triple = 2 * precision * recall / (precision + recall)
    return precision, recall, F1_triple


def get_answer_pair(each_answer):
    answer_pair = []
    for k in range(0, len(each_answer), 2):
        answer_pair.append((each_answer[k], each_answer[k+1]))
    answer_pair = set(answer_pair)
    if (0, 0) in answer_pair:
        answer_pair.remove((0, 0))
    return answer_pair


def calculate_result_generate_select(output_dict_eval, output_dict_eval_select):
    # logits_for_NER_eval.shape = [bz, span_len, num_ner_label] ==> [bz, real_span_len]
    logits_for_NER_eval = output_dict_eval["logits_for_NER"]
    _, predicted_entity_labels = logits_for_NER_eval.max(2)
    predicted_entity_labels = predicted_entity_labels.cpu().numpy()

    class_logits_eval = output_dict_eval["class_logits"]
    head_logits_eval = output_dict_eval["head_logits"]
    tail_logits_eval = output_dict_eval["tail_logits"]

    # [bz, num_triple]
    _, select_logits_eval = output_dict_eval_select["select_logits"].max(2)

    _, predicted_relation_labels = class_logits_eval.softmax(-1).max(2)
    predicted_relation_labels_select = predicted_relation_labels * select_logits_eval

    _, predicted_head = head_logits_eval.max(2)
    _, predicted_tail = tail_logits_eval.max(2)

    predicted_relation_labels = predicted_relation_labels.cpu().numpy()
    predicted_head = predicted_head.cpu().numpy()
    predicted_tail = predicted_tail.cpu().numpy()
    predicted_relation_labels_select = predicted_relation_labels_select.cpu().numpy()
    return predicted_relation_labels, predicted_head, predicted_tail, predicted_entity_labels, \
           predicted_relation_labels_select











