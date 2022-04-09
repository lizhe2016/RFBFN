# -*- coding : utf-8 -*-
from torch.utils import data
import torch

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import RobertaTokenizer, RobertaModel

from constant import nyt_ner_labels_constant, nyt_rel_labels_constant, nyt_rel_to_question, get_labelmap, \
    webnlg_ner_labels_constant, webnlg_rel_labels_constant, webnlg_rel_to_question, \
    webnlg_rel_to_question_no_star, webnlg_rel_labels_constant_no_star
from data_config import ArgumentParser

import numpy as np
import random
import json
import logging
from tqdm import tqdm
import pickle
import os
import sys

data_config = ArgumentParser()


class BFT_data(data.DataLoader):
    def __init__(self, data_config, data_json, data_type, save_path, step):
        self.step = step
        self.config = data_config
        self.data_type = data_type
        self.data_json = data_json

        self.pretrained_model = self.config.pretrained_model
        if self.pretrained_model != "roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model)

        self.task = self.config.task
        self.star = self.config.star
        if self.task == "nyt":
            ner_label2id, ner_id2label = get_labelmap(nyt_ner_labels_constant)
        else:
            if self.star == 1:
                ner_label2id, ner_id2label = get_labelmap(webnlg_ner_labels_constant)
            else:
                ner_label2id, ner_id2label = get_labelmap(webnlg_ner_labels_constant_no_star)
        if os.path.exists(save_path + self.task + "_preprocessed_" + self.data_type + "_data.pkl"):
            preprocessed_data = pickle.load(open(save_path + self.task + "_preprocessed_" + self.data_type +
                                                 "_data.pkl", "rb"))
            print("Load preprocessed data finish!")
        else:
            preprocessed_data = self.prepare_data(ner_label2id)
            pickle.dump(preprocessed_data, open(save_path + self.task + "_preprocessed_" + self.data_type
                                                + "_data.pkl", "wb"))
            print("Save preprocessed data finish!")

        if os.path.exists(save_path + self.task + "_all_" + self.data_type + "_samples.pkl"):
            self.all_samples = pickle.load(open(save_path + self.task + "_all_" + self.data_type +
                                                "_samples.pkl", "rb"))
            print("Load all data finish!")
        else:
            self.all_samples = self.convert_all_data_to_input_form(preprocessed_data)
            pickle.dump(self.all_samples, open(save_path + self.task + "_all_" + self.data_type +
                                               "_samples.pkl", "wb"))
            print("Save all data finish!")
        print("Prepare data finish!")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, item):
        return self.all_samples[item]

    def generate_question_with_template(self, r):
        question_tokens = []
        question_tokens.append(self.tokenizer.mask_token)

        if self.task == "nyt":
            rel_to_question = nyt_rel_to_question
        else:
            if self.star == 1:
                rel_to_question = webnlg_rel_to_question
            else:
                rel_to_question = webnlg_rel_to_question_no_star

        question_tokens += self.tokenizer.tokenize(rel_to_question[r])
        question_tokens.append(self.tokenizer.mask_token)
        question_tokens.append(self.tokenizer.sep_token)
        question_token_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
        return question_token_ids

    def get_decoder_input(self, sample):
        each_relation = {}
        for items in sample["relations"].values():
            for item in items:
                relation = item[2]
                answer = [item[0], item[1]]
                if relation not in each_relation.keys():
                    each_relation[relation] = []
                each_relation[relation].append(answer)

        if self.task == "nyt":
            rel_labels_constant = nyt_rel_labels_constant
        else:
            if self.star == 1:
                rel_labels_constant = webnlg_rel_labels_constant
            else:
                rel_labels_constant = webnlg_rel_labels_constant_no_star

        rel_label2id, rel_id2label = get_labelmap(rel_labels_constant)

        for r in rel_labels_constant:
            if r not in each_relation.keys():
                each_relation[r] = []

        if self.config.sorted_entity:
            for i in each_relation.items():
                each_relation[i[0]] = sorted(i[1], key=lambda x: 1000000 * x[0] + x[1])
        else:
            for i in each_relation.items():
                random.shuffle(i[1])

        if self.config.sorted_relation:
            each_relation = sorted(each_relation.items(), key=lambda x: x[0])
        else:
            each_relation = sorted(each_relation.items(), key=lambda x: x[0])
            random.shuffle(each_relation)

        relations_and_answers = each_relation

        relation_id_for_each_template = []
        questions_input = []
        answers_input = []
        mask_pos_input = []
        question_token_type_ids = []
        for relation, answers in relations_and_answers:
            answer_input = []
            question_input = self.generate_question_with_template(relation) * self.config.duplicate_questions
            question_input[-1] = self.tokenizer.convert_tokens_to_ids(".")

            if self.config.token_type_ids:
                question_token_type_id = self.get_token_type_ids(question_input)
            else:
                question_token_type_id = [0] * len(question_input)

            mask_pos = np.argwhere(np.array(question_input) ==
                                   self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token))
            mask_pos = mask_pos.squeeze().tolist()
            answer_input.extend(answers)
            if len(answer_input) > self.config.duplicate_questions:
                answer_input = answer_input[:self.config.duplicate_questions]
            for i in range(self.config.duplicate_questions - len(answers)):
                answer_input.append([-1, -1])

            relation_id_for_each_template.append([rel_label2id[relation]])
            questions_input.append(question_input)
            answers_input.extend([answer_input])
            mask_pos_input.append(mask_pos)
            question_token_type_ids.append(question_token_type_id)

        answers_input = np.array(answers_input)
        answers_input = answers_input.reshape(answers_input.shape[0], -1).tolist()
        # questions_input.shape = (relation_num, *)
        # answers_input.shape = (relation_num, 20)
        # mask_pos_input.shape = (relation_num, 20)
        return relation_id_for_each_template, questions_input, answers_input, mask_pos_input, question_token_type_ids

    def get_SPN_input(self, sample):
        if self.task == "nyt":
            rel_labels_constant = nyt_rel_labels_constant
        else:
            if self.star == 1:
                rel_labels_constant = webnlg_rel_labels_constant
            else:
                rel_labels_constant = webnlg_rel_labels_constant_no_star

        rel_label2id, rel_id2label = get_labelmap(rel_labels_constant)
        relation_id = []
        head_entity_pos = []
        tail_entity_pos = []
        for items in sample["relations"].values():
            for item in items:
                relation_id.append(rel_label2id[item[2]])
                head_entity_pos.append(item[0])
                tail_entity_pos.append(item[1])
        return relation_id, head_entity_pos, tail_entity_pos

    def get_token_type_ids(self, question_input):
        token_type_ids = []
        flag = True
        for q_id in question_input:
            if flag:
                token_type_ids.append(0)
            else:
                token_type_ids.append(1)
            if q_id == self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token):
                flag = not flag
        return token_type_ids

    def convert_each_data_to_input_form(self, sample):
        id2start = []
        id2end = []
        tokenizer_tokens = []
        input_sample = []

        tokenizer_tokens.append(self.tokenizer.cls_token)
        for word in sample["sentences"]:
            id2start.append(len(tokenizer_tokens))
            sub_tokens = self.tokenizer.tokenize(word)
            tokenizer_tokens += sub_tokens
            id2end.append(len(tokenizer_tokens)-1)
        # tokenizer_tokens.append(self.tokenizer.sep_token)

        input_id = self.tokenizer.convert_tokens_to_ids(tokenizer_tokens)
        input_spans = [[id2start[span[0]], id2end[span[1]], span[2]] for span in sample["spans"]]
        relation_id, head_entity_pos, tail_entity_pos = self.get_SPN_input(sample)
        relation_id_for_each_template, input_decoder_questions, input_decoder_answers, input_decoder_mask_pos, \
        input_question_token_type_ids = self.get_decoder_input(sample)

        negative_probability = None
        if self.step == "1":
            negative_probability = self.config.negative_probability
        if self.step == "2":
            negative_probability = 0

        for r_id, q, a, m, t in zip(relation_id_for_each_template, input_decoder_questions, input_decoder_answers,
                                    input_decoder_mask_pos, input_question_token_type_ids):
            p = random.random()
            if sum(a) == -len(a) and p < negative_probability:
                input_each_sample = {}
                input_each_sample["question"] = torch.tensor(q)     # list
                input_each_sample["template_relation_id"] = torch.tensor(r_id)
                input_each_sample["answer"] = torch.tensor(a)
                input_each_sample["decoder_mask_pos"] = torch.tensor(m)
                input_each_sample["question_token_type_id"] = torch.tensor(t)

                input_each_sample["id"] = torch.tensor(input_id)
                input_each_sample["spans"] = torch.tensor(input_spans)
                input_each_sample["span_labels"] = torch.tensor(sample["spans_label"])
                input_each_sample["doc_key"] = sample["doc_key"]

                input_each_sample["relation_id"] = torch.tensor(relation_id)
                input_each_sample["head_entity_pos"] = torch.tensor(head_entity_pos)
                input_each_sample["tail_entity_pos"] = torch.tensor(tail_entity_pos)
                input_sample.append(input_each_sample)
            elif sum(a) != -len(a):
                input_each_sample = {}
                input_each_sample["question"] = torch.tensor(q)  # list
                input_each_sample["template_relation_id"] = torch.tensor(r_id)
                input_each_sample["answer"] = torch.tensor(a)
                input_each_sample["decoder_mask_pos"] = torch.tensor(m)
                input_each_sample["question_token_type_id"] = torch.tensor(t)

                input_each_sample["id"] = torch.tensor(input_id)
                input_each_sample["spans"] = torch.tensor(input_spans)
                input_each_sample["span_labels"] = torch.tensor(sample["spans_label"])
                input_each_sample["doc_key"] = sample["doc_key"]

                input_each_sample["relation_id"] = torch.tensor(relation_id)
                input_each_sample["head_entity_pos"] = torch.tensor(head_entity_pos)
                input_each_sample["tail_entity_pos"] = torch.tensor(tail_entity_pos)
                input_sample.append(input_each_sample)
            else:
                pass
        if len(input_sample) == 0:
            rand_id = int(random.random() * len(input_decoder_questions))
            input_each_sample = {}
            input_each_sample["question"] = torch.tensor(input_decoder_questions[rand_id])  # list
            input_each_sample["template_relation_id"] = torch.tensor(relation_id_for_each_template[rand_id])
            input_each_sample["answer"] = torch.tensor(input_decoder_answers[rand_id])
            input_each_sample["decoder_mask_pos"] = torch.tensor(input_decoder_mask_pos[rand_id])
            input_each_sample["question_token_type_id"] = torch.tensor(input_question_token_type_ids[rand_id])

            input_each_sample["id"] = torch.tensor(input_id)
            input_each_sample["spans"] = torch.tensor(input_spans)
            input_each_sample["span_labels"] = torch.tensor(sample["spans_label"])
            input_each_sample["doc_key"] = sample["doc_key"]

            input_each_sample["relation_id"] = torch.tensor(relation_id)
            input_each_sample["head_entity_pos"] = torch.tensor(head_entity_pos)
            input_each_sample["tail_entity_pos"] = torch.tensor(tail_entity_pos)
            input_sample.append(input_each_sample)
        return input_sample

    def convert_all_data_to_input_form(self, preprocessed_data):
        all_samples = []
        for sample in tqdm(preprocessed_data):
            input_sample = self.convert_each_data_to_input_form(sample)
            all_samples.extend(input_sample)
        return all_samples

    def generate_samples(self, doc):
        keys_to_ignore = ["doc_key", "clusters"]
        keys = [key for key in doc.keys() if key not in keys_to_ignore]
        lengths = [len(doc[k]) for k in keys]
        assert len(set(lengths)) == 1
        length = lengths[0]
        res = [{k: doc[k][i] for k in keys} for i in range(length)]
        return res

    def prepare_data(self, ner_label2id):
        with open(self.data_json) as f:
            lines = f.readlines()
            documents = [eval(ele) for ele in lines]
        max_span_num = 0
        span_num_count = 0
        final_samples = []
        for sentence_id, sample in tqdm(enumerate(documents)):
            final_sample = {}
            tokens = self.remove_accents(sample["sentText"]).split(" ")
            final_sample["doc_key"] = "sentence" + str(sentence_id)
            final_sample["sentences"] = tokens

            triples = sample["relationMentions"]
            final_sample["ner"] = {}
            for triple in triples:
                head_entity = self.remove_accents(triple["em1Text"]).split(" ")
                head_start, head_end = self.list_index(head_entity, tokens)
                tail_entity = self.remove_accents(triple["em2Text"]).split(" ")
                tail_start, tail_end = self.list_index(tail_entity, tokens)

                this_head_start = head_start
                this_head_end = head_end
                final_sample["ner"][(this_head_start, this_head_end)] = (tokens[this_head_start: this_head_end + 1],
                                                                         "entity")

                this_tail_start = tail_start
                this_tail_end = tail_end
                final_sample["ner"][(this_tail_start, this_tail_end)] = (tokens[this_tail_start: this_tail_end + 1],
                                                                         "entity")
                if this_tail_end - this_tail_start >= self.config.max_span_length:
                    print("\n" + "Note: there is one entity longer than the max_span_length: "
                          + str(self.config.max_span_length), ". The entity is ( "
                          + " ".join(tokens[this_tail_start: this_tail_end + 1]) + " ). Its length is "
                          + str(this_tail_end - this_tail_start) + " .")

            span2id = {}
            final_sample['spans'] = []
            final_sample['spans_label'] = []
            for i in range(len(tokens)):
                for j in range(i, min(len(tokens), i + self.config.max_span_length)):
                    span_start = i
                    span_end = j
                    final_sample['spans'].append((span_start, span_end, j - i + 1))
                    span2id[(span_start, span_end)] = len(final_sample['spans']) - 1
                    if (span_start, span_end) not in final_sample["ner"].keys():
                        final_sample['spans_label'].append(0)
                    else:
                        label_name = final_sample["ner"][(span_start, span_end)][1]
                        final_sample['spans_label'].append(ner_label2id[label_name])

            max_span_num = max(len(final_sample['spans_label']), max_span_num)
            span_num_count = span_num_count + len(final_sample['spans_label'])
            final_sample["relations"] = {}

            for rel in triples:
                head_entity = self.remove_accents(rel["em1Text"]).split(" ")
                e1_start, e1_end = self.list_index(head_entity, tokens)
                tail_entity = self.remove_accents(rel["em2Text"]).split(" ")
                e2_start, e2_end = self.list_index(tail_entity, tokens)

                if e1_end - e1_start >= 8 or e2_end - e2_start >= 8:
                    pass
                else:
                    e1_span_id = span2id[(e1_start, e1_end)]
                    e2_span_id = span2id[(e2_start, e2_end)]
                    if ((e1_start, e1_end), (e2_start, e2_end)) not in final_sample["relations"].keys():
                        final_sample["relations"][((e1_start, e1_end), (e2_start, e2_end))] = []
                    final_sample["relations"][((e1_start, e1_end), (e2_start, e2_end))].append((e1_span_id,
                                                                                                e2_span_id,
                                                                                                rel["label"]))
            final_samples.append(final_sample)
        print("The max length of spans is ", max_span_num, ". The avg length of spans is ",
              int(span_num_count/len(documents)))
        return final_samples

    def remove_accents(self, text: str) -> str:
        accents_translation_table = str.maketrans(
            "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
            "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
        )
        return text.translate(accents_translation_table)

    def list_index(self, list1: list, list2: list) -> list:
        index = (0, 0)
        start = [i for i, x in enumerate(list2) if x == list1[0]]
        end = [i for i, x in enumerate(list2) if x == list1[-1]]
        if len(start) == 1 and len(end) == 1:
            return start[0], end[0]
        else:
            for i in start:
                for j in end:
                    if i <= j:
                        if list2[i:j + 1] == list1:
                            index = (i, j)
                            break
            if list1[0] != "Sudan":
                return index[0], index[1]
            else:
                return index[0], index[1]


if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join(data_config.log_path, "info.log"),
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(os.path.join(data_config.log_path, "data.log"), 'w'))
    logger.info(sys.argv)
    logger.info(data_config)
    if data_config.task == "webnlg":
        if data_config.step == "1":
            if data_config.star == 1:
                train_data_json = "./data/webnlg_star/train.json"
                test_data_json = "./data/webnlg_star/test.json"
                dev_data_json = "./data/webnlg_star/dev.json"
            else:
                train_data_json = "./data/webnlg/train.json"
                test_data_json = "./data/webnlg/test.json"
                dev_data_json = "./data/webnlg/dev.json"
            print("-" * 10, "train", "-" * 10)
            save_path = data_config.save_path
            train_data_type = "train"
            train_data_BFT = BFT_data(data_config, train_data_json, train_data_type, save_path, step="1")

            print("-" * 10, "test", "-" * 10)
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="1")

            print("-" * 10, "dev", "-" * 10)
            dev_data_type = "dev"
            dev_data_BFT = BFT_data(data_config, dev_data_json, dev_data_type, save_path, step="1")
        if data_config.step == "2":
            save_path = "../pred_result/pred_data/"
            test_data_json = "../pred_result/generate_result/pred_json_for_BF.json"
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="2")
    else:
        if data_config.step == "1":
            if data_config.star == 1:
                train_data_json = "./data/nyt_star/train.json"
                test_data_json = "./data/nyt_star/test.json"
                dev_data_json = "./data/nyt_star/dev.json"
            else:
                train_data_json = "./data/nyt/train.json"
                test_data_json = "./data/nyt/test.json"
                dev_data_json = "./data/nyt/dev.json"
            print("-" * 10, "train", "-" * 10)
            save_path = data_config.save_path
            train_data_type = "train"
            train_data_BFT = BFT_data(data_config, train_data_json, train_data_type, save_path, step="1")

            print("-" * 10, "test", "-" * 10)
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="1")

            print("-" * 10, "dev", "-" * 10)
            dev_data_type = "dev"
            dev_data_BFT = BFT_data(data_config, dev_data_json, dev_data_type, save_path, step="1")
        if data_config.step == "2":
            save_path = "../pred_result/pred_data/"
            test_data_json = "../pred_result/generate_result/pred_json_for_BF.json"
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="2")


