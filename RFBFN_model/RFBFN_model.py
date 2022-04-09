# -*- coding : utf-8 -*-
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertTokenizer, RobertaTokenizer

import numpy as np
import logging
logger = logging.getLogger('root')

from .Span_encoder import Span_Encoder
from .BF_decoder import BF_Decoder
from .RD_decoder import RD_Decoder_no_pretrain
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append("..")
from data_preprocess import nyt_ner_labels_constant, webnlg_ner_labels_constant

class RFBFN_Model(nn.Module):
    def __init__(self, config_for_model):
        super(RFBFN_Model, self).__init__()
        self.config_for_model = config_for_model
        pretrained_model = self.config_for_model.pretrained_model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = Model_ALL(config_for_model)
        # self.shared_parameters()
        self._model_device = "cpu"
        self.move_model_to_cuda()

    def forward(self, data_list):
        batch_tokens, batch_tokens_attention_mask, batch_spans, batch_span_labels, batch_span_mask, \
        batch_question_tokens, batch_question_tokens_attention_mask, batch_decoder_mask_pos, batch_answer, \
        batch_question_token_type_ids = self.get_mask(data_list)

        output_dict = {}
        loss, NER_loss, RE_loss_for_BF, RE_loss_for_RD, logits_for_NER, answer_logits, batch_answer, class_logits, \
        head_logits, tail_logits = \
            self.model(batch_tokens.to(self._model_device), batch_tokens_attention_mask.to(self._model_device),
                       batch_spans.to(self._model_device), batch_span_labels.to(self._model_device),
                       batch_span_mask.to(self._model_device), batch_question_tokens.to(self._model_device),
                       batch_question_tokens_attention_mask.to(self._model_device),
                       batch_decoder_mask_pos.to(self._model_device),
                       batch_answer.to(self._model_device), batch_question_token_type_ids.to(self._model_device),
                       self._model_device, data_list)

        output_dict["loss_total"] = loss
        output_dict["loss_NER"] = NER_loss
        output_dict["loss_RE_in_BF"] = RE_loss_for_BF
        output_dict["loss_RE_in_RD"] = RE_loss_for_RD

        output_dict["logits_for_NER"] = logits_for_NER
        output_dict["logits_for_answer"] = answer_logits
        output_dict["class_logits"] = class_logits
        output_dict["head_logits"] = head_logits
        output_dict["tail_logits"] = tail_logits

        output_dict["batch_answer"] = batch_answer

        return output_dict

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.model.cuda()
        logger.info('# GPUs = %d' % (torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def get_mask(self, data_list):
        max_tokens = 0
        max_spans = 0
        max_question_tokens = 0

        batch_tokens_final = None
        batch_tokens_attention_mask_final = None
        batch_spans_final = None
        batch_span_labels_final = None
        batch_span_mask_final = None
        batch_question_tokens_final = None
        batch_question_token_type_ids_final = None
        batch_question_tokens_attention_mask_final = None
        batch_decoder_mask_pos_final = None
        batch_answer_final = None

        for sample in data_list:
            if sample["id"].shape[0] > max_tokens:
                max_tokens = sample["id"].shape[0]
            if sample["spans"].shape[0] > max_spans:
                max_spans = sample["spans"].shape[0]
            if sample["question"].shape[0] > max_question_tokens:
                max_question_tokens = sample["question"].shape[0]

        for sample in data_list:
            current_tokens_len = sample["id"].shape[0]
            pad_tokens_len = max_tokens - current_tokens_len
            final_tokens = sample["id"].unsqueeze(0)
            token_attention_mask = torch.full([1, current_tokens_len], 1, dtype=torch.long)
            if pad_tokens_len > 0:
                tokens_pad = torch.full([1, pad_tokens_len], self.tokenizer.pad_token_id, dtype=torch.long)
                final_tokens = torch.cat((final_tokens, tokens_pad), dim=1)

                attention2 = torch.full([1, pad_tokens_len], 0, dtype=torch.long)
                token_attention_mask = torch.cat((token_attention_mask, attention2), dim=1)

            current_spans_len = sample["spans"].shape[0]
            pad_spans_len = max_spans - current_spans_len
            final_spans = sample["spans"].unsqueeze(0)
            final_span_labels = sample["span_labels"].unsqueeze(0)
            spans_mask = torch.full([1, current_spans_len], 1, dtype=torch.long)
            if pad_spans_len > 0:
                spans_pad = torch.full([1, pad_spans_len, sample["spans"].shape[1]], 0, dtype=torch.long)
                final_spans = torch.cat((final_spans, spans_pad), dim=1)

                span_labels_pad = torch.full([1, pad_spans_len], 0, dtype=torch.long)
                final_span_labels = torch.cat((final_span_labels, span_labels_pad), dim=1)

                attention2 = torch.full([1, pad_spans_len], 0, dtype=torch.long)
                spans_mask = torch.cat((spans_mask, attention2), dim=1)

            current_question_tokens_len = sample["question"].shape[0]
            pad_question_tokens_len = max_question_tokens - current_question_tokens_len
            final_question_tokens = sample["question"].unsqueeze(0)
            final_question_token_type_ids = sample["question_token_type_id"].unsqueeze(0)
            question_tokens_attention_mask = torch.full([1, current_question_tokens_len], 1, dtype=torch.long)
            if pad_question_tokens_len > 0:
                question_tokens_pad = torch.full([1, pad_question_tokens_len], self.tokenizer.pad_token_id,
                                                 dtype=torch.long)
                final_question_tokens = torch.cat((final_question_tokens, question_tokens_pad), dim=1)
                final_question_token_type_ids = torch.cat((final_question_token_type_ids, question_tokens_pad), dim=1)

                attention2 = torch.full([1, pad_question_tokens_len], 0, dtype=torch.long)
                question_tokens_attention_mask = torch.cat((question_tokens_attention_mask, attention2), dim=1)

            if batch_tokens_final == None:
                batch_tokens_final = final_tokens
                batch_tokens_attention_mask_final = token_attention_mask
                batch_spans_final = final_spans
                batch_span_labels_final = final_span_labels
                batch_span_mask_final = spans_mask
                batch_question_tokens_final = final_question_tokens
                batch_question_token_type_ids_final = final_question_token_type_ids
                batch_question_tokens_attention_mask_final = question_tokens_attention_mask
                batch_decoder_mask_pos_final = sample["decoder_mask_pos"].unsqueeze(0)
                batch_answer_final = sample["answer"].unsqueeze(0)

            else:
                batch_tokens_final = torch.cat((batch_tokens_final, final_tokens), dim=0)
                batch_tokens_attention_mask_final = torch.cat((batch_tokens_attention_mask_final, token_attention_mask), dim=0)
                batch_spans_final = torch.cat((batch_spans_final, final_spans), dim=0)
                batch_span_labels_final = torch.cat((batch_span_labels_final, final_span_labels), dim=0)
                batch_span_mask_final = torch.cat((batch_span_mask_final, spans_mask), dim=0)
                batch_question_tokens_final = torch.cat((batch_question_tokens_final, final_question_tokens), dim=0)
                batch_question_token_type_ids_final = torch.cat((batch_question_token_type_ids_final,
                                                                 final_question_token_type_ids), dim=0)
                batch_question_tokens_attention_mask_final = torch.cat((batch_question_tokens_attention_mask_final,
                                                                        question_tokens_attention_mask), dim=0)
                batch_decoder_mask_pos_final = torch.cat((batch_decoder_mask_pos_final,
                                                          sample["decoder_mask_pos"].unsqueeze(0)), dim=0)
                batch_answer_final = torch.cat((batch_answer_final, sample["answer"].unsqueeze(0)), dim=0)
        return batch_tokens_final, batch_tokens_attention_mask_final, batch_spans_final, batch_span_labels_final, \
               batch_span_mask_final, batch_question_tokens_final, batch_question_tokens_attention_mask_final, \
               batch_decoder_mask_pos_final, batch_answer_final, batch_question_token_type_ids_final


class Model_ALL(nn.Module):
    def __init__(self, config_for_model):
        super(Model_ALL, self).__init__()
        self.config_for_model = config_for_model
        pretrained_model = self.config_for_model.pretrained_model

        if config_for_model.task == "webnlg":
            num_ner_label = len(webnlg_ner_labels_constant) + 1
        else:
            num_ner_label = len(nyt_ner_labels_constant) + 1

        self.num_ner_label = num_ner_label

        self.encoder = Span_Encoder.from_pretrained(pretrained_model, config_for_model=config_for_model,
                                                    num_ner_label=num_ner_label)
        config = self.encoder.config
        self.decoder_for_BF = BF_Decoder.from_pretrained(pretrained_model, config_for_model=config_for_model)
        self.decoder_for_RD = RD_Decoder_no_pretrain(config, config_for_model=config_for_model)

    def forward(self, batch_tokens, batch_tokens_attention_mask, batch_spans, batch_span_labels, batch_span_mask,
                batch_question_tokens, batch_question_tokens_attention_mask, batch_decoder_mask_pos, batch_answer,
                token_type_ids, device, data_list):
        logits_for_NER, logits_for_BF, logits_for_RD, sequence_output_of_encoder_for_BF, \
        sequence_output_of_encoder_for_RD = self.encoder(batch_tokens, batch_tokens_attention_mask, batch_spans)
        NER_loss = self.get_NER_loss(logits_for_NER, batch_span_labels, batch_span_mask, device)

        batch_span_mask_for_BF = torch.cat((torch.ones(batch_span_mask.shape[0], 1).to(device), batch_span_mask), dim=1)

        answer_logits = self.decoder_for_BF(batch_tokens_attention_mask, batch_span_mask_for_BF, logits_for_BF,
                                             sequence_output_of_encoder_for_BF, batch_question_tokens,
                                             batch_question_tokens_attention_mask, batch_decoder_mask_pos, batch_answer,
                                             token_type_ids)
        class_logits, head_logits, tail_logits = \
            self.decoder_for_RD(batch_tokens_attention_mask, batch_span_mask, sequence_output_of_encoder_for_RD,
                                logits_for_RD)

        batch_answer = batch_answer + 1
        RE_loss_for_BF = self.get_RE_loss_for_BF(answer_logits, batch_answer, batch_span_mask_for_BF, device)

        RE_loss_for_RD = self.get_RE_loss_for_RD(class_logits, head_logits, tail_logits, device, data_list)

        loss = self.config_for_model.NER_loss_parameter * NER_loss + \
               self.config_for_model.RE_loss_for_BF_parameter * RE_loss_for_BF + \
               self.config_for_model.RE_loss_for_RD_parameter * RE_loss_for_RD

        return loss, NER_loss, RE_loss_for_BF, RE_loss_for_RD, logits_for_NER, answer_logits, batch_answer, \
               class_logits, head_logits, tail_logits

    def get_NER_loss(self, logits_for_NER, batch_span_labels, batch_span_mask, device):
        if self.config_for_model.NER_none_span_reweighting_in_BF == 0:
            loss_fc_NER = CrossEntropyLoss(reduction="sum")
        else:
            w = torch.ones(self.num_ner_label)
            w[0] = self.config_for_model.NER_none_span_reweighting_in_BF
            loss_fc_NER = CrossEntropyLoss(reduction="sum", weight=w.to(device))
        active_logits = logits_for_NER.view(-1, logits_for_NER.shape[-1])
        active_mask = batch_span_mask.view(-1) == 1
        active_labels = torch.where(active_mask, batch_span_labels.view(-1),
                                    torch.tensor(loss_fc_NER.ignore_index).type_as(batch_span_labels))
        NER_loss = loss_fc_NER(active_logits, active_labels)
        # NER_loss = NER_loss.sum()
        return NER_loss

    def get_RE_loss_for_BF(self, answer_logits, batch_answer, batch_span_mask, device):
        # RE ; NER
        # answer_logits.shape = [bz, mask_num (default 20), span_len + 1 ([CLS])] ; [bz, span_len, num_ner_label]
        # batch_answer.shape = [bz, mask_num (default 20)] ; [bz, span_len]
        # batch_span_mask.shape = [bz, span_len + 1 ([CLS])] ; [bz, span_len]
        answer_logits = answer_logits.masked_fill((1 - batch_span_mask.unsqueeze(1)).bool(), -10000.0)
        if self.config_for_model.RE_no_answer_reweighting_in_BF == 0:
            loss_fc_RE = CrossEntropyLoss(reduction="sum")
        else:
            w = torch.ones(answer_logits.shape[-1])
            w[0] = self.config_for_model.RE_no_answer_reweighting_in_BF
            loss_fc_RE = CrossEntropyLoss(reduction="sum", weight=w.to(device))
        active_logits = answer_logits.view(-1, answer_logits.shape[-1])
        RE_loss = loss_fc_RE(active_logits, batch_answer.view(-1))
        # RE_loss = RE_loss.sum()
        return RE_loss

    def get_RE_loss_for_RD(self, class_logits, head_logits, tail_logits, device, data_list):
        bsz = class_logits.shape[0]
        pred_relation_logits = class_logits.flatten(0, 1).softmax(-1)
        pred_head_logits = head_logits.flatten(0, 1).softmax(-1)
        pred_tail_logits = tail_logits.flatten(0, 1).softmax(-1)

        gold_rel = torch.cat([sample["relation_id"].to(device) for sample in data_list])
        gold_head = torch.cat([sample["head_entity_pos"].to(device) for sample in data_list])
        gold_tail = torch.cat([sample["tail_entity_pos"].to(device) for sample in data_list])

        if len(gold_rel) != 0:
            gold_rel = gold_rel.type(torch.LongTensor)
            gold_head = gold_head.type(torch.LongTensor)
            gold_tail = gold_tail.type(torch.LongTensor)
            cost = - self.config_for_model.RD_cost_relation * pred_relation_logits[:, gold_rel] \
                   - self.config_for_model.RD_cost_head * pred_head_logits[:, gold_head] \
                   - self.config_for_model.RD_cost_tail * pred_tail_logits[:, gold_tail]
            cost = cost.view(bsz, self.config_for_model.queries_num_for_RD, -1).cpu()
            num_gold_triplets = [len(sample["relation_id"]) for sample in data_list]
            indices = []
            batch_idx_with_relation = []
            batch_idx_without_relation = []
            samples_with_relation = []
            for i, c in enumerate(cost.split(num_gold_triplets, -1)):
                if c[i].shape[-1] != 0:
                    indices.append(linear_sum_assignment(c[i].detach().numpy()))
                    # batch_idx_with_relation_num.extend([i] * c[i].shape[-1])
                    batch_idx_with_relation.append(i)
                    samples_with_relation.append(data_list[i])
                else:
                    batch_idx_without_relation.append(i)
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triplets, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                       for i, j in indices]

            loss_for_rel_at_least_one, loss_for_head_in_RD, loss_for_tail_in_RD = \
                self.triplet_at_least_one_loss(samples_with_relation, indices,
                                               class_logits[batch_idx_with_relation],
                                               head_logits[batch_idx_with_relation],
                                               tail_logits[batch_idx_with_relation], device)
            loss_for_rel_without_relation = self.no_triplet_RD_loss(class_logits[batch_idx_without_relation], device)

            loss_for_rel_in_RD = loss_for_rel_at_least_one + loss_for_rel_without_relation
            loss_for_RD = self.config_for_model.RD_cost_relation * loss_for_rel_in_RD \
                          + self.config_for_model.RD_cost_head * loss_for_head_in_RD \
                          + self.config_for_model.RD_cost_tail * loss_for_tail_in_RD
        else:
            loss_for_RD = self.no_triplet_RD_loss(class_logits, device)

        return loss_for_RD

    def get_src_idx_permutation(self, indices):
        # batch_idx = torch.as_tensor(batch_idx_with_relation, dtype=torch.int64)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def no_triplet_RD_loss(self, class_logits, device):
        target_classes = torch.full(class_logits.shape[:2], 0, dtype=torch.int64, device=class_logits.device)
        if self.config_for_model.no_rel_reweighting_in_RD == 0:
            loss_fc_rel_in_RD = CrossEntropyLoss(reduction="sum")
        else:
            w = torch.ones(class_logits.shape[-1])
            w[0] = self.config_for_model.no_rel_reweighting_in_RD
            loss_fc_rel_in_RD = CrossEntropyLoss(reduction="sum", weight=w.to(device))
        loss_for_rel_in_RD = loss_fc_rel_in_RD(class_logits.flatten(0, 1), target_classes.flatten(0, 1))
        loss_for_RD = loss_for_rel_in_RD
        return loss_for_RD

    def triplet_at_least_one_loss(self, samples_with_relation, indices, class_logits, head_logits, tail_logits, device):
        idx = self.get_src_idx_permutation(indices)

        # relation loss
        target_classes_permutation = torch.cat([sample["relation_id"][trg].to(device)
                                                for sample, (_, trg) in zip(samples_with_relation, indices)])
        target_classes = torch.full(class_logits.shape[:2], 0, dtype=torch.int64, device=class_logits.device).to(device)
        target_classes[idx] = target_classes_permutation

        if self.config_for_model.no_rel_reweighting_in_RD == 0:
            loss_fc_rel_in_RD = CrossEntropyLoss(reduction="sum")
        else:
            w = torch.ones(class_logits.shape[-1])
            w[0] = self.config_for_model.no_rel_reweighting_in_RD
            loss_fc_rel_in_RD = CrossEntropyLoss(reduction="sum", weight=w.to(device))
        loss_for_rel_in_RD = loss_fc_rel_in_RD(class_logits.flatten(0, 1), target_classes.flatten(0, 1))

        # entity loss
        selected_pred_head = head_logits[idx]
        selected_pred_tail = tail_logits[idx]
        target_head = torch.cat([sample["head_entity_pos"][trg].to(device)
                                 for sample, (_, trg) in zip(samples_with_relation, indices)])
        target_tail = torch.cat([sample["tail_entity_pos"][trg].to(device)
                                 for sample, (_, trg) in zip(samples_with_relation, indices)])
        loss_fc_head_in_RD = CrossEntropyLoss(reduction="sum")
        loss_fc_tail_in_RD = CrossEntropyLoss(reduction="sum")
        loss_for_head_in_RD = loss_fc_head_in_RD(selected_pred_head, target_head)
        loss_for_tail_in_RD = loss_fc_tail_in_RD(selected_pred_tail, target_tail)
        return loss_for_rel_in_RD, loss_for_head_in_RD, loss_for_tail_in_RD






