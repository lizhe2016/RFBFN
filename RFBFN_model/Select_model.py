# -*- coding : utf-8 -*-
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer

from allennlp.nn.util import batched_index_select
from allennlp.modules import FeedForward
from transformers import BertPreTrainedModel, BertModel
import sys
sys.path.append("..")
from data_preprocess import nyt_rel_labels_constant, webnlg_rel_labels_constant, webnlg_rel_labels_constant_no_star

import logging
logger = logging.getLogger('root')


class RFBFN_Model_SELECT(nn.Module):
    def __init__(self, config_for_RFBFN):
        super(RFBFN_Model_SELECT, self).__init__()
        self.config_for_model = config_for_RFBFN
        pretrained_model = self.config_for_model.pretrained_model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        self.model = Model_ALL_SELECT(config_for_RFBFN)
        self._model_device = "cpu"
        self.share_parameter()
        self.move_model_to_cuda()

    def forward(self, data_list, class_logits):
        batch_tokens, batch_tokens_attention_mask, batch_spans, batch_span_labels, batch_span_mask, \
        batch_class_select_labels = self.get_mask(data_list, class_logits)

        output_dict = {}
        loss, select_logits = \
            self.model(batch_tokens.to(self._model_device), batch_tokens_attention_mask.to(self._model_device),
                       batch_spans.to(self._model_device), batch_span_mask.to(self._model_device),
                       batch_class_select_labels.to(self._model_device), class_logits.to(self._model_device),
                       self._model_device, data_list)

        output_dict["loss_total"] = loss
        output_dict["select_logits"] = select_logits
        output_dict["select_labels"] = batch_class_select_labels
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

    def share_parameter(self):
        parameter_in_SPN = torch.load(self.config_for_model.SPN_parameter_sharing_path)
        model_dict = self.model.state_dict()
        sharing_dict = {}
        for k, v in parameter_in_SPN.items():
            if "bert" in k:
                sharing_dict[k.strip(r"model").strip(r".")] = v
        model_dict.update(sharing_dict)
        self.model.load_state_dict(model_dict)

    def get_mask(self, data_list, class_logits):
        max_tokens = 0
        max_spans = 0
        _, predicted_relation_labels = class_logits.softmax(-1).max(2)

        batch_tokens_final = None
        batch_tokens_attention_mask_final = None
        batch_spans_final = None
        batch_span_labels_final = None
        batch_span_mask_final = None
        # [bz, triple_num]
        batch_class_select_labels = None

        for sample in data_list:
            if sample["id"].shape[0] > max_tokens:
                max_tokens = sample["id"].shape[0]
            if sample["spans"].shape[0] > max_spans:
                max_spans = sample["spans"].shape[0]

        for i, sample in enumerate(data_list):
            current_select_label = []
            for triple_id in range(predicted_relation_labels.shape[1]):
                if predicted_relation_labels[i][triple_id] in sample["relation_id"].cuda():
                    current_select_label.append(1)
                else:
                    current_select_label.append(0)
            current_select_label = torch.tensor(current_select_label).unsqueeze(0)

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

            if batch_tokens_final == None:
                batch_tokens_final = final_tokens
                batch_tokens_attention_mask_final = token_attention_mask
                batch_spans_final = final_spans
                batch_span_labels_final = final_span_labels
                batch_span_mask_final = spans_mask
                batch_class_select_labels = current_select_label
            else:
                batch_tokens_final = torch.cat((batch_tokens_final, final_tokens), dim=0)
                batch_tokens_attention_mask_final = torch.cat((batch_tokens_attention_mask_final, token_attention_mask), dim=0)
                batch_spans_final = torch.cat((batch_spans_final, final_spans), dim=0)
                batch_span_labels_final = torch.cat((batch_span_labels_final, final_span_labels), dim=0)
                batch_span_mask_final = torch.cat((batch_span_mask_final, spans_mask), dim=0)
                batch_class_select_labels = torch.cat((batch_class_select_labels, current_select_label), dim=0)
        return batch_tokens_final, batch_tokens_attention_mask_final, batch_spans_final, batch_span_labels_final, \
               batch_span_mask_final, batch_class_select_labels


class Model_ALL_SELECT(nn.Module):
    def __init__(self, config_for_model):
        super(Model_ALL_SELECT, self).__init__()
        self.config_for_model = config_for_model
        pretrained_model = self.config_for_model.pretrained_model

        if config_for_model.task == "webnlg":
            if config_for_model.star == 1:
                num_rel_label = len(webnlg_rel_labels_constant) + 1
            else:
                num_rel_label = len(webnlg_rel_labels_constant_no_star) + 1
        else:
            num_rel_label = len(nyt_rel_labels_constant) + 1

        self.encoder = Encoder_select.from_pretrained(pretrained_model, config_for_model=config_for_model,
                                                          num_rel_label=num_rel_label)

    def forward(self, batch_tokens, batch_tokens_attention_mask, batch_spans, batch_span_mask,
                batch_class_select_labels, class_logits, device, data_list):
        logits_for_select = self.encoder(batch_tokens, batch_tokens_attention_mask, batch_spans, class_logits)
        # [bz, num_triple, 1]
        # [bz, num_triple]
        select_loss = self.get_select_loss(logits_for_select, batch_class_select_labels, device)
        return select_loss, logits_for_select

    def get_select_loss(self, logits_for_select, batch_class_select_labels, device):
        if self.config_for_model.Select_none_span_reweighting_in_BF == 0:
            loss_fc_SELECT = CrossEntropyLoss(reduction="sum")
        else:
            w = torch.ones(2)
            w[0] = self.config_for_model.Select_none_span_reweighting_in_BFT
            loss_fc_SELECT = CrossEntropyLoss(reduction="sum", weight=w.to(device))
        logits_for_select = logits_for_select.view(-1, logits_for_select.shape[-1])
        select_loss = loss_fc_SELECT(logits_for_select, batch_class_select_labels.view(-1))
        # NER_loss = NER_loss.sum()
        return select_loss


class Encoder_select(BertPreTrainedModel):
    def __init__(self, config, config_for_model, num_rel_label):
        super().__init__(config)
        self.config_for_model = config_for_model

        self.bert = BertModel(config)
        if config_for_model.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(config_for_model.max_span_length + 1, config_for_model.width_embedding_dim)
        self.select_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size + num_rel_label,
                        num_layers=2,
                        hidden_dims=config_for_model.hidden_dim_for_select,
                        activations=torch.nn.ReLU(),
                        dropout=0.2),
            nn.Linear(config_for_model.hidden_dim_for_NER, 2)
        )
        self.init_weights()

    def forward(self, batch_tokens, batch_tokens_attention_mask, batch_spans, class_logits):
        batch_spans_embedding, sequence_output_of_encoder, pooled_output = \
            self.get_spans_embedding(batch_tokens, batch_tokens_attention_mask, batch_spans)

        # pooled_output.shape = [bz, config.hidden_size(768)]
        # class_logits.shape = [bz, num_triple, 171]
        pooled_output = pooled_output.unsqueeze(1).repeat(1, class_logits.shape[1], 1)
        logits_for_select = torch.cat((pooled_output, class_logits), dim=-1)

        FFNN_hidden = []
        hidden = logits_for_select
        for layer_each in self.select_classifier:
            hidden = layer_each(hidden)
            FFNN_hidden.append(hidden)
        # [bz, num_triple, 1]
        logits_for_select = FFNN_hidden[-1]
        # logits_select_1 = torch.where(logits_for_select > torch.tensor(self.config_for_model.threshold).cuda(), 1, 0)
        return logits_for_select

    def get_spans_embedding(self, batch_tokens, batch_tokens_attention_mask, batch_spans):
        sequence_output, pooled_output = self.bert(input_ids=batch_tokens, attention_mask=batch_tokens_attention_mask,
                                                   return_dict=False)
        sequence_output_drop = self.hidden_dropout(sequence_output)
        batch_spans_start = batch_spans[:, :, 0].view(batch_spans.size(0), -1)
        batch_spans_start_embedding = batched_index_select(sequence_output_drop, batch_spans_start)

        batch_spans_end = batch_spans[:, :, 1].view(batch_spans.size(0), -1)
        batch_spans_end_embedding = batched_index_select(sequence_output_drop, batch_spans_end)

        batch_width = batch_spans[:, :, 2].view(batch_spans.size(0), -1)
        batch_width_embedding = self.width_embedding(batch_width)

        batch_spans_embedding = torch.cat((batch_spans_start_embedding, batch_spans_end_embedding,
                                           batch_width_embedding), dim=-1)
        return batch_spans_embedding, sequence_output, pooled_output








