# -*- coding : utf-8 -*-
import torch
from torch import nn

from .Bert_Achieved import Bert_for_RD
from data_preprocess import nyt_rel_labels_constant, webnlg_rel_labels_constant, webnlg_rel_labels_constant_no_star

import sys
sys.path.append("..")
BertLayerNorm = torch.nn.LayerNorm


class RD_Decoder_no_pretrain(nn.Module):
    def __init__(self, config, config_for_model):
        super().__init__()
        self.config_for_model = config_for_model
        self.config = config
        self.encoder = Bert_for_RD(config, config_for_model=config_for_model)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(config_for_model.queries_num_for_RD, config.hidden_size)

        if config_for_model.task == "webnlg":
            if config_for_model.star == 1:
                num_relation_label = len(webnlg_rel_labels_constant) + 1
            else:
                num_relation_label = len(webnlg_rel_labels_constant_no_star) + 1
        else:
            num_relation_label = len(nyt_rel_labels_constant) + 1

        self.decoder2class = nn.Linear(config.hidden_size, num_relation_label)
        self.head_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        self.tail_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, batch_tokens_attention_mask, batch_span_mask, sequence_output_of_encoder, logits_for_RD):

        bsz = batch_span_mask.shape[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))

        if self.config_for_model.cross_attention_mode_for_RD == "2":
            batch_encoder_attention_mask = batch_span_mask
        else:
            batch_encoder_attention_mask = batch_tokens_attention_mask

        encoder_extended_attention_mask = self.get_extended_encoder_or_decoder_mask(batch_encoder_attention_mask)

        RD_decoder_output = self.encoder(hidden_states, sequence_output_of_encoder, encoder_extended_attention_mask)

        class_logits = self.decoder2class(RD_decoder_output)
        head_logits = self.head_metric_3(torch.tanh(
            self.head_metric_1(RD_decoder_output).unsqueeze(2) +
            self.head_metric_2(logits_for_RD).unsqueeze(1))).squeeze(-1)
        tail_logits = self.tail_metric_3(torch.tanh(
            self.tail_metric_1(RD_decoder_output).unsqueeze(2) +
            self.tail_metric_2(logits_for_RD).unsqueeze(1))).squeeze(-1)

        head_logits = head_logits.masked_fill((1 - batch_span_mask.unsqueeze(1)).bool(), -10000.0)
        tail_logits = tail_logits.masked_fill((1 - batch_span_mask.unsqueeze(1)).bool(), -10000.0)

        return class_logits, head_logits, tail_logits

    def get_extended_encoder_or_decoder_mask(self, attention_mask):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable
            # to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        # extended_attention_mask = extended_attention_mask.to(torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask








