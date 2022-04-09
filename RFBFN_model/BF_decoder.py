# -*- coding : utf-8 -*-
import torch
from torch import nn
from allennlp.nn.util import batched_index_select
from .Bert_Achieved import Bert_for_BF

from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers import BertTokenizer, BertPreTrainedModel
import sys
sys.path.append("..")


class BF_Decoder(BertPreTrainedModel):
    def __init__(self, config, config_for_model):
        super().__init__(config)
        self.config_for_model = config_for_model
        self.config = config

        self.embeddings = BertEmbeddings(config)
        if config_for_model.fix_bert_embeddings:
            self.embeddings.word_embeddings.weight.requires_grad = False
            self.embeddings.position_embeddings.weight.requires_grad = False
            self.embeddings.token_type_embeddings.weight.requires_grad = False

        self.encoder = Bert_for_BF(config, config_for_model=config_for_model)

        self.linear_for_span1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_for_span2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_for_span3 = nn.Linear(config.hidden_size, 1, bias=False)

        self.init_weights()

        torch.nn.init.orthogonal_(self.linear_for_span1.weight, gain=1)
        torch.nn.init.orthogonal_(self.linear_for_span2.weight, gain=1)

    def forward(self, batch_tokens_attention_mask, batch_span_mask, logits_for_RE, sequence_output_of_encoder,
                batch_question_tokens, batch_question_tokens_attention_mask, batch_decoder_mask_pos, batch_answer,
                token_type_ids):

        # device = batch_question_tokens.device
        if self.config_for_model.cross_attention_mode_for_BF == "2":
            batch_encoder_attention_mask = batch_span_mask
        else:
            batch_encoder_attention_mask = batch_tokens_attention_mask

        encoder_extended_attention_mask = self.get_extended_encoder_or_decoder_mask(batch_encoder_attention_mask)
        extended_attention_mask = self.get_extended_encoder_or_decoder_mask(batch_question_tokens_attention_mask)

        embedding_output = self.embeddings(input_ids=batch_question_tokens, token_type_ids=token_type_ids)

        hidden_states = embedding_output
        question_sequence_output = self.encoder(hidden_states, extended_attention_mask, sequence_output_of_encoder,
                                                encoder_extended_attention_mask)
        # question_sequence_output.shape = [bz, question_seq_len, config.hidden_size(768)]
        answer_logits = self.match_answer(logits_for_RE, question_sequence_output, batch_decoder_mask_pos)
        return answer_logits

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

    def match_answer(self, logits_for_RE, question_sequence_output, batch_decoder_mask_pos):
        mask_logits = batched_index_select(question_sequence_output, batch_decoder_mask_pos)
        # mask_logits = [bz, mask_len, 768]  logits_for_RE = [bz, span_len + 1, 768]  ====> [bz, mask_len, span_len + 1]
        if self.config_for_model.answer_attention == "add":
            answer_logits = self.linear_for_span3(torch.tanh(
                self.linear_for_span1(mask_logits).unsqueeze(2) +
                self.linear_for_span2(logits_for_RE).unsqueeze(1))).squeeze(-1)
        else:
            answer_logits = torch.bmm(
                torch.tanh(self.linear_for_span1(mask_logits)),
                torch.tanh(self.linear_for_span2(logits_for_RE)).transpose(1, 2))
        # answer_logits.shape = [bz,  mask_num (default 20), span_len + 1 ([CLS])]
        return answer_logits





