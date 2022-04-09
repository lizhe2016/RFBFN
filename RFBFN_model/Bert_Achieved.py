# -*- coding : utf-8 -*-
from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertIntermediate, BertOutput, BertAttention


class Bert_for_RD(nn.Module):
    def __init__(self, config, config_for_model):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Bert_Layer_with_Cross_Attention(config) for _
                                    in range(config_for_model.num_decoder_layers_for_RD)])

    def forward(self, hidden_states, sequence_output_of_encoder,
                encoder_extended_attention_mask):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=None,
                encoder_hidden_states=sequence_output_of_encoder,
                encoder_attention_mask=encoder_extended_attention_mask
            )
            hidden_states = layer_outputs
        sequence_output = hidden_states

        return sequence_output


class Bert_for_BF(nn.Module):
    def __init__(self, config, config_for_model):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Bert_Layer_with_Cross_Attention(config) for _
                                    in range(config.num_hidden_layers)])

    def forward(self, hidden_states, extended_attention_mask, sequence_output_of_encoder,
                encoder_extended_attention_mask):

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=sequence_output_of_encoder,
                encoder_attention_mask=encoder_extended_attention_mask
            )
            hidden_states = layer_outputs
        sequence_output = hidden_states

        return sequence_output


class Bert_Layer_with_Cross_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        self_attention_outputs = self.attention(hidden_states=hidden_states, attention_mask=attention_mask)
        attention_output = self_attention_outputs[0]
        cross_attention_outputs = self.crossattention(hidden_states=attention_output,
                                                      encoder_hidden_states=encoder_hidden_states,
                                                      encoder_attention_mask=encoder_attention_mask)
        attention_output = cross_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

