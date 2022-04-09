# -*- coding : utf-8 -*-

import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="webnlg", help="nyt webnlg")
    parser.add_argument('--star', type=int, default=1,  help="whether to choose dataset")

    parser.add_argument('--pretrained_model', type=str, default='bert-base-cased',
                        help="bert-base-cased, bert-base-uncased")
    parser.add_argument('--num_decoder_layers_for_RD', type=int, default=4)
    parser.add_argument('--queries_num_for_RD', type=int, default=12, help="num of queries for SPN decoder")

    parser.add_argument('--save_path', type=str, default="./data_preprocess/processed_data/",
                        help="path to the saved data")
    parser.add_argument('--log_path', type=str, default="./log/",
                        help="path to the log")
    parser.add_argument('--model_save_path', type=str, default="./save_model/",
                        help="path to save the model")
    parser.add_argument('--gpu_setting', type=str, default="0",
                        help="visible GPU, used in os.environ setting, if -1 means all gpu")

    parser.add_argument("--epoch", type=int, default=150, help="epoch of training")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--is_single_batch', type=int, default=0,
                        help="whether long sentence in single batch")

    parser.add_argument('--max_span_length', type=int, default=8,
                        help="length up to max_span_length are considered as candidates")
    parser.add_argument('--width_embedding_dim', type=int, default=150, help="embedding dim of span width")
    parser.add_argument('--hidden_dim_for_NER', type=int, default=150, help="hidden dim for NER")
    parser.add_argument('--hidden_dim_for_select', type=int, default=150, help="hidden dim for NER")

    parser.add_argument('--cross_attention_mode_for_BF', type=str, default="2",
                        help="mode 1 refers to use the sequence embedding of the encoder as the decoder input; "
                             "mode 2 refers to use the span embedding of the encoder as the decoder input")
    parser.add_argument('--cross_attention_mode_for_RD', type=str, default="2",
                        help="mode 1 refers to use the sequence embedding of the encoder as the decoder input; "
                             "mode 2 refers to use the span embedding of the encoder as the decoder input")

    parser.add_argument('--NER_mode', type=str, default="1",
                        help="mode 1 refers to use the same downstream parameters as RE; "
                             "mode 2 refers to not use the same downstream parameters as RE")

    parser.add_argument('--NER_loss_parameter', type=int, default=1,
                        help="loss = NER_loss_parameter * NER_loss + RE_loss_parameter * RE_loss")
    parser.add_argument('--RE_loss_for_BF_parameter', type=int, default=10,
                        help="loss = NER_loss_parameter * NER_loss + RE_loss_parameter * RE_loss")
    parser.add_argument('--RE_loss_for_RD_parameter', type=int, default=5,
                        help="loss = NER_loss_parameter * NER_loss + RE_loss_parameter * RE_loss")
    parser.add_argument('--RD_cost_relation', type=int, default=1)
    parser.add_argument('--RD_cost_head', type=int, default=2)
    parser.add_argument('--RD_cost_tail', type=int, default=2)

    parser.add_argument('--NER_none_span_reweighting_in_BF', type=float, default=0.5,
                        help="CrossEntropyLoss, see pytorch doc for help, 0 for no reweighting")
    parser.add_argument('--RE_no_answer_reweighting_in_BF', type=float, default=0.6,
                        help="CrossEntropyLoss, see pytorch doc for help, 0 for no reweighting")
    parser.add_argument('--no_rel_reweighting_in_RD', type=float, default=0.4,
                        help="CrossEntropyLoss, see pytorch doc for help, 0 for no reweighting")

    parser.add_argument('--token_type_ids', type=int, default=0, help="whether to use 0, 1 for token type ids")
    parser.add_argument('--answer_attention', type=str, default="add", help="add or dot for answer attention")

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate_for_bert_encoder', type=float, default=1e-5,
                        help="learning rate for the BERT encoder")
    parser.add_argument('--learning_rate_for_NER_encoder', type=float, default=1e-4,
                        help="learning rate for NER")


    parser.add_argument('--learning_rate_for_bert_decoder_in_BF', type=float, default=1e-5,
                        help="learning rate for the BERT decoder")
    parser.add_argument('--learning_rate_for_RE_decoder_in_BF', type=float, default=1e-4,
                        help="learning rate for RE")
    parser.add_argument('--learning_rate_for_crossattention_in_BF', type=float, default=5e-5,
                        help="learning rate for crossattention")
    parser.add_argument('--learning_rate_in_RD', type=float, default=2e-5,
                        help="learning rate for SPN decoder")

    parser.add_argument('--lr_mode', type=str, default="2",
                        help="mode 1 refers to lr decay; mode 2 refers to warmup")
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_decay', type=float, default=0.01, help="lr_mode == 1")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help="lr_mode == 2; the ratio of the warmup steps to the total steps")
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fix_bert_embeddings', type=int, default=0)

    parser.add_argument('--print_loss_step', type=int, default=100)

    # SELECT
    parser.add_argument("--epoch_select", type=int, default=20, help="epoch of training")
    parser.add_argument('--learning_rate_for_SELECT_encoder', type=float, default=1e-4,
                        help="learning rate for SELECTION")
    parser.add_argument('--Select_none_span_reweighting_in_BF', type=float, default=0.5,
                        help="CrossEntropyLoss, see pytorch doc for help, 0 for no reweighting")
    # generate_step
    parser.add_argument('--generate_step', type=str, default="1", help="1 for step1; 2 for step 2")

    return parser.parse_args()



