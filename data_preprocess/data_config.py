# -*- coding : utf-8 -*-

import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default="1", help="1 for step1; 2 for step 2")
    parser.add_argument('--pretrained_model', type=str, default='bert-base-cased',
                        help="bert-base-cased, bert-base-uncased")
    parser.add_argument('--task', type=str, default="webnlg",
                        help="nyt, webnlg")
    parser.add_argument('--star', type=int, default=1,
                        help="whether to choose dataset")

    parser.add_argument('--save_path', type=str, default="./processed_data/",
                        help="path to save data")
    parser.add_argument('--log_path', type=str, default="./processed_data/",
                        help="path to log")
    parser.add_argument('--max_span_length', type=int, default=8,
                        help="length up to max_span_length are considered as candidates")
    parser.add_argument('--negative_probability', type=float, default=0,
                        help="it defines how many negative relations with empty answers")
    parser.add_argument('--sorted_entity', type=int, default=1,
                        help="whether sort the head and the tail entity by the order in sentences")
    parser.add_argument('--sorted_relation', type=int, default=0,
                        help="whether sort relations by the specific order")
    parser.add_argument('--duplicate_questions', type=int, default=3,
                        help="the duplicate number for each question")
    parser.add_argument('--token_type_ids', type=int, default=1,
                        help="whether to use 0, 1 for token type ids")
    return parser.parse_args()
