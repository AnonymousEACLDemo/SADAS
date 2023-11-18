import argparse
import copy
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as st
import sys
import random
import json
import os

from ccu import CCU

sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(sys_path)

from PrefixTuning.transformers.src.transformers import AutoConfig
from PrefixTuning.transformers.examples.control.run_generation_clean import load_prefix_model, \
    generate_privacy_topic_instances, generate_optimus_instances
from PrefixTuning.transformers.examples.control.run_language_modeling_clean import train_prefix, initilize_gpt2, \
    ModelArguments, DataTrainingArguments, TrainingArguments
from lifelong.model.module import topic_lstm_layer, label_lstm_layer
from data_loader import get_data_loader
# -------------------------------------------------
import os
import logging

emar_prefix = os.path.dirname(os.path.abspath(__file__))

# f = open(emar_prefix + "config/config.json", "r")
# config = json.loads(f.read())

# print (torch.cuda.is_available())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['TRANSFORMERS_CACHE'] = os.path.join(sys_path, '.cache/huggingface')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import logging
import os.path
from functools import reduce

import zmq

from generate_privacy_examples import load_rewrite_model, generate_example


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data

def recognize_norm_rule(norm_detection_model, context, norm_category, norm_rules):
    rule_id = str(norm_detection_model.execute(context, norm_category))
    logging.info('Detected norm rule ID is: %s' % rule_id)
    intervention_rule = norm_rules[rule_id] if rule_id in norm_rules else None
    return intervention_rule

def select_responses(response_ranking_model, context, reduced_generated_response):
    return response_ranking_model.execute(context[-3:], reduced_generated_response)


def post_processing(generated_response):
    repls = (' ', ''), ('[SEP]', ''), ('[CLS]', '')
    reduced_generated_response = [reduce(lambda a, kv: a.replace(*kv), repls, x) for x in generated_response]
    return reduced_generated_response

if __name__ == "__main__":

    # PROJECT_ABSOLUTE_PATH = '/home/zlii0182/PycharmProjects'
    #
    # print(PROJECT_ABSOLUTE_PATH)


    gpt2_model_path = "models/gpt2"

    rewrite_model_path = "models/disentanglement_rewrite"

    bert_model_path = "models/bert"

    device = 'cpu'

    #print("rewrite_model_path", rewrite_model_path)
    #print(gpt2_model_path)

    prefix_model, sentence_tokenizer, config, label_embeds, generation_args = load_rewrite_model(rewrite_model_path, gpt2_model_path, bert_model_path, device)

    logging.info('Rewriting model is loaded!')

    logging.info('Response ranking model is loaded!')

    # utterance_to_rewrite = "李工，想让你帮我看看这个设计是不是合理。"
    # -------------------------------------------------------

    flag = True

    rewrite_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    rewrite_result_queue = CCU.socket(CCU.queues['CONTROL'], zmq.PUB)
    utterance_to_rewrite = ""

    while True:
        # Waits for an offensive message on the result queue.
        # For all such messages, it sends a text_popup message so that the
        # Hololens will notify the operator to not do that.
        if flag:
            print("INTO WHILE LOOP")
            flag = False

            # utterance_to_rewrite = "李工，想让你帮我看看这个设计是不是合理。"
            # generated_response = generate_example(utterance_to_rewrite, generation_args, prefix_model, label_embeds,
            #                                       config, sentence_tokenizer,
            #                                       gpt2_model_path)  # generate_model.gpt2_causal_text_generation(context)
            # reduced_generated_response = post_processing(generated_response)
            # logging.info('The generation model composes the response: {}'.format(reduced_generated_response))

        rewrite_message = CCU.recv(rewrite_queue)
        if rewrite_message and rewrite_message['type'] == 'offensive_speech':
            logging.info("offensive speech: {}".format(rewrite_message))
            utterance_to_rewrite = rewrite_message['input_text']
            generated_response = generate_example(utterance_to_rewrite, generation_args, prefix_model, label_embeds,
                                                  config,
                                                  sentence_tokenizer, gpt2_model_path)
            reduced_generated_response = post_processing(generated_response)
            message = CCU.base_message('translation')
            message['translation'] = reduced_generated_response
            logging.info(message)
            CCU.send(rewrite_result_queue, message)




