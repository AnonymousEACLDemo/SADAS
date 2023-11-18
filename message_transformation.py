"""
This module contains functions for transforming the TA2 messages into the TA1 messages.

Functions:
    read_jsonl: read jsonl file
    write_into_file: write the extracted and converted contents into the result txt file
    transform_message: extract the contents from TA2 and convert them following the TA1 messages' format.

Author: Yuncheng (Devin) Hua <devin.hua@monash.edu>
Date: March 1, 2023
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'


def read_jsonl(path):
    with open(path, 'r', encoding="UTF-8") as load_f:
        lines = load_f.readlines()
    return [line for line in lines if line[0] != '#']


def write_into_file(path, lines):
    path_ = str(path).replace('.jsonl', '_transformation.txt')
    # path_ = path
    if not os.path.exists(os.path.dirname(path_)):
        os.makedirs(os.path.dirname(path_))

    with open(path_, 'w', encoding="UTF-8") as fw:
        for line in lines:
            fw.write(line + '\n')

    logging.info('The transformed file is written into %s', path_)

def create_df(columns):
    # Create the DataFrame
    df_ = pd.DataFrame(columns=columns)
    return df_

def get_start_end(dict_):
    start_value, end_value = 0, 0
    if 'start_seconds' not in dict_['message'] and 'start_chars' not in dict_['message']:
        start_value = 0
    else:
        if 'start_seconds' in dict_['message']:
            start_value = dict_['message']['start_seconds']
        elif 'start_chars' in dict_['message']:
            start_value = dict_['message']['start_chars']

    if 'end_seconds' not in dict_['message'] and 'end_chars' not in dict_['message']:
        end_value = 0
    else:
        if 'end_seconds' in dict_['message']:
            end_value = dict_['message']['end_seconds']
        elif 'end_chars' in dict_['message']:
            end_value = dict_['message']['end_chars']
    return start_value, end_value


def transform_message(dicts, file_name):
    res_lines = []

    norm_df = create_df(['file_id', 'speaker', 'norm', 'start', 'end', 'status', 'llr', 'container_name'])
    valence_df = create_df(['file_id', 'speaker', 'start', 'end', 'valence_continuous', 'container_name'])
    arousal_df = create_df(['file_id', 'speaker', 'start', 'end', 'arousal_continuous', 'container_name'])
    emotion_df = create_df(['file_id', 'speaker', 'emotion', 'start', 'end', 'llr', 'container_name'])
    change_point_df = create_df(['file_id', 'speaker', 'timestamp', 'llr', 'container_name'])

    for dict_ in dicts:
        if 'message' in dict_ and 'type' in dict_['message']:
            if dict_['message']['type'] == 'norm_occurrence':
                speaker_value = dict_['message']['speaker'] if 'speaker' in dict_['message'] else 'NA'
                norm_value = dict_['message']['name'] if 'name' in dict_['message'] else 'NA'
                status_value = dict_['message']['status'] if 'status' in dict_['message'] else 'NA'
                llr_value = dict_['message']['llr'] if 'llr' in dict_['message'] else -1.0
                container_name = dict_['message']['container_name'] if 'container_name' in dict_['message'] else 'NA'
                start_value, end_value = get_start_end(dict_)
                new_row = pd.DataFrame({'file_id': [file_name], 'speaker': [speaker_value], 'norm': [norm_value],
                                          'start': [start_value], 'end': [end_value], 'status': [status_value],
                                          'llr': [llr_value], 'container_name': [container_name]})
                norm_df = pd.concat([norm_df, new_row], ignore_index=True)

            elif dict_['message']['type'] == 'valence':
                speaker_value = dict_['message']['speaker'] if 'speaker' in dict_['message'] else 'NA'
                container_name = dict_['message']['container_name'] if 'container_name' in dict_['message'] else 'NA'
                valence_continuous_value = dict_['message']['level'] if 'level' in dict_['message'] else -1
                start_value, end_value = get_start_end(dict_)
                new_row = pd.DataFrame({'file_id': [file_name], 'speaker': [speaker_value], 'start': [start_value], 'end': [end_value],
                                                'valence_continuous': [valence_continuous_value], 'container_name': [container_name]})
                valence_df = pd.concat([valence_df, new_row], ignore_index=True)

            elif dict_['message']['type'] == 'arousal':
                speaker_value = dict_['message']['speaker'] if 'speaker' in dict_['message'] else 'NA'
                container_name = dict_['message']['container_name'] if 'container_name' in dict_['message'] else 'NA'
                arousal_continuous_value = dict_['message']['level'] if 'level' in dict_['message'] else -1
                start_value, end_value = get_start_end(dict_)
                new_row = pd.DataFrame({'file_id': [file_name], 'speaker': [speaker_value], 'start': [start_value], 'end': [end_value],
                                                'arousal_continuous': [arousal_continuous_value], 'container_name': [container_name]})
                arousal_df = pd.concat([arousal_df, new_row], ignore_index=True)

            elif dict_['message']['type'] == 'emotion':
                speaker_value = dict_['message']['speaker'] if 'speaker' in dict_['message'] else 'NA'
                emotion_value = dict_['message']['name'] if 'name' in dict_['message'] else 'NA'
                container_name = dict_['message']['container_name'] if 'container_name' in dict_['message'] else 'NA'
                llr_value = dict_['message']['llr'] if 'llr' in dict_['message'] else -1.0
                start_value, end_value = get_start_end(dict_)
                new_row = pd.DataFrame({'file_id': [file_name], 'speaker': [speaker_value], 'emotion': [emotion_value],
                                          'start': [start_value], 'end': [end_value], 'llr': [llr_value], 'container_name': [container_name]})
                emotion_df = pd.concat([emotion_df, new_row], ignore_index=True)

            elif dict_['message']['type'] == 'change_point':
                speaker_value = dict_['message']['speaker'] if 'speaker' in dict_['message'] else 'NA'
                container_name = dict_['message']['container_name'] if 'container_name' in dict_['message'] else 'NA'
                timestamp_value = -1.0
                if 'timestamp' in dict_['message']:
                    timestamp_value = dict_['message']['timestamp']
                elif 'chars' in dict_['message']:
                    timestamp_value = dict_['message']['chars']
                llr_value = dict_['message']['llr'] if 'llr' in dict_['message'] else -1.0
                # direction_value = dict_['message']['direction'] if 'direction' in dict_['message'] else 'NA'
                new_row = pd.DataFrame({'file_id': [file_name], 'speaker': [speaker_value], 'timestamp': [timestamp_value], 'llr': [llr_value], 'container_name': [container_name]})
                change_point_df = pd.concat([change_point_df, new_row], ignore_index=True)

    for df_ in [norm_df, valence_df, arousal_df, emotion_df, change_point_df]:
        # Convert the DataFrame to a string with tab-separated values
        csv_str = df_.to_csv(index=False, sep='\t')

        # Split the string into lines
        lines = csv_str.split('\n')

        res_lines.extend(lines)

    return res_lines


if __name__ == "__main__":
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    # command line parameters
    # sys.argv = ['message_transformation.py', '-l=LC1_OP2_MON_A0103_FLE01_SME04_20221115_normalization.jsonl']
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", required=True, help="Load the log file for further transformation!")
    args = parser.parse_args()

    path_ = PROJECT_ABSOLUTE_PATH + args.load
    lines = read_jsonl(path_)
    dicts = [json.loads(line) for line in lines]

    logging.info("Load file from %s, continue transformation...", path_)

    res_lines = transform_message(dicts, path_.split('/')[-1].split('.')[0])

    write_into_file(path_, res_lines)