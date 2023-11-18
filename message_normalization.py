import argparse
import json
import logging
import os
import sys

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'


def read_jsonl(path):
    with open(path, 'r', encoding="UTF-8") as load_f:
        lines = load_f.readlines()
    return [line for line in lines if line[0] != '#']


def write_into_jsonl(path, content):
    with open(path, 'r', encoding="UTF-8") as load_f:
        lines = [line for line in load_f.readlines() if line[0] == '#']
    content_lines = [json.dumps(dict_)+'\n' for dict_ in content]
    lines.extend(content_lines)

    path_ = str(path).replace('.jsonl', '_normalization.jsonl')
    # path_ = path
    if not os.path.exists(os.path.dirname(path_)):
        os.makedirs(os.path.dirname(path_))

    with open(path_, 'w', encoding="UTF-8") as fw:
        fw.writelines(lines)

    logging.info('The normalized file is written into %s', path_)


if __name__ == "__main__":
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    # command line parameters
    # sys.argv = ['message_normalization.py', '-l=LC1_OP1_MON_E0223_FLE03_SME06_20221117.jsonl']
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", required=True, help="Load the log file for further extraction")
    args = parser.parse_args()

    path_ = PROJECT_ABSOLUTE_PATH + args.load
    lines = read_jsonl(path_)
    dicts = [json.loads(line) for line in lines]

    logging.info("Load file from %s, continue extracting...", path_)

    for dict_ in dicts:
        if 'message' in dict_ and 'type' in dict_['message'] and 'trigger_id' in dict_['message'] and dict_['message']['type'] == 'norm_occurrence':
            norm_dict = dict_
            trigger_id = norm_dict['message']['trigger_id'][0]
            for trigger_dict in dicts:
                if 'message' in trigger_dict and 'type' in trigger_dict['message'] and trigger_dict['message']['type'] == 'asr_result':
                    if 'uuid' in trigger_dict['message'] and trigger_dict['message']['uuid'] == trigger_id:
                        if 'start_seconds' in trigger_dict['message'] and 'end_seconds' in trigger_dict['message']:
                            norm_dict['message']['start_seconds'] = trigger_dict['message']['start_seconds']
                            norm_dict['message']['end_seconds'] = trigger_dict['message']['end_seconds']

        if 'message' in dict_ and 'type' in dict_['message'] and 'trigger_id' in dict_['message'] and dict_['message'][
            'type'] in ['arousal', 'valence', 'emotion']:
            emotion_dict = dict_
            # trigger_id should be a list, therefore in future we will change it into ['trigger_id'][0], corresponding to what we will change in the ta2.py script.
            # trigger_id = emotion_dict['message']['trigger_id'][0]
            trigger_id = emotion_dict['message']['trigger_id']
            for trigger_dict in dicts:
                if 'message' in trigger_dict and 'type' in trigger_dict['message'] and trigger_dict['message']['type'] == 'purdue_' + emotion_dict['message']['type']:
                    if 'uuid' in trigger_dict['message'] and trigger_dict['message']['uuid'] == trigger_id:
                        if 'originial_text' in trigger_dict['message']:
                            original_text = trigger_dict['message']['originial_text']
                            for asr_dict in dicts:
                                if 'message' in asr_dict and 'type' in asr_dict['message'] and asr_dict['message']['type'] == 'asr_result':
                                    if 'asr_text' in asr_dict['message'] and asr_dict['message'][
                                        'asr_text'] == original_text:
                                        if 'start_seconds' in asr_dict['message'] and 'end_seconds' in asr_dict[
                                            'message']:
                                            emotion_dict['message']['start_seconds'] = asr_dict['message'][
                                                'start_seconds']
                                            emotion_dict['message']['end_seconds'] = asr_dict['message']['end_seconds']
                                            emotion_dict['message']['original_text'] = original_text

    # check
    for dict_ in dicts:
        if 'message' in dict_ and 'type' in dict_['message'] and dict_['message']['type'] == 'norm_occurrence':
            if 'start_seconds' in dict_['message'] and 'end_seconds' in dict_['message']:
                pass
            else:
                logging.info('The norm message is wrong.')
                logging.info(dict_)

        if 'message' in dict_ and 'type' in dict_['message'] and dict_['message']['type'] in ['arousal', 'valence', 'emotion']:
            if 'start_seconds' in dict_['message'] and 'end_seconds' in dict_['message']:
                pass
            else:
                logging.info('The arousal/valence/emotion message is wrong.')
                logging.info(dict_)

    write_into_jsonl(path_, dicts)