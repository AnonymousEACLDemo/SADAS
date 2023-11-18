#! /usr/bin/env python
# This script filters jsonl logs in various ways.

from argparse import RawDescriptionHelpFormatter
import argparse
import base64
import json
import logging
import sys
import time

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

first = True
summary = {"message_count": 0}

def make_human_readable(message):
    """Trims any string field which is longer than max_len characters (usually 60)
       down to 60 characters and adds a ... if it does so."""
    max_len = 60
    for key, value in message.items():
        if type(value) == str and len(value) > max_len:
            message[key] = value[:max_len] + " ..."

parser = argparse.ArgumentParser('Program to filter jsonl log files. A sort of "Swiss Army Knife" for jsonl files.', 
        formatter_class=RawDescriptionHelpFormatter,
        epilog='\nSome common uses for this program:\n'+
        '  * Get a quick messages count: -s\n'+
        '  * Convert to json so other tools can read the files: -u or -l\n'+
        '    Use -u to remove the queue and time information.\n'+
        '    Use -l to just convert from jsonl to json format.\n'+
        '  * Just see some queues: -q (example: -q RESULT,ACTION)\n'+
        '  * Just see some messages: -q (example: -q asr_result,emotion)\n'+
        '  * Make sure your jsonl file is formatted correctly: -n\n\n'+
        'Example command lines:\n'+
        '  Count the number of asr_result messages in the jsonl file:\n'+
        '    ./jsonlfilter.py -m asr_result -s t2.jsonl\n'+
        '  Convert the five messages messages that NIST looks at into a json\n'+
        '  file, and then format it (using python\'s json.tool module):\n'+
        '    ./jsonlfilter.py -m emotion,valence,arousal,change_point,norm_occurrence -u t2.jsonl | python -m json.tool\n')
parser.add_argument('-c', '--containers', type=str,
                    help = 'Names of containers you want to log, seperated by commas.  For example: --containers sri-whisper-en will only print out messages from that one container, while -c sri-whisper-en,columbia-norms will print out message from those two containers.  Do not put spaces around the commas.  Names must match exactly without any kind of wildcard characters or partial matching.  Queue filtering (the -q option) is done first.')
parser.add_argument('-e', '--exclude', action='store_true',
                    help = 'Exclude queues or messages from the output.  Used with the -q and -m options.')
parser.add_argument('-f', '--fields', action='store_true',
                    help = 'Print out messsage field names but not content.')
parser.add_argument('infile', type=argparse.FileType('r', encoding='utf-8'), nargs='?', default=sys.stdin,
                    help = 'The file to use.  Stdin is used if no file is given.')
parser.add_argument('-i', '--ids', type=str,
                    help = 'UUIDs of messasges you want to log, seperated by commas.  For example: --ids 1d2ac0da-cb08-488a-b2a4-b95ac54357c3 will print out just that one message.  You can print more than one with a comma seperated list.  Do not put spaces around the commas.  UUIDs must match exactly without any kind of wildcard characters or partial matching.  Queue filtering (the -q option) and message filtering (the -m option) are done first.')
parser.add_argument('-k', '--no_output', action='store_true',
                    help = 'Do not print jsonl files as output.  This is implied by the summarize option.')
parser.add_argument('-l', '--list', action='store_true',
        help = 'The result will be a list of json objects, not a jsonl file.  This option can be used to feed into json tools.  You can only choose one of these options: -s, -u, -l.')
parser.add_argument('-m', '--messages', type=str,
                    help = 'Names of messasges you want to log, seperated by commas.  For example: --messasges valence will print out valence messages only, while -m valence,norm_occurrence,hello will print out those three message types.  Do not put spaces around the commas.  Names must match exactly without any kind of wildcard characters or partial matching.  Queue filtering (the -q option) is done first.')
parser.add_argument('-n', '--no_error_messages', action='store_true',
                    help = 'Do not print error messages related to message formatting or fields.')

parser.add_argument('-p', '--pretty', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction,
                    help = 'Pretty print output: json messages will be more understandable, but multiline.')
parser.add_argument('-q', '--queues', type=str,
                    help = 'Names of queues you want to log, seperated by commas.  For example: --queues RESULT will print out the result queue only, while --q RESULT,LONGTHROW_DEPTH will print out all messages on two queues.  Do not put spaces around the commas, and they must be all caps.')
parser.add_argument('-s', '--summarize', action='store_true',
                    help = 'Save a summary of messages to a text file.')
parser.add_argument('-t', '--trim', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction,
                    help = 'Trim long fields.')
parser.add_argument('-u', '--unwrap', action='store_true',
        help = 'Print out just the message of each line.  The result will be a list of json objects, not a jsonl file.  This option can be used to feed into json tools.  You can only choose one of these options: -s, -u, -l.')
parser.add_argument('-w', '--writes', type=str,
                    help = 'Do not write out the whole message, instead just write out these fields, one per line and seperated by tabs.  Field names are seperated by commas.  For example: --write trigger_id will print out the trigger_id one per line, but --write uuid,container_id,trigger_id will print out three fields.  Do not put spaces around the commas.')

parser.parse_args()
args = parser.parse_args()    

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)

if args.queues is None:
    queue_names = None
else:
    queue_names = args.queues.split(',')
    
if args.messages is None:
    message_names = None
else:
    message_names = args.messages.split(',')
    
if args.containers is None:
    container_names = None
else:
    container_names = args.containers.split(',')    
    
if args.ids is None:
    ids = None
else:
    ids = args.ids.split(',')
    
if args.writes is None:
    writes = None
else:
    writes = args.writes.split(',')     

if args.list or args.unwrap:
    print('[')

for trippleStr in args.infile:
    # We ignore lines that start with # or are blank:
    if trippleStr[0] != '#' and len(trippleStr.strip()) > 0:
        #print(trippleStr)
        tripple = None
        try:
            tripple = json.loads(trippleStr)
            # Each line has three fields: queue, time_seconds, and message
        except Exception as e:
            print('')
            print(e)
            print(f'Error reading {trippleStr} from {tripple}')
            continue

        if 'queue' not in tripple:
            logging.error('Bad jsonl format.  No queue in the tripple.')
            continue
        if 'message' not in tripple:
            logging.error('Bad jsonl format.  No message in the tripple.')
            continue

        queue_name = tripple["queue"]
        message = tripple['message']
        
        # if there is no type, then we skip it, but for other missing data, we continue.
        if 'type' not in message:
            logging.error('No type in this line so skipping: {trippleStr}')
            continue
        if 'uuid' not in message or message['uuid'] == '':
            logging.warning(f'No uuid in this line: {trippleStr}')
        if 'datetime' not in message or message['datetime'] == '':
            logging.warning(f'No datetime in this line: {trippleStr}')
        
        # First, filter the results
        if args.exclude:
            if queue_names is not None and queue_name in queue_names: 
                continue
            if message_names is not None and message['type'] in message_names:
                continue
            if 'container_name' in message and container_names is not None and message['container_name'] in container_names:
                continue                
            if ids is not None and message['uuid'] in ids:
                continue                
        else:
            if queue_names is not None and queue_name not in queue_names:
                continue
            if message_names is not None and message['type'] not in message_names:
                continue
            if 'container_name' in message and container_names is not None and message['container_name'] not in container_names:
                continue 
            if 'container_name' not in message and container_names is not None:
                continue                
            if ids is not None and message['uuid'] not in ids:
                continue    
                
        # Then do something
        if args.trim:
            make_human_readable(message)
            tripple['message'] = message
        if args.summarize:
            summary['message_count'] = summary['message_count']+1
            if not message['type'] in summary:
                summary[message['type']] = 1
            else:
                summary[message['type']] = summary[message['type']] + 1
        elif args.fields:
            print(', '.join(message.keys()))                
        elif args.unwrap:
            if first:
                print(json.dumps(tripple['message']))
                first = False
            else:
                print(','+json.dumps(tripple['message']))
        elif writes is not None:
            output = ''
            for write in writes:
                if write in message:
                    output = output + message[write] + '\t'
                else:
                    output = output + 'EMPTY' + '\t'
            print(output)
        else:
            # this covers both -l (--list) output and default output
            if not args.no_output:
                if args.list:
                    if first:
                        print(json.dumps(tripple))
                        first = False
                    else:
                        print(','+json.dumps(tripple))
                else:
                    print(json.dumps(tripple))



if not args.summarize:
    if args.list or args.unwrap:
        print(']')
else:
    summary_file = open(args.summarize,'w')
    summary_file.write(json.dumps(summary, indent=4))
    summary_file.write('\n')
    summary_file.flush()
    summary_file.close()

