# This script simulates events coming out of the Hololens.
# It simulates several operator move events and one operator pointing event.
# There is commented out code to simulate a frame from a video feed.
# (But this is not used because it is so large it makes it hard to follow the logs.)

import argparse
import base64
import json
import logging
import sys
import time

import zmq

from ccu import CCU


def open_pub_queues():
    for name, queue in CCU.queues.items():
        queue['socket'] = CCU.socket(queue, zmq.PUB) 

current_time = 0.0
    
parser = argparse.ArgumentParser('Program to send many messages for testing.')
parser.add_argument('-d', '--debug', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Print out more details of each message sent.')
parser.add_argument('-j', '--jsonl', type=open, 
                    help = 'Run the script in the provided file (in JSON line format).')
parser.add_argument('-q', '--quiet', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Dont print out dots.')
parser.add_argument('-o', '--one', type=open, 
                    help = 'Inject the message in the provided file (in JSON format).')
# These arguments are used for passing messages one by one, which is a future feature.
#parser.add_argument('-M', '--message', type=str, 
#                    help = 'Inject the message in the provided file (in JSON format).')
#parser.add_argument('-Q', '--queue', type=str, 
#                    help = 'Inject the message in the provided file (in JSON format).')
#parser.add_argument('-W', '--wait', type=float, default=0.0, 
#                    help = 'Inject the message in the provided file (in JSON format).')

parser.parse_args()
args = parser.parse_args()    

CCU.config()
last_queue = None
last_wait = None

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)

if args.one is not None:
    logging.info('Injecting message in {args.one} file.')
    message = json.load(args.one)
    print(message)
    #CCU.send(arg.queue, message)
            
if args.jsonl is not None:
    logging.info(f'Running script in {args.jsonl.name} file with CCU {CCU.__version__} on {CCU.host_ip}.')
    logging.info(f'with config file {CCU.config_file_path}.')  
    for trippleStr in args.jsonl:
        # We ignore lines that start with # or are blank:
        if trippleStr[0] != '#' and len(trippleStr.strip()) > 0:
            #print(trippleStr)
            tripple = None
            try:
                tripple = json.loads(trippleStr)
                # Each line has three fields: queue, time_seconds, and message
                queue_name = tripple["queue"]
                queue = CCU.queues[queue_name]
                trigger_time = tripple['time_seconds']
                message = tripple['message']                
            except Exception as e:
                print('')
                print(e)
                print(f'Error reading {trippleStr} from {tripple}')
                continue
            
            # We open sockets as we need them, so we don't waste time opening
            # sockets we do not need.
            if 'socket' not in queue:
                queue['socket'] = CCU.socket(queue, zmq.PUB)
            
            if trigger_time > current_time:
                time.sleep(trigger_time-current_time)
            current_time = trigger_time
            
            # If messages don't have UUIDs or timestamps, then add them
            if 'uuid' not in message or message['uuid'] == '':
                message['uuid'] = CCU.uuid()
            if 'timestamp' not in message or message['timestamp'] == '':
                message['timestamp'] = CCU.now()             
            
            if args.debug:
                print(f'At {current_time} sending {message["type"]} message on {queue_name}.')
            CCU.send(queue['socket'], message)
            if not args.debug and not args.quiet:
                print('.', end='', flush=True)

#if args.queue is not None and args.message is None:
#    print('No -M or --message to go with the -Q and -M/-Q must be provided together.')
#    sys.exit(1)
    
#if args.message is not None:
#    if args.queue is None:
#        print('No -Q or --queue option on command line and -M/-Q must be provided together.')
#        sys.exit(1)
#    queue = named_queues[args.queue]
#    time.sleep(args.wait)
#    current_time = current_time + args.wait
#    message_json = json.loads(args.message)
#    CCU.send(queue, message_json)
#    if not args.quiet:
#        print('.', end='')    