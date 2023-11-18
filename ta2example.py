# Very simple example of a TA2 process.
# Any offensive action is warned via a text popup.


import logging
import zmq

from ccu import CCU

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)
logging.info('TA2 Example Running')

result = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
action = CCU.socket(CCU.queues['ACTION'], zmq.PUB)

def text_popup(contents):
    """Creates a simple messsage of type 'text-popup'.  
       The base_message function also populates the 'uuid'
       and 'datetime' fields in the message (which is really just a
       Python dictionary).
    """    
    message = CCU.base_message('text-popup')
    message['contents'] = contents
    return message

while True:
    # Waits for an offensive message on the result queue.
    # For all such messages, it sends a text_popup message so that the
    # Hololens will notify the operator to not do that.    
    message = CCU.recv_block(result)
    if message['type'] == 'offensive':
        offensive_action = message['detail']
        logging.debug('Sent a fix for an offensive action.')
        CCU.send(action, text_popup(f'Please stop {offensive_action}'))