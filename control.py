import logging
import zmq
from ccu import CCU



if __name__ == "__main__":
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    control_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

    flag = True
    while True:
        # Waits for an offensive message on the result queue.
        # For all such messages, it sends a text_popup message so that the
        # Hololens will notify the operator to not do that.
        if flag:
            logging.info("INTO WHILE LOOP...")
            flag = False

        x = input('If you want to end the session, please type \"END\"\n')

        if x.lower() == 'end':
            control_message = CCU.base_message('control')
            control_message['action'] = 'end_session'
            logging.info(control_message)
            CCU.send(control_queue, control_message)







