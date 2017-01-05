import sys
import urllib.request
from os.path import join

import time
import threading
import queue as Queue

im_dir = sys.argv[1]

# utility - spawn a thread to execute target for each args
def run_parallel_in_threads(target, args_list):
    result = Queue.Queue()
    # wrapper to collect return value in a Queue
    def task_wrapper(*args):
        result.put(target(*args))
    threads = [threading.Thread(target=task_wrapper, args=args) for args in args_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return result


# for line in sys.stdin:

def fetch_url(line):
    url = line.split(',')[0]
    identifier = url.split('media/')[-1].replace('/','-').strip()

    try:
        urllib.request.urlretrieve(url, join(im_dir, identifier))
    except:
        pass


run_parallel_in_threads(fetch_url, [(x,) for x in sys.stdin.readlines()])

