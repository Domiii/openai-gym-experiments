
from multiprocessing import Process, Queue
import multiprocessing as mp
from functools import wraps


##########################################################################################
# Parallel + multithreading tools
##########################################################################################

# https://turtlemonvh.github.io/python-multiprocessing-and-corefoundation-libraries.html


def run_in_forked_process(f):
    def wrapf(q, f, args, kwargs):
        q.put(f(*args, **kwargs))

    @wraps(f)
    def wrapper(*args, **kwargs):
        q = Queue()
        Process(target=wrapf, args=(q, f, args, kwargs)).start()
        return q.get()

    return wrapper

# allows us to run multiple functions in parallel (on separate threads)
# WARNING: be careful to not let the threads read/write share data, or add thread-safety explicitely


class TaskQueue:
  def __init__(self):
    self.processes = []

  def add(self, task, *args):
    p = mp.Process(target=task,
                   args=args)
    p.start()
    self.processes.append(p)

  def join(self):
    for p in self.processes:
      p.join()
