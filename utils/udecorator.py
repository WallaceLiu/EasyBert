# -*- coding: utf-8 -*-
"""
@time           : 2019/5/24 下午10:22
@author         : liuning@shanshu.ai
@file           : udecorator.py
@description    : udecorator


    ##########################################################
    #
    #
    #
    #
    ##########################################################


"""
import datetime
import functools
import time
import logging
from queue import Queue
from threading import Thread
import threading
from functools import wraps


def elapsed_time(func):
    """
    elapsed time of function
    :param func:
    :return:
    """

    def wrapper(*args, **kw):
        start_time = datetime.datetime.now()
        res = func(*args, **kw)
        over_time = datetime.datetime.now()
        etime = (over_time - start_time).total_seconds()
        logging.info('Elapsed time: current function <{0}> is {1} s'.format(func.__name__, etime))
        return res

    return wrapper


class asynchronous(object):
    """
    asynchronous
    """

    def __init__(self, func):
        self.func = func

        def threaded(*args, **kwargs):
            self.queue.put(self.func(*args, **kwargs))

        self.threaded = threaded

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        self.queue = Queue()
        thread = Thread(target=self.threaded, args=args, kwargs=kwargs)
        thread.start()
        return asynchronous.Result(self.queue, thread)

    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message

    class Result(object):
        def __init__(self, queue, thread):
            self.queue = queue
            self.thread = thread

        def is_done(self):
            return not self.thread.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException('the call has not yet completed its task')

            if not hasattr(self, 'result'):
                self.result = self.queue.get()

            return self.result


def synchronized(func):
    """
    simple lock
    :param func:
    :return:
    """
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


def log_debug(*arg):
    def _log(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.debug('%s, %s...args=%s, kwargs=%s' % ('.'.join(arg), func.__name__, args, kwargs))
            ret = func(*args, **kwargs)
            logging.debug('%s, %s.' % ('.'.join(arg), func.__name__))
            return ret

        return wrapper

    return _log


def log_info(*arg):
    def _log(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info('%s, %s...args=%s, kwargs=%s' % ('.'.join(arg), func.__name__, args, kwargs))
            ret = func(*args, **kwargs)
            logging.info('%s, %s.' % ('.'.join(arg), func.__name__))
            return ret

        return wrapper

    return _log
#
#
# @log('module1', 'module2')
# def test1(s):
#     print('test1 ..', s)
#     return s
#
#
# # @log('module1')
# # def test2(s1, s2):
# #     print('test2 ..', s1, s2)
# #     return s1 + s2
#
#
# test1('a')
# # test2('a', 'bc')
