# -*- coding: utf8 -*-
import threading

import numpy as np

ThreadClass = lambda func, name, args:threading.Thread(None, func, name, args)

def wrapper_func(func, args=(), name=""):
    """run a func in another thread"""
    assert callable(func)
    assert isinstance(args, tuple)
    assert isinstance(name, str)

    func_thread = ThreadClass(func, name, args)
    func_thread.setDaemon(True)
    return func_thread


def flatten_list(list_input):
    list_re = []
    for l in list_input:
        if isinstance(l, list):
            list_re.extend(flatten_list(l))
        else:
            list_re.append(l)
    return list_re
