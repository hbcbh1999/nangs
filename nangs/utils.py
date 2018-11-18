import numpy as np 

def checkIsListOfStr(l, name):
    assert isinstance(l, list), name+' must be a list of strings'
    for i in l: assert isinstance(i, str), name+' must be a list of strings'
    return

def checkNoRepeated(l1, l2):
    for i in l1: 
        if i in l2: raise ValueError('Repeated item '+i)
    return

def checkNoRepeatedSame(l):
    for i, item in enumerate(l): 
        for j, item2 in enumerate(l): 
            if item == item2 and i != j:
                raise ValueError('Repeated item '+item)
    return

def checkValidArray(a):
    assert isinstance(a, np.ndarray), 'Values must be numpy arrays !'
    assert a.ndim == 1, 'Arrays must have only one dimensions !'

def checkValidDict(d):
    assert isinstance(d, dict), 'Values must be a dictionary !'
    for k in d:
        checkValidArray(d[k])
    return

def checkDictArray(d, a):
    for k in a:
        assert k in d, k + ' is required'
    return

def initValues(l):
    v = []
    for i in l: v.append([])
    return v

def setValue(l, v, k, val):
    ix = l.index(k)
    v[ix] = val
    return