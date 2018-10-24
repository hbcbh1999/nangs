def checkInputs(inputs, input_type):
    if isinstance(inputs, str): return [inputs]
    elif isinstance(inputs, list): 
        for input in inputs:
            if not isinstance(input, str): raise ValueError('{} must be stings !'.format(input_type))
        return inputs
    elif not inputs: return []
    else: raise ValueError('{} must be an array of stings or a single string !'.format(input_type))

def checkRepeated(a, b):
    if a and b:
        for i in a:
            if i in b: raise ValueError('{} already exists'.format(i))

def checkSelfRepeated(a):
    if a:
        for i in range(len(a)):
            for j in range(len(a)):
                if i != j: 
                    if a[i] == a[j]:
                        raise ValueError('{} already exists'.format(a[i]))

def addInputs(inputs, input_type, check1, check2, check3=None):
    # check and add correct input
    check1 += checkInputs(inputs, input_type)
    # check if input already exists
    checkSelfRepeated(check1)
    checkRepeated(check1, check2)
    checkRepeated(check1, check3)
    return check1