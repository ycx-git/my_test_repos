import numpy as np
def fun1(start_value):
    #from 0 to 1 
    #lixiang used model
    end_value=start_value/(1-start_value)
    return end_value

def fun2(f0):
    return f0+1

def fun3(f0):
    return f0*np.e

def fun4(f0):
    return 100*f0/(1-f0)

def fun5(f0):
    return 1/(1-f0)**2

def fun6(f0):
    return 