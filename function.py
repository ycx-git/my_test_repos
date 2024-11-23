import numpy as np
def fun1(start_value):
    #from 0 to 1 
    #lixiang used model
    end_value=start_value/(1-start_value)
    return end_value

def fun2(start_value):
    return np.e**start_value