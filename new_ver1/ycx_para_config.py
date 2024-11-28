import torch 
import numpy as np
from function import *

def fun1_real(fn, x, dx):
    return fn**2/(1-dx*fn)
def fun2_real(fn, x, dx):
    return 2*x+dx
def fun3_real(fn, x, dx):
    return fn*(torch.exp(dx)-1)/dx


#  Data
if_know_real_kernel=True
choose_fun=fun3
choose_real_fun=fun3_real

choose_seg_step=16
batch=256
epoch=10000
debug_epoch=1000
lr=0.01
f_start_min=0.1
f_start_max=0.9
random_amplititude=0.3

use_lr_schedule=True
lr_schedule_patience=60
lr_schedule_cooldown=3

consist_para_file_name=f'./model_parameter_consist/batch{batch}_epoch{epoch}.pth'

only_boundary_para_file_name=f'./model_parameter/batch{batch}_epoch{epoch}.pth'

use_multi_path=False

#  test parameter

ylim=(0,0.005)
