import torch 
import numpy as np
from function import *
from my_NN import *

def fun1_real(fn, x, dx):
    return fn**2/(1-dx*fn)
def fun2_real(fn, x, dx):
    return 2*x+dx
def fun3_real(fn, x, dx):
    return fn*(torch.exp(dx)-1)/dx


#  Data
if_know_real_kernel=False
choose_fun=fun1 
function_name='fun1'
#normally unchanged
choose_real_fun=fun3_real

NN=Mynetwork_2_128

choose_seg_step=16 # normally unchanged because in lx's model the seg_step is 16
batch=128
epoch=10000
debug_epoch=1000
lr=0.01
f_start_min=0.1
f_start_max=0.9
random_amplititude=0.3

#used in little loop before lr_schedule
iteration_num=4

use_lr_schedule=True
lr_schedule_patience=20
lr_schedule_cooldown=2

consist_para_file_name=f'./model_parameter_consist/batch{batch}_epoch{epoch}_{NN.name}.pth'

only_boundary_para_file_name=f'./model_parameter/batch{batch}_epoch{epoch}_{NN.name}.pth'

use_multi_path=False

#  test parameter

ylim=(0,0.005)
relative_error=False
aver_num=64

'''
调节参数：
1.NN  7个
2.batch  128 256 64 32
3.iteration_num  4 8 16 32
4.choose_seg_step  16 32 64 128
5.random_amplititude  0.1 0.2 0.3 0.4  


6.rand_move  func
'''