""" 
This traning script can run on single gpu with debug mode 
and also in large training run with distributed data parallel (DDP)

to run on single GPU 
$ python train.py --batch_size=32 --compile=False 

to run with DDP on 4 GPUs on 1 node, example: 
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 GPUs across 2 nodes, examples 
- Run on the first (master) node with example IP 123.456.789 
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.789 --master_port=1234 train.py 
- Run on the worker node : 
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.789 --master_port=1234 train.py 

(if your cluster doesn't have Infiniband interconnect prepend NCC_IB_DISABLE=1)
"""

import os 
import time 
import math 
import pickle 
from contextlib import nullcontext

import numpy as np 
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group 

from model import GPTConfig,GPT 

# ----------------------------------------------
# default config values designed ti train gpt2(124M) on OpenWebText 
# I/O 
out_dir='out'
eval_interval=2000
log_interval=1 
eval_iters=200 
eval_only=False # True, script exists right after the first eval
always_save_checkpoint=True  # True, always save a ckpt after each eval 
init_from='scratch' # 'scratch', or 'resume' or "gpt2*"
# wandb logging 
wandb_log=False #disabled by deafult 
wandb_project='owt'
wandb_run_name='gpt2' # 'run' + str(time.time())

# data 
dataset='openwebtext'
gradient_acculation_steps=5*8 # used for simulating largeer batch sizes 
batch_size=12 
block_size= 1024 

# model 
n_layer=12 
n_head=12 
n_embd=768 
dropout=0.0  # for pre-training 0 is good, for finetuning try 0.1+ 
bias=False  # we don't use bias inside LN and LL 

# adam optimizer 
learning_rate=6e-4 
max_iters= 6000000 #max_lr 
weigh_decay= 1e-1 
beta1= 0.9 
beta2= 0.95 
grad_clip = 1.0  #clip gradients at this value, or disable if==0.0 

# learning rate decay settings 
decay_lr=True # Whether to decay the LR 
warmup_iters= 2000 # how many steps to warmup for 
lr_decay_iters=6000000 # should be ~= max_iters 
min_lr= 6e-5 # minimum learning rate, should be ~=learning_rate/10 

# DDP settings 
backend='nccl' # 'nccl','gloo', etc 

# system 
device='cuda' 
dtype='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile= True 

# ------------------------------- 
config_keys=[k for k,v in globals.items() if not k.startswith('_') and isinstance(v,(int,float,bool,str))]
exec(open('configurator.py').read()) # overrides from command line or conifg line  
config={k:globals()[k] for k in config_keys}
