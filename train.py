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
weight_decay= 1e-1 
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
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or conifg line  
config={k:globals()[k] for k in config_keys} # will be used for logging 

# --------------------------------

# various inits, delivered attributes. I/O setup 
ddp= int(os.environ.get('RANK','-1')) != -1 # is this ddp run ? 
if ddp : 
    init_process_group(backend=backend)
    ddp_rank=int(os.environ['RANK'])
    ddp_local_rank=int(os.environ['LOCAL_RANK'])
    ddp_world_size=int(os.environ['WORLD_SIZE'])
    device=f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device=device)
    master_process= ddp_rank == 0 # this process will do logging, ckpting, etc 
    seed_offset= ddp_rank # each process gets adiffernt seed 
    # world size number of processes will be training simultenouly,so we can scale 
    # down the desired gradient accumulation iterations per process proportionally 
    assert gradient_acculation_steps % ddp_world_size == 0 
    gradient_acculation_steps//=ddp_world_size 
else: 
    # if not ddp, we are running on single GPU, and one process 
    master_process=True 
    seed_offset=0 
    ddp_world_size=1 

tokens_per_iter= gradient_acculation_steps * ddp_world_size * batch_size * block_size 

print(f"tokens per iterations will be : {tokens_per_iter}")

if master_process : 
    os.makedirs(out_dir,exist_ok=True)
torch.manual_seed(1337+ seed_offset)
torch.backends.cuda.matmul.allow_tf32= True # allow tf32 
torch.backends.cudnn.allow_tf32= True # allow tf32 on cudnn 
device_type='cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype={'float32':torch.float32, 'bfloat16':torch.bfloat16, 'float16':torch.float16}[dtype]
ctx= nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type,dtype=ptdtype)

# poor man's data loader 
data_dir= os.path.join('data',dataset)
def get_batch(split): 
    # we recreate np.memmap every batch to avoid memory leak, as per 
    if split == "train": 
        data= np.memmap(os.path.join(data_dir,'train.bin'),dtype=np.uint16, mode='r')
    else: 
        data= np.memmap(os.path.join(data_dir,'val.bin'),dtype=np.uint16, mode='r')
    
    ix= torch.randint(len(data)-block_size,(batch_size,))
    x= torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y= torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda": 
        # pin arrays x,y which allows us to move them from cpu to gpu asychrounsoulsy (non_blocking=True)
        x,y= x.pin_memory().to(device,non_blocking=True), y.pin_memory().to(device,non_blocking=True)
    else: 
        x,y= x.to(device), y.to(device)
    return x,y 

# init these up here, can ovveride if init_from='resume' 
iter_num=0 
best_val_loss= 1e9 

# attempt to derive vocab_size from the dataaset 
meta_path= os.path.join(data_dir,'meta.pkl')
meta_vocab_size= None 
if os.path.exists(meta_path): 
    with open(meta_path,'rb') as f: 
        meta= pickle.load(f)
    meta_vocab_size=meta['vocab_size'] 
    print(f"found vcab size = {meta_vocab_size} (inside {meta_path})")

# model init 
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line 

if init_from == "scratch": 
    # init a new model from scratch 
    print("Initaitializing a new model from scratch")
    # determine the vocab size we'll use from-scractch training  
    if meta_vocab_size is None: 
        print("defautling to vocab_size of GPT-2 to 50304 (50257 rounded for effciency)")
    model_args['vocab_size']=meta_vocab_size if meta_vocab_size is not None else 50304 
    gptconf= GPTConfig(**model_args)
    model=GPT(gptconf)

elif init_from == "resume": 
    print(f"Resuming traning from{out_dir}")
    # resume training from scratch : 
    ckpt_path= os.path.join(out_dir,'ckpt.pt')
    checkpoint=torch.load(ckpt_path,map_location=device)
    checkpoint_model_args= checkpoint['model_args']
    # force these config attributes ti be equal otherwise we can't even resume traning 
    # the rest of thr attributes (e.g dropout) can stay desired from command line 
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']: 
        model_args[k]= checkpoint_model_args[k]
    
    # create the model 
    gptconf= GPTConfig(**model_args)
    model= GPT(gptconf)
    state_dict= checkpoint['model']

    # fix the keys of the state dictionary
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery 
if block_size< model.config.block_size: 
    model.crop_block_size(block_size)
    model_args['block_size']= block_size # so the ckpt have te right value 

model.to(device)

# initalize a GradScaler. if enabled=False scaler is no-op
# Note: GradScaler only works with CUDA, not MPS
scaler= torch.amp.GradScaler(device_type, enabled=(dtype=='float16' and device_type=='cuda'))

# optimizer 
optimizer= model.configure_optimizers(weight_decay,learning_rate, (beta1,beta2), device_type)
if init_from  == 'resume': 
    optimizer.laod_state_dict(checkpoint['optimizer'])
checkpoint=None # free up memory 

# compile the model 
if compile: 
    print("compiling the model ... (takes a ~ minute)")
    unoptimized_model= model
    model=torch.compile(model) # requires torch 2.0 


# wap model into DDP container 
if ddp : 
    model=DDP(model,device_ids=[ddp_local_rank])

# helps estimate a arbitarily accurate loss over split using many batches 
@torch.no_grad()
def estimate_loss(): 
    out= {}
    model.eval()
    for split in ['train','val']: 
        losses= torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            with ctx: 
                logits,loss= model(x,y)
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out 

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
 
 # logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)



# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration 
    lr= get_lr(iter_num) if decay_lr else learning_rate 
    for param_group in optimizer.param_groups: 
        param_group['lr']= lr 

    # evaluate the loss on train/val sets and write checkpoints 
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                 checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                 print(f"saving checkpoint to {out_dir}")
                 torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to siulate larger batch size
    # and using the GradScaler if data type is float16 

    for micro_step in range(gradient_acculation_steps): 
        if ddp: 
            model.require_backward_grad_sync= (micro_step==gradient_acculation_steps-1)
        
        with ctx: 
            logits, loss= model(X,Y)
            loss= loss/ gradient_acculation_steps # scale the loss to account for gradient accumulation 

        # imediate async prefetch next batch while model is doing the fprward pass on the GPU 
        X,Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16 
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip!= 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
    
    # step the optimizer and scaler if training in fp16 
    scaler.step(optimizer)
    scaler.update()
    # flush the gradient as soon as we can, no need for this memory anymore 
    optimizer.zero_grad(set_to_none=True)

    # timing and logging 
    t1= time.time()
    dt= t1 -t0 
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_acculation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_acculation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions 
    if iter_num > max_iters : 
        break 

if ddp: 
    destroy_process_group() 
