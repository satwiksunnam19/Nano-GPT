''' 
The model defined by Sir.Andrej 
'''

import math 
import inspect # useful for the getting info from the oython objs 
from dataclasses import dataclass 

import torch
import torch.nn as nn
from torch.nn import functional as F

from masking import Causal_Masking, Intra_Document_Masking
from positional_encoding import RoPE
from attention_ablation import create_sliding_window_mask

class LayerNorm(nn.Module):
    """ 
 LayerNorm but with an optional bias. Pytorch Doesn't support bias=Flase    
    """
    def __init__(self,ndim,bias):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(ndim))        
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self,input):
        return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()

        # because each head needs equal dimension. if n_embd=768 and n_head=12, each head gets 64 dims.
        # why head needs equal dimension ?  becasue in MHA preocesses all heads in parallel using batch operation
        assert config.n_embd % config.n_head==0

        # key, query, value projections of all heads, but in a batch

        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd,bias=config.bias)

        # output projection
        self.c_proj= nn.Linear(config.n_embd,config.n_embd,bias=config.bias)

        # regularization
        self.attn_dropout=nn.Dropout(config.dropout)
        self.resid_dropout=nn.Dropout(config.dropout)
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        self.dropout=config.dropout

        # Sliding window parameters
        self.window_size = getattr(config, 'window_size', None)
        self.sink_size = getattr(config, 'sink_size', 0)
        self.use_sliding_window = self.window_size is not None

        # flash attention make GPU got burrrrr
        self.flash = hasattr(torch.functional,'scaled_dot_product_attention')
        # Disable flash attention if using custom masking
        if self.use_sliding_window:
            self.flash = False  # sliding window needs custom masking

        # RoPE for rotary position embedding
        self.use_rope = config.position_encoding == "rope"
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rope = RoPE(head_dim, max_seq_len=config.block_size)

        if not self.flash or self.use_sliding_window:
            if not self.use_sliding_window:
                print("Warning:using slow attention, flash attention requires pytorch >=2.0 and compitable NV GPU")
                # casual mask to ensure that attention is only applied to the left in the seq
                self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size, config.block_size))
            else:
                # sliding window will create mask dynamically
                self.register_buffer("bias", None)

    
    def forward(self, x, mask=None, return_attention=False):
        B, T, C = x.size()  # batch size, seq length, embedding dimensionality (n_embd)

        # calculating query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # apply RoPE if enabled (rotates q and k based on position)
        if self.use_rope:
            q, k = self.rope(q, k, T)

        # causal self attention; Self-attend (B, nh, T, S) x (B, nh, T, T)
        if self.flash and mask is None and not self.use_sliding_window:
            # efficient attention using Flash Attention CUDA Kernels (only for standard causal)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
            att = None
        else:
            # manual implementation of self attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if mask is not None:
                # use provided mask (e.g., intra-document mask)
                att = att.masked_fill(mask == 0, float('-inf'))
            elif self.use_sliding_window:
                # use sliding window mask
                sliding_mask = create_sliding_window_mask(T, self.window_size, self.sink_size, x.device)
                att = att.masked_fill(~sliding_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                # fallback to standard causal mask
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all heads outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if return_attention:
            return y, att
        return y 
    
class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd,bias=config.bias)
        self.gelu=nn.GELU()
        self.c_proj= nn.Linear(4*config.n_embd,config.n_embd,bias=config.bias)
        self.dropout=nn.Dropout(config.dropout)
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        x=self.dropout(x)

        return x 

class Block(nn.Module):
    """
    key-concepts:
    a) pre-norm architecture:
    LN -> Attn -> Add(residual)
    LN -> MLP-> Add(residual)
    b) residual connections
    helps grad flow
    allow deep networks (12+ layers)
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        # Select attention mechanism based on config
        if config.attention_type == "gqa":
            from attention_ablation import GQA
            self.attn = GQA(config)
        elif config.attention_type == "mla":
            from attention_ablation import MLA
            self.attn = MLA(config)
        else:  # "standard"
            self.attn = CausalSelfAttention(config)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        

    def forward(self, x, mask=None, return_attention=False):
        if return_attention:
            attn_out, att_weights = self.attn(self.ln_1(x), mask=mask, return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln_1(x), mask=mask)

        x = x + self.mlp(self.ln_2(x))

        if return_attention:
            return x, att_weights

        return x 


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LN, in GPT-2 False: a bit faster and better
    # Attention mechanism: "standard", "gqa", "mla"
    attention_type: str = "standard"
    n_kv_heads: int = 4  # for GQA: number of key-value heads (must divide n_head)
    latent_dim: int = 128  # for MLA: latent dimension for KV compression
    # Masking: "causal" or "intra_document"
    masking_type: str = "causal"
    eos_token_id: int = 50256  # for intra-document masking
    # Position encoding: "learned" or "rope"
    position_encoding: str = "learned"
    # Sliding window attention: None = full causal, or specify window size
    window_size: int = None  # number of recent tokens to attend to (e.g., 512)
    sink_size: int = 0  # number of initial tokens always accessible (e.g., 4)

class GPT(nn.Module):

    def __init__(self,config): 
        super().__init__()
        assert config.vocab_size is not None 
        assert config.block_size is not None 
        self.config=config 

        self.transformer= nn.ModuleDict(dict(
            wte= nn.Embedding(config.vocab_size,config.n_embd), #Token embeddings 
            wpe= nn.Embedding(config.block_size, config.n_embd),  # Position Embeddings
            drop=nn.Dropout(config.dropout),  
            h=nn.ModuleList([Block(config)for _ in range(config.n_layer)]), # stack of 12 transformer blocks 
            ln_f= LayerNorm(config.n_embd, bias=config.bias)
        ))


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # embeddings to vocab probs
        """
        weight tying: input embedding and output projection share weights
        reduces params, better performance empirically,
        makes sense: similar tokens should have similar embeddings and predictions
        """
        self.transformer.wte.weight = self.lm_head.weight

        # masking module
        if config.masking_type == "intra_document":
            self.masking = Intra_Document_Masking(config)
        else:
            self.masking = None  # use default causal in attention  

        # init all weights 
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper 
        for pn, p in self.named_parameters(): 
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
        # report no of params  
        print("Number of parameters : %2fM" %(self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the no.of params in the model
        For non-embedding count (default), the position embeddings get substracted.
        The token embeddings would too, except due to the param sharing these params
        are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.position_encoding != "rope":
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self,module): 
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module,nn.Embedding): 
            torch.nn.init.normal_(module.weight,mean=0.0, std=0.02)

    def forward(self, idx, targets=None, document_ids=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward seq of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # create mask if using intra-document masking
        mask = None
        if self.masking is not None:
            if document_ids is None:
                # auto-detect document boundaries from eos tokens
                document_ids = Intra_Document_Masking.get_document_ids(idx, self.config.eos_token_id)
            mask = self.masking(document_ids)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)

        # RoPE handles position in attention, so skip adding pos_emb
        if self.config.position_encoding == "rope":
            x = self.transformer.drop(tok_emb)
        else:
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, mask=mask)
        x = self.transformer.ln_f(x)

        if targets is not None: 
            # if we are given some desired targets also calculate the loss 
            logits= self.lm_head(x)
            loss= F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: 
            # inference time mini optimization: only forward the lm_head on the very last position
            logits= self.lm_head(x[:,[-1],:]) 
            loss=None 
        return logits,loss 
    
    def crop_block_size(self,block_size): 
        # model surgery to decrease the block size if necessary 
        # e.g., we may load the GPT2 pre-trained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller,  simpler model 
        assert block_size <= self.config.block_size 
        self.transformer.wpe.weight = nn.parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h: 
            if hasattr(block.attn,'bias'): 
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod 
    def from_pretrained(cls, model_type, ovveride_args=None): 
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        ovveride_args= ovveride_args or {} #default to empty dict 
        # only dropout can be overidden see more notes below 
        assert all(k =='dropout' for k in ovveride_args) 
        from transformers import GPT2LMHeadModel 
        print("Loading weights from pre-trained  gpt: %s " % model_type)

        config_args= {
            'gpt2': dict(n_layer=12, n_head=12, m_embd=768), 
            'gpt2-medium': dict(n_layer= 24, n_head=16, n_embd=1024), 
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd= 1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("forcing vocab_size= 50257, block_size=1024, bias=True")

        config_args['vocab_size']= 50257 # always 50257 for gpt2 model checkpoints 
        config_args['block_size']= 1024  # always 1024 for GPT model checkpoints 
        config_args['bias']= True # always True for GPT model checkpoints  
        # we can ovveride the drp out rate, if desired 
        if 'dropout' in ovveride_args: 
            print(f"ovveriding dropout rate to {ovveride_args['dropout']}")
            config_args['dropout']=ovveride_args['dropout']
        # create a from-scratch initialized miniGPT model 
        config=GPTConfig(**config_args)
        model=GPT(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys= [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # init a hf/transformer model 
        model_hf= GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf=model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes 
        sd_hf_keys=sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self,weight_decay,learning_rate,betas,device_type): 
        # start with all of the candidate parameters 
        param_dict={pn:p for pn,p in self.named_parameters()}
        # filter out theoose  that do not require grad 
        param_dict={pn:p for pn,p in param_dict.items() if p.requires_grad}
        # cretae option groups. Any parameters that is 2D will be might decayed, oherwise no 
        # i.e., all weight in tensors in matmuls + embedding decay, all biases and layernorms don't 
        decay_params= [ p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for n,p in param_dict.items() if p.dim()<2]

        optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters 
        use_fused= fused_available and device_type=='cuda'
        extra_args=dict(fused=True) if use_fused else dict()
        optimizer=torch.optim.AdamW(optim_groups,lr=learning_rate,betas=betas,**extra_args)
        print(f"Using Fused AdamW:{use_fused}")

        return optimizer

    def estimate_mfu(self,fwdbwd_per_iter,dt):
        """ estimate model flops utilization (MFU) as percentage of device peak FLOPS"""
        # first estimate the no of flops we do per iteration
        N= self.get_num_params()
        cfg= self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of device peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second

        # Determine device-specific peak FLOPS
        # Get first parameter's device to detect what we're running on
        device = next(self.parameters()).device
        if device.type == 'cuda':
            # A100 GPU bfloat16 peak flops is 312 TFLOPS
            flops_promised = 312e12
        elif device.type == 'mps':
            # Apple Silicon M4 (base) FP32 peak ~2 TFLOPS
            # M4 Pro ~3.5 TFLOPS, M4 Max ~5.5 TFLOPS
            # Using conservative 2 TFLOPS estimate for base M4
            flops_promised = 2e12
        else:
            # CPU or other device - use A100 as reference for comparison
            flops_promised = 312e12

        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=1.0,top_k=None):
        """
        take a conditioning seq of indices idx(LongTnsor of shape (b,t)) and complete.
        the seq max_new_tokens times, feeding the pres back into the model each time.
        most likely you'll want to make sure to be in model.eval() mode of operation this.
        """

        for _ in range(max_new_tokens):
            # if the seq context is growing too long we must crop it at block size
            idx_cond= idx if idx.size(1) <=self.config.block_size else idx[:,-self.config.block_size:]
            # forward the model to get the final step and scale by dsesired temperature
            logits,_= self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    