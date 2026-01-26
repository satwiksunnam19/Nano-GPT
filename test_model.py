import torch 
from model import LayerNorm,CausalSelfAttention,MLP
from dataclasses import dataclass 
import torch.nn as nn 

# layernorm normalizes each (batch,seq_position) independently.
ln=LayerNorm(ndim=768,bias=True)
x=torch.randn(2,10,768) #(b,seq_len,n_embd)

# fwd pass 
out=ln(x)
# check op should have normalized features 
print(f"input mean:{x.mean()}, std:{x.std()}")
# this is global computing off std/mean but across the entire tensor, but layernorm normalizes per-position across the feature dim 
print(f"ouput mean:{x.mean()}, std:{x.std()}") 
print(f"output mean:{out.mean(dim=-1)}, std:{out.std(dim=-1)}")  
# mean should be close to ~0 and std ~1 (before weight/bias applied internally) 

@dataclass 
class Config: 
    n_embd:int= 768 
    n_head: int= 12 
    block_size: int =1024 
    bias: bool= True
    dropout: float =0.1 

config=Config()
attn=CausalSelfAttention(config)

# test input 
x= torch.randn(2,16,768) #(batch,seq_len,n_embd)
# Forward Pass 
output=attn(x) 

print(f"Input shape:{x.shape}")
print(f"Output shape:{output.shape}")

# why head should be equal dimensions? 
B,T,C = 2,16, 768 
n_head= 12 
head_dim= C//n_head # 64  768//12= 64 
print(f"Total embedding: {C}")
print(f"Number of heads:{n_head}")
print(f"Dimensions Per Head: {head_dim}")
print(f"Verify:{n_head}x{head_dim}={n_head*head_dim} should equal {C}")

# simulate the reshape  
x= torch.randn(B,T,C)
x_reshaped=x.view(B,T,n_head,head_dim)
print(f"Original shape:{x.shape}")
print(f"After split into heads:{x_reshaped.shape}")

# concatenate back 
x_concat=x_reshaped.view(B,T,C)
print(f"After concat heads;{x_concat.shape}")
print(f"All elements preserved: {torch.allclose(x,x_concat)}")

# let's check the gradient flow through the layernorm and CSA 
print("Testing LN Gradient Flow")
ln=LayerNorm(ndim=768, bias=True)
x=torch.randn(2,10,768,requires_grad=True)  # requires_grad is the key ! 

# forward pass 
out= ln(x) 
loss=out.sum()  # dummy stuff 
# bkwrd pass 
loss.backward()

# check gradients 
print(f"Input gradient exists: {x.grad is not None}")
print(f"input gradient shape: {x.grad.shape}")
print(f"Input gradient norm: {x.grad.norm().item():.4f}")
print(f"Weights gradients:{ln.weight.grad.norm().item():.4f}")
print(f"bias gradient : {ln.bias.grad.norm().item():.4f}")

print("Testing Gradient flow of CSA")
config=Config()
attn=CausalSelfAttention(config)
x=torch.randn(2,16,768,requires_grad=True)

# Forward pass
output = attn(x)
# Dummy loss
loss = output.sum()
# Backward pass
loss.backward()

# Check gradients
print(f"Input gradient exists: {x.grad is not None}")
print(f"Input gradient norm: {x.grad.norm().item():.4f}") 


# check gradients for each param 
for name, param in attn.named_parameters():
    if param.grad is not None: 
        print(f"{name}:grad norm={param.grad.norm().item():.4f}")
    else:
        print(f"{name}:No Gradient (problem!)") 

# let's see the heatmap 
import matplotlib.pyplot as plt 

grad=[]
names=[]
for name, param in attn.named_parameters():
    if param.grad is not None: 
        names.append(name)
        grad.append(param.grad.norm().item())

plt.figure(figsize=(12, 6))
plt.bar(range(len(grad)), grad)
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.xlabel('Parameter')
plt.ylabel('Gradient Norm')
plt.title('Gradient Flow Through Attention Layers')
plt.tight_layout() 
plt.show()

# visulaize the attention weights 
attn.eval()
x = torch.randn(1, 8, 768)  # Small sequence for easier visualization

with torch.no_grad():
    output,att_weights= attn(x,return_attention=True)

# visualize attn pattern for first head 
if att_weights is not None:
    # shape:(B,n_head,T,T)
    first_head=att_weights[0,2].cpu().numpy()

    plt.figure(figsize=(8,8))
    plt.imshow(first_head,cmap='viridis')
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key position")
    plt.ylabel("Query Position")
    plt.title("Attention Pattern (Head 2)")
    plt.tight_layout()
    plt.show()

else: 
    print("Flash Attention enabled- can't visualize weights")


# create MLP 
mlp=MLP(config)
x= torch.randn(2,16,768,requires_grad=True)

out= mlp(x)
loss=out.sum()
loss.backward()
print(x.shape==out.shape, "shape preserved")

# check dimensions 
print(f"MLP Architecture: ")
print(f"Input:{config.n_embd}")
print(f"Hidden (4x) : {4*config.n_embd}")
print(f"Output: {config.n_embd}")

# check gradients 
print(f"Input grad exists: {x.grad is not None}")
print(f"input gradient norm: {x.grad.norm().item():.4f}")

for name, param in mlp.named_parameters():
    if param.grad is not None : 
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    else: 
        print(f"{name}: NO GRADIENT !")

# check non-linearity 
mlp.eval()

x_sim=torch.ones(1,1,768)

with torch.no_grad():
    hidden=mlp.c_fc(x_sim)
    print(f"After Expansion:{hidden.shape}(should be 4x larger)")

    # apply GELU 
    activ= mlp.gelu(hidden)
    print(f"AFTER GELU:{activ.shape}")
    print(f"GELU INTRODUCES nonlinearity:{not torch.allclose(hidden,activ)}")

    out=mlp(x_sim)
    print(f"Final output: {out.shape}")

# param count 
print(" param analysis")
total_params=sum(p.numel() for  p in mlp.parameters())
print(f"total parameters{total_params}")

for name, param in mlp.named_parameters():
    print(f"{name}:{param.shape}={param.numel():,} params")


# ================= 
# Test block module 
# =================

print("\n" + "="*50)
print("Testing Block Module")
from model import Block 
block=Block(config)

x= torch.randn(2,16,768) 
out=block(x)

print(f"Input Shape:{x.shape}")
print(f"output Shape: {out.shape}")
print(f"Shape Preserved:{x.shape==out.shape}")

# Test Residual connections 
print(f"\n Residual Connection Test:")
print(f'The Output should be Different from input(due to learned transformations)')
print(f"Input and Op are same: {torch.allclose(x,out)}")

# Gradient flow through block
x_grad = torch.randn(2, 16, 768, requires_grad=True)
out_grad = block(x_grad)
loss = out_grad.sum()
loss.backward() 

print(f"\nGradient flow:")
print(f"Input gradient exists: {x_grad.grad is not None}")
print(f"Input gradient norm: {x_grad.grad.norm().item():.4f}")
 
 # Count parameters in one block
block_params = sum(p.numel() for p in block.parameters())
print(f"\nParameters in one block: {block_params:,}")


# ============================================
# Test GPT Module
# ============================================
print("\n" + "="*50)
print("Testing Full GPT Model")
print("="*50) 

from model import GPT, GPTConfig

# Create small GPT for testing
gpt_config = GPTConfig(
    block_size=16,      # Smaller for testing
    vocab_size=1000,     # Smaller vocab
    n_layer=16,           # Fewer layers
    n_head=8,
    n_embd=512,          # Smaller embedding
    dropout=0.1,
    bias=True
)

# Create model
gpt = GPT(gpt_config) 

# Test input (batch of tokenIDs)
batch_size=2 
seq_len=16 
idx= torch.randint(0,gpt_config.vocab_size,(batch_size,seq_len))
print(f"Input token IDs shape: {idx.shape}")

# Forward pass (without targets= Infer Mode)
logits,loss= gpt(idx,targets=None)
print(f"expected: (batch_size,1,vocab_size) for inference ")
print(f"Loss:{loss}") 

# Frwd pass wirh targets (training mode)
targets=torch.randint(0,gpt_config.vocab_size,(batch_size,seq_len))
logits_train,loss_train=gpt(idx,targets=targets)

print(f"\n Traning mode")
print(f"\n Model Parameters:{gpt.get_num_params():,}")
print(f"Model params(excluding positional embeddings:{gpt.get_num_params(non_embedding=True):,})")

# Test gradients flow through full model 
idx_grad= torch.randint(0,gpt_config.vocab_size,(batch_size,seq_len))
targets_grad= torch.randint(0,gpt_config.vocab_size,(batch_size,seq_len))

gpt.train()
logits,loss= gpt(idx_grad,targets=targets_grad)
loss.backward()

print(f"\nGradient flow through full model:")
grad_norms = {}
for name, param in gpt.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

print(f"Total parameters with gradients: {len(grad_norms)}")
print(f"Sample gradient norms:")
for name, norm in list(grad_norms.items())[:5]:
    print(f"  {name}: {norm:.4f}")

# ============================================
# Test Weight Tying
# ============================================
print("\n" + "="*50)
print("Testing Weight Tying")
print("="*50)

wte_params = gpt.transformer.wte.weight.numel()
lm_head_params = gpt.lm_head.weight.numel()

print(f"Token embedding params: {wte_params:,}")
print(f"LM head params: {lm_head_params:,}")
print(f"Weights are tied: {gpt.transformer.wte.weight.data_ptr() == gpt.lm_head.weight.data_ptr()}")
print(f"Memory saved by weight tying: {wte_params:,} parameters")


# ============================================
# Test Position Embeddings
# ============================================
print("\n" + "="*50)
print("Testing Position Embeddings")
print("="*50)

# Tokens at different positions should have different representations
idx_test = torch.tensor([[5, 5, 5, 5]])  # Same token at different positions
with torch.no_grad():
    gpt.eval()
    # Get embeddings after position encoding
    tok_emb = gpt.transformer.wte(idx_test)
    pos = torch.arange(0, 4)
    pos_emb = gpt.transformer.wpe(pos)
    combined = tok_emb + pos_emb
    print(f"Token embedding (same for all positions): {tok_emb[0, 0, :5]}")
    print(f"Position 0 embedding: {pos_emb[0, :5]}")
    print(f"Position 1 embedding: {pos_emb[1, :5]}")
    print(f"Position embeddings differ: {not torch.allclose(pos_emb[0], pos_emb[1])}")


# stack 12 blocs and check gradient magnitude at first vs last layer 
deep_config= Config()
blocks=torch.nn.Sequential(*[Block(deep_config) for _ in range(12)])

x= torch.randn(2,16,768,requires_grad=True)
out=blocks(x)
loss=out.sum()
loss.backward()

# check gradients at different depths 
for i, block in enumerate(blocks): 
    grad_norm=block.ln_1.weight.grad.norm().item()
    print(f"Blocj {i} gradient : {grad_norm:.6f}")


# Diversity 
# QSN : Do Multiple heads learn different patterns ? 

attn.eval()
x= torch.randn(1,8,768)

with torch.no_grad():
    output, att_weights = attn(x, return_attention=True)
 

if att_weights is not None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(12):
        ax = axes[i // 4, i % 4]
        head = att_weights[0, i].cpu().numpy()
        im = ax.imshow(head, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Head {i}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


# Test both activations
mlp_gelu = MLP(config)
mlp_relu = MLP(config)
mlp_relu.gelu = nn.ReLU()  # Replace GELU with ReLU

x = torch.randn(2, 16, 768, requires_grad=True)

# Compare outputs
out_gelu = mlp_gelu(x)
out_relu = mlp_relu(x)

# Compare gradients
out_gelu.sum().backward()
grad_gelu = x.grad.clone()

x.grad.zero_()
out_relu.sum().backward()
grad_relu = x.grad

print(f"GELU gradient norm: {grad_gelu.norm():.4f}")
print(f"ReLU gradient norm: {grad_relu.norm():.4f}")

# Visualize activation functions
x_range = torch.linspace(-3, 3, 100)
gelu_out = nn.GELU()(x_range)
relu_out = nn.ReLU()(x_range)

plt.plot(x_range, gelu_out, label='GELU')
plt.plot(x_range, relu_out, label='ReLU')
plt.legend()
plt.title('GELU vs ReLU')
plt.show()

# # Does Weight Tying really help ?

# # Create TWO models: one with tying, one without
# from model import GPTConfig
# config = GPTConfig()

# # Model 1: With weight tying (current)
# gpt_tied = GPT(config)

# # Model 2: Without weight tying
# gpt_untied = GPT(config)
# gpt_untied.lm_head.weight = nn.Parameter(torch.randn_like(gpt_untied.lm_head.weight))

# print(f"Tied params: {gpt_tied.get_num_params():,}")
# print(f"Untied params: {gpt_untied.get_num_params():,}")
# print(f"Difference: {gpt_untied.get_num_params() - gpt_tied.get_num_params():,}")

# Attention Entropy over layers  

gpt.eval()
idx= torch.randint(0,1000,(1,16))

entropies=[]
with torch.no_grad():
    tok_emb= gpt.transformer.wte(idx)
    pos_emb= gpt.transformer.wpe(torch.arange(16))
    x=tok_emb+pos_emb

    for i, block in enumerate(gpt.transformer.h):
        # get attetntion weighst from each layer 
        x,att_weights= block(x,return_attention=True)

        # att_weights shape : (B,n_head,T,T) = (1,12,16,16)
        avg_att= att_weights[0].mean(dim=0)

        # calculate entropy for each query position 
        # each row is a prob distribution over keys 
        
        layer_entropies=[]
        for query_pos in range(avg_att.shape[0]):
            probs= avg_att[query_pos] 
            entropy=-(probs*torch.log(probs+1e-9)).sum().item()
            layer_entropies.append(entropy)
        
        # Average entropy across all positions in this layer
        avg_entropy = sum(layer_entropies) / len(layer_entropies)
        entropies.append(avg_entropy)
        print(f"Layer {i+1}: Avg entropy = {avg_entropy:.3f}")

# Plot: Layer depth vs Attention entropy
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(entropies)+1), entropies, marker='o', linewidth=2, markersize=8)
plt.xlabel('Layer Depth', fontsize=12)
plt.ylabel('Average Attention Entropy', fontsize=12)
plt.title('Attention Entropy Across Transformer Layers', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nEntropy trend: {'Decreasing' if entropies[-1] < entropies[0] else 'Increasing'}")
