# The attention pattern we use during training also matters. Let's have a look at attention masking 
import torch
import torch.nn as nn 
from torch.nn import functional as F 

# ------------ Causal Masking ------------------ 

class Causal_Masking(nn.Module): 
    """ 
    For a Given Matrix The current token only has a access to the current and the past tokens
    """
    def __init__(self,config): 
        super().__init__()

        self.config=config

        mask= torch.tril(torch.ones(config.block_size,config.block_size))
        self.register_buffer("mask",mask.view(1,1,config.block_size,config.block_size))

    def forward(self,seq_len=None): 
        if seq_len is None : 
            return self.mask 
        return self.mask[:, :, :seq_len, :seq_len] 

class Intra_Document_Masking(nn.Module): 
    """
    Idea of building: 
    Step 1: Detect the boundries of the docs using the special tokens. [Doc_IDs]
    Step 2: j <=i and doc_id[i] == doc_id[j]
    Step 3: Building the mask 
    Given Doc IDs : [1,1,1,2,2,2,3,3,4]
    Create "Same Document" Matrix : 
    compare each position with every other position 
    same_doc[i,j]=(doc_id[i]==doc_id[j])
    create a causal mask 
    causal[1,j]=(j<=i)
    finally combine two matrices like element wise multiplication
    """
    def __init__(self,config): 
        super().__init__()
        self.config= config

    @staticmethod
    def get_document_ids(tokens,eos_token_id): 
        """ 
        Convert token sequence to document IDs 
        """
        is_eos=(tokens==eos_token_id) 
        # cumsum after each eos increments the document id, 
        # shift right so eos belong to its document
        document_ids= is_eos.cumsum(dim=1)
        document_ids = torch.cat([
                torch.zeros_like(document_ids[:, :1]),   # Take first column shape, fill with 0
                document_ids[:, :-1]                      # Take all but last column
            ], dim=1)
        return document_ids 

    def forward(self,document_ids): 

        seq_len=document_ids.size(1)
        # same document matrix (batch, seq_len,seq_len)
        same_doc=document_ids.unsqueeze(2)==document_ids.unsqueeze(1)

        # causal mask: (seq_len,seq_len)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=document_ids.device))

        mask= same_doc&causal.bool()

        return mask.unsqueeze(1)
    
