import torch 
def create_sliding_window_mask(T,window_size=3,sink_size=2): 
    # causal + window 
    mask=torch.tril(torch.ones(T,T,dtype=int))
    # print("mask ",mask)
    band_mask=torch.triu(torch.ones(T,T,dtype=int),diagonal=-(window_size-1)) 
    # print("band mask \n ",band_mask)
    mask= mask&band_mask 
    # attention sinks  
    mask[:,:sink_size]=True
    for i in range(sink_size):
        mask[i,i+1:]=False 
    return mask 



print(create_sliding_window_mask(8,3,2))