from __future__ import annotations 
import torch 
from torch.nn import Module 
from torch import nn 
import torch.nn.functional as F 

from slot_attention import SlotAttention 
from multi_head_slot_attention import MultiHeadSlotAttention 

def log(t,eps = 1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_softmax(logtis,temperature=1.):
    dtype,size = logtis.dtype, logtis.shape[-1]
    assert temperature >0
    scaled_logits =logtis/temperature 
    noise_logits = scaled_logits + gumbel_noise(scaled_logits)
    
    indices = noise_logits.argmax(dim=-1)
    
    hard_one_hot = F.one_hot(indices,size).type(dtype)
    
    soft = scaled_logits.softmax(dim=-1)
    
    hard_one_hot = hard_one_hot + soft - soft.detach() 
    
    return hard_one_hot,indices 

class AdaptiveSlotWrapper(Module):
    def __init__(self,slot_attn:SlotAttention|MultiHeadSlotAttention,temperature=1.):
        super().__init__()
        self.slot_attn = slot_attn 
        dim =slot_attn.dim 
        self.temperature = temperature 
        self.pred_keep_slot = nn.Linear(dim,2,bias=False)
    
    def forward(self,x,**slot_kwargs):
        slots = self.slot_attn(x,**slot_kwargs)
        keep_slot_logtis = self.pred_keep_slot(slots)
        keep_slots,_ = gumbel_softmax(keep_slot_logtis,temperature=self.temperature)
        keep_slots = keep_slots[...,-1] 
        return slots,keep_slots
    
    
if __name__ == "__main__":
    x = torch.randn(1,3,2)
    size =x.shape[-1]
    indices = x.argmax(dim=-1)
    print(indices.shape)
    hard_one_hot = F.one_hot(indices,size)
    print(hard_one_hot.shape)