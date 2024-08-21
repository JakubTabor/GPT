import torch
import torch.nn as nn

class Block(nn.Module):
  
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(e_embd)
    self.ln2 = nn.LayerNorm(e_mebd)
    
  def forward(self, x):
    # Making x = x + self-attention and feed-forward, so we fork-off and do communication, then come back
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x