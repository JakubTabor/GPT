import torch
import torch.functional as F
import torch.nn as nn

class FeedForward(nn.Model):
  
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd),
      nn.ReLU(),
    )
    
  def forward(self, x):
    return self.net(x)