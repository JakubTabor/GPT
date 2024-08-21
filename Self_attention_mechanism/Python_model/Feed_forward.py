import torch
import torch.functional as F
import torch.nn as nn

class FeedForward(nn.Model):
  
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 *  n_embd),
      nn.ReLU(),
      # This is the projection layer going back to residual path-way
      nn.Linear( 4 * n_embd, n_embd),
    )
    
  def forward(self, x):
    return self.net(x)