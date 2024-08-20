import torch 
import torch.nn as nn
import torch.functional as F

batch_size = 64 
block_size = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()
  
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

class Head(nn.Module):
  
  def __init__(self, head_size):
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout)
    
  def __forward__(self, x):
    
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    
    wei = q @ k.transpose(-2, -1) * k.shape[-1]*-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def __forward__(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
  
  
class BigramLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.sa_heads = MultiHeadAttention(4, n_embd/4)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    
  def forward(self, idx, targets=None):
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.sa_heads(x)
    logits = self.lm_head(x)
    
    if targets is None:
      loss = None
    else: 
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
      
    return idx
  
model = BigramLanguageModel()
n = model.to(device)