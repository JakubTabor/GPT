# In self-attention mechanism there is no notion of space  
* In self-attention there is just set of vectors out in space
* They can communicate and if we want them to know to have notion of space we need to add it
* Which we done when we calculate positional encoding and add that information to the vectors

# The elements across batch-dim, which are independent examples never talk to each other, they are processed independently
* wei (q @ k) apply matrix multiplication in parallel across batch-dim, in fact we have 4 separate pools of 8 tokens
* This 8 tokens talk to each other, but in total there are 32 tokens, that are being processed, but in 4 separate pools

# What we implemented here is a decoder, because it's decoding language, where we need to mask with tril matrix, so the tokens from the future never talk to the past
* Because they would give away the answer
