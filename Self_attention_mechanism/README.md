# In self-attention mechanism there is no notion of space  
* In self-attention there is just set of vectors out in space
* They can communicate and if we want them to know to have notion of space we need to add it
* Which we done when we calculate positional encoding and add that information to the vectors

# The elements across batch-dim, which are independent examples never talk to each other, they are processed independently
* wei (q @ k) apply matrix multiplication in parallel across batch-dim, in fact we have 4 separate pools of 8 tokens
* This 8 tokens talk to each other, but in total there are 32 tokens, that are being processed, but in 4 separate pools

# What we implemented here is a decoder, because it's decoding language, where we need to mask with tril matrix, so the tokens from the future never talk to the past
* Because they would give away the answer
* If we por example build the sentiment analysis with a transformer, we may have all tokens talk to each other fully
* In that case we will use encoder block, which means that we just need to delete the masking part of our code

# In our code we use self-attention, but also exist cross-attention, whats the difference?
* In self-attention keys, queries and values comes from one source - x, so this tokens are self-attending
* But in encoder, decoder transformers, we can have a case when queries are produced from x, but keys and values come from whole separate external source
* Por example from encoder blocks, that encode some context, that we would like to condition on
* So cross-attention is used when, there is separate source of tokens we'd like to pull informations from into our tokens 
