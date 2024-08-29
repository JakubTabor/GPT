# Now we are gonna take a look at Multi-Head Attention
# It is applying multiple attentions in parallel, then concatenating there results in channel-dim (-1)
* The original single-attention have have head size n_embd 32 
* And we instead of having one communication channel, we have four of them 
* We want to have 8-dim self-attention n_embd/4
* So we have four 8-dim vectors,(4, n_embd/4) which gives us original number 32

# Multi-head attention allows the model to  attend to information from different representation at different positions. With a single attention head, averaging inhibits this
# In other way we can call it group convolution
![](https://github.com/JakubTabor/GPT/blob/main/Images/convolutions_vs_group_convolutions.png)


# Feed-forward is little single layer followed by relu non-linearity
* We went way to fast to calculate logits, so tokens look at each other buy don't have much time to think on what they found from the other tokens
* Its called sequentially after self-attention, so it self-attend then it feed-forward 

# Feed-forward based on per token level, so once self-attention, which is communication, gather all data it need to think on that data individually

# Very deep neural networks suffer from optimization issues, so we need to introduce residual connections
![](https://github.com/JakubTabor/GPT/blob/main/Images/Residual_connection.png)
* Basically it means that we transform the data, but then we have skip connection with addition from the previous features
* So we go from inputs to targets via residual pathway and we are free to fork-off from it to perform the computations
* Then project back to residual pathwayvia addition
* It is useful during backpropagation, because addition distributes gradients equally to both of its branches, that feed the input
* So it allow the gradients to have straight way that goes directly from supervision to input

# I introduce Layer-Norm, which is similar to Batch Normalization
* We just do normalization across rows, not columns
* Also we don't distinguished between training and test, we also don't need buffers
* Because the computation do not span across the examples

# Size of Layer-Norm is n_embed, so 32
* When Layer-Norm is normalizing our features, the mean and variance is taken over 32 numbers
* So batch and time act as batch-dimensions, this is like per token transformation, that normalizes the features
* And make them unit mean and unit gaussian at initialization
