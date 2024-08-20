  # Feed-forward is little single layer followed by relu non-linearity
  * We went way to fast to calculate logits, so tokens look at each other buy don't have much time to think on what they found from the other tokens
  * Its called sequentially after self-attention, so it self-attend then it feed-forward 
  
  # Feed-forward based on per token level, so once self-attention, which is communication, gather all data it need to think on that data individually

  
