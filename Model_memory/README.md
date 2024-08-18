# First i create the memory of model by average the past context  
# We are gonna for every single batch element independently and every token in that sequence
# We calculate average of all vectors in all previous tokens and at also this token
![](https://github.com/JakubTabor/GPT/blob/main/Images/model_memory.png)

# Then i gonna rewrite it using softmax, that's the final version which i use in my code to develop the self attention block
# Now we can do weighted aggregation of the past elements by using matrix multiplication of a lower triangular fashion
# And the elements in lower triangular part are telling how much of each element fuzes into these position
![](https://github.com/JakubTabor/GPT/blob/main/Images/model_memory_softmax.png)
