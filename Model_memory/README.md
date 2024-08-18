![](https://github.com/JakubTabor/GPT/blob/main/Images/model_memory.png)
# First i create the memory of model by average the past context  
* We are gonna for every single batch element independently and every token in that sequence
* We calculate average of all vectors in all previous tokens and at also this token
![](https://github.com/JakubTabor/GPT/blob/main/Images/model_memory_softmax.png)
