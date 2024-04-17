import numpy as np
text=np.load('text.npy')
print(len(text))
vocab=set(text)
vocab_list=list(vocab)
print(vocab_list)