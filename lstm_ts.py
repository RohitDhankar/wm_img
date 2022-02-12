
## FOOBAR -- TODO -->> https://github.com/search?q=torch+x%5B%3A%5D+%3D+np.array&type=code
## torch x[:] = np.array

# import numpy as np 
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


N = 100
L = 1000
T = 20

x = np.empty((N,L),np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
# above reshape to add to the other Array
y = np.sin(x/1.0/T).astype(np.float32)
print("here")

print(x.shape)
print(y.shape)

plt.figure(figsize=(10,8))
plt.title("Sine wave")
plt.xlabel("X")
plt.ylabel("Y")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
print("------x.shape[1]---------",x.shape[1])
plt.plot(np.arange(x.shape[1]), y[0,:], 'r', linewidth=2.0)
#plt.show()

class lstm_Model(nn.Module):
    def __init__(self, n_hidden=51):
        super(lstm_Model,self).__init__() # Why not any Params inside the __init__()
        self.n_hidden = n_hidden

        self.lstm1 = nn.LSTMCell(1, self.n_hidden) 
        # CELL_1 - with INPUT ==1 , as we feed it the TS values 1 by 1 ?? 
        # CELL_1 - with OUTPUT == n_hidden
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        # CELL_2 - with INPUT == n_hidden
        # CELL_2 - with OUTPUT == n_hidden
        self.linear = nn.Linear(self.n_hidden, 1)
        # LINEAR_LAYER - with INPUT == n_hidden
        # LINEAR_LAYER - with OUTPUT == 1

    def forward(self,x,future=0):
        """
        Args:
            x ([type]): This is the TENSOR with the Actual Data 
            future (int, optional): Dhankar-->> How many OutOfSample to Forecast?. Defaults to 0.
        """
        print("----forward---type(x)--------",type(x))

        outputs = []
        n_samples = x.shape[0]

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32) # Initial >> Hidden-State 
        ##https://pytorch.org/docs/stable/generated/torch.zeros.html
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32) # Initial >> Cell-State 

        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32) # Initial >> Hidden-State - for the 2nd LSTM Cell 
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32) # Initial >> Cell-State - for the 2nd LSTM Cell 

        ## We will go over the TENSOR - 1 element at a time - thats why above in the Class 
        # we have CELL_1 - with INPUT ==1

        # Will Split the Tensor into Chunks 
        # Each Chunk is a View of the Tensor 



        


