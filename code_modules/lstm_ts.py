
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

        # Will Split the Tensor -- x -- into Chunks 
        # Each Chunk is a View of the Tensor 

        for input_t in torch.split(x, 1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) ## Input here is -- Hidden State OutPut from earlier Cell -->> h_t
            output_linear_1 = self.linear(h_t2) ## output of the Linear Layer # Input is -- Hidden State OutPut from earlier Cell -->> h_t2
            outputs.append(output_linear_1)

        for iter_k in range(future):
            h_t, c_t = self.lstm1(output_linear_1, (h_t, c_t)) # output_linear_1 -- from Linear Layer above is INPUT of 1st LSTMCell 
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        print("--forward--type(outputs[1])----",type(outputs[0]))
        print("--forward--outputs[1])----\n",outputs[0])

        outputs = torch.cat(outputs, dim=1) # Concatenates the given sequence of tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        return outputs

if __name__ == "__main__":
    # when - shape y = 100 , 1000
    train_input = torch.from_numpy(y[3:, :-1]) # this shape (97 , 999) - below also same shape (97 , 999)
    train_output = torch.from_numpy(y[3:, 1:]) # this is SHIFTED Data >> train_input is SHIFTED 

    test_input = torch.from_numpy(y[:3, :-1]) # Elements at index -- 0,1,2
    test_output = torch.from_numpy(y[:3, 1:])  # this shape (3 , 999) - above also same shape (3 , 999)   
    ## 3 SAMPLES and 999 VALUES 

    n_hidden = 51
    model = lstm_Model(n_hidden=n_hidden)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8) # pass in Learning Rate as a Variable  - lr
    ## LBFGS -- is diff from the ADAM Optimizer ?? How ?? What ??
    ## https://en.wikipedia.org/wiki/Limited-memory_BFGS

    n_steps = 30 # n_steps == how many rounds of TRAINING 

    for iter_k in range(n_steps):
        print("Step___in n_steps__>>", iter_k)

        def closure():
            optimizer.zero_grad() ## Empty the GRADIENTS -- ZERO the GRADIENTS --- when you start your training loop, 
            #ideally you should zero out the gradients so that you do the parameter update correctly -- SO -- https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            output = model(train_input)
            loss = criterion(output, train_output)
            ##criterion = nn.MSELoss() -- this is done above 
            print("Loss", loss.item())
            loss.backward() # this -- backward -- applies the Back Propagation 
            # default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call. 
            # TODO -->> SO -- https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            return loss
        
        optimizer.step(closure) # passing - closure - without any Params to >> optimizer
        ## Training done -- now Predict -- on the TEST Data >> test_input
        print("-----Training done ------ now Predict on >> test_input")

        
        ## Predict on >> test_input
        with torch.no_grad(): # Without Gradients 
            future = 1000
            pred = model(test_input, future=future) # this here - PRED will also include the -- future Values 
            loss = criterion(pred[:, :-future], test_output) # from this here -- PRED -- we Delete the FUTURE Vals
            #criterion = nn.MSELoss() -- this is done above 
            print("Test Loss", loss.item())
            y = pred.detach().numpy() ## .detach() -->> Returns a new Tensor, detached from the current graph.
            # detach the tensor and convert to Numpy Array -->> https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            print("--SHAPE >> pred.detach().numpy()---",y.shape)

        
        # Plot -- ACTUAL vs. PREDICT 
        plt.figure(figsize=(12, 6))
        plt.title(f"Step {iter_k+1}")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1]

        def draw(y_i, color):
            plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0) ## ACTUAL 
            plt.plot(np.arange(n, n+future), y_i[n:], color + ":", linewidth=2.0) ## PREDICT

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')

        plt.savefig("pred_step_%d_.pdf" % iter_k)
        plt.close()















        


