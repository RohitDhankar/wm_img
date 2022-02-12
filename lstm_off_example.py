from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    #data = torch.load('traindata.pt')
    data = torch.load('own_data_T_20_L_1000_N_20_.pt')

    print("----initial_data___\n",type(data))


    input = torch.from_numpy(data[3:, :-1])
    print("---input.shape----\n",input.shape)
    target = torch.from_numpy(data[3:, 1:])
    print("---target.shape----\n",target.shape)

    test_input = torch.from_numpy(data[:3, :-1])
    print("---test_input.shape----\n",test_input.shape)
    test_target = torch.from_numpy(data[:3, 1:])
    print("---test_target.shape----\n",test_target.shape)
    


    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000 ## Original 
            #future = 10 ## FOOBAR CHANGED 
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
            print("-------PRED__y_pred.detach().numpy()__",y)

        # draw the result
        plt.figure(figsize=(30,10))
        plt.title(f'PRED_TEST_Loss __>>{loss.item()}', fontsize=10)
        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        #draw(y[2], 'b') ##IndexError: index 2 is out of bounds for axis 0 with size 2
        plt.savefig('pred_official_%d_.pdf'%i)
        plt.close()
