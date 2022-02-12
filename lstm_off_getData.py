import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000 # 1000
N = 20 # 100

#T_20_L_100_N_2_
#T_20_L_1000_N_20_


x = np.empty((N, L), 'int64') #
# Empty NumpyArray of shape --> N,L == Number of Arrays , Count of Elements in Each Array 

print("---x.shape----\n",x.shape)
print("---x.ndim----\n",x.ndim) # 2 - 2D_ARRAY with SHAPE ---> N,L ->> 2,10
print("---x[:5]----\n",x[:5])
print("  "*60)
print("---x[:1]----\n",x[:1])
print("  "*60)
print("---x[:2]----\n",x[:2])
print("  "*60)
print("---x[:-10]----\n",x[:-10])


x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)

data = np.sin(x / 1.0 / T).astype('float64')

print("---data.shape----\n",data.shape)
print("---data[:5]----\n",data[:5])
print("---data[:-10]----\n",data[:-10])
print("------type(data)----",type(data)) ##- <class 'numpy.ndarray'>

torch.save(data, open('own_data_T_20_L_1000_N_20_.pt', 'wb'))
## traindata.pt -- 1.3MB -- ZIP File 