
### FOOBAR_Check_if_Same_Memory_

import torch
## CUDA Driver is available 
print(torch.cuda.is_available()) ## True 



# Everything in pytorch is based on Tensor operations.
# A tensor can have different dimensions
# so it can be 1d, 2d, or even 3d and higher

# scalar, vector, matrix, tensor

# torch.empty(size): uninitiallized
#x = torch.empty(1) # scalar
#print(x) #tensor([-1.7462e-11])

#x = torch.empty(3) # vector, 1D
#print(x) #tensor([0., 0., 0.])


# x = torch.empty(2,5) # matrix, 2D
# print(x)

## below -- (2,5)
# tensor([[-3.2019e-09,  4.5619e-41, -3.1959e-09,  4.5619e-41, -3.1952e-09],
#         [ 4.5619e-41, -3.1959e-09,  4.5619e-41, -1.6779e-08,  4.5619e-41]])
## below -- (2,3)
# tensor([[3.0304e+35, 1.8470e+31, 3.3702e-12],
#         [8.1578e-33, 1.3563e-19, 1.3563e-19]])



# x = torch.empty(1,1,5) # tensor, 3 dimensions
# print(x)


# tensor([[[1.9684e-19, 1.8589e+34, 1.8888e+31, 1.8891e+31, 1.3556e-19],
#          [1.3563e-19, 1.3563e-19, 1.3563e-19, 6.3739e+31, 2.9758e+29]],

#         [[1.9203e+31, 2.6800e+20, 6.7541e+34, 4.4721e+21, 2.6762e+20],
#          [7.3078e+28, 1.6926e+22, 6.8608e+22, 7.5631e+28, 2.1136e-10]]])

# tensor([[[0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [0.0000e+00, 0.0000e+00, 0.0000e+00]],

#         [[0.0000e+00, 0.0000e+00, 1.4013e-45],
#          [0.0000e+00, 0.0000e+00, 0.0000e+00]]])

#x = torch.empty(2,2,2,3) # tensor, 4 dimensions
#print(x)


# torch.rand(size): random numbers [0, 1]
# x = torch.rand(5, 3)
# print(x)



# # torch.zeros(size), fill with 0
# # torch.ones(size), fill with 1
# x = torch.zeros(5, 3)
# print(x)

# # check size
# print(x.size())

# # check data type
# print(x.dtype)

# # specify types, float32 default
# x = torch.zeros(5, 3, dtype=torch.float16)
# print(x)

# # check type
# print(x.dtype)

# # construct from data
# x = torch.tensor([5.5, 3])
# print(x.size())

# # requires_grad argument
# # This will tell pytorch that it will need to calculate the gradients for this tensor
# # later in your optimization steps
# # i.e. this is a variable in your model that you want to optimize
# x = torch.tensor([5.5, 3], requires_grad=True)


## 12:30_TimeLine

# Operations
# y = torch.rand(2, 2)
# x = torch.rand(2, 2)
# print(id(y))
# print(id(x))

"""
139653067627728
139653067627888
"""


# elementwise addition
# z = x + y
# print(id(y))
# print(id(x))
# print(id(z))

"""
140153464806768 -- print(id(y))
140153464806128 -- print(id(x))
140153464806768 -- print(id(y))
140153464806128 -- print(id(x))
140153464804848 -- print(id(z))
"""

# # torch.add(x,y)

# # in place addition, everythin with a trailing underscore is an inplace operation
# # i.e. it will modify the variable
# # y.add_(x)

# # substraction
# z = x - y
# z = torch.sub(x, y)

# # multiplication
# z = x * y
# z = torch.mul(x,y)

# # division
# z = x / y
# z = torch.div(x,y)


## 15:26__TimeLine
# # Slicing---similar to NUMP Arrays --(5,3) 5-ROWS and 3-COLUMNS
x = torch.rand(5,3)
# print(x)
# print(x[:, 0]) # all rows, column 0
"""
tensor([[0.8246, 0.7389, 0.1673],
        [0.7929, 0.2824, 0.3094],
        [0.8261, 0.3785, 0.6233],
        [0.8745, 0.8505, 0.9347],
        [0.3323, 0.8340, 0.5433]])
tensor([0.8246, 0.7929, 0.8261, 0.8745, 0.3323])
"""

### FOOBAR_Check_if_Same_Memory_
#https://discuss.pytorch.org/t/any-way-to-check-if-two-tensors-have-the-same-base/44310/9

# print(x[1, :]) # row 1, all columns
# print(x[1,1]) # element at 1, 1


# # Get the actual value if only 1 element in your tensor
print(x[1,1].item()) ## FOOBAR_this Value will change as its a RANDOM Number -- rand()
#0.27338242530822754
#0.49868154525756836






# # Reshape with torch.view()
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# # if -1 it pytorch will automatically determine the necessary size
# print(x.size(), y.size(), z.size())

# # Numpy
# # Converting a Torch Tensor to a NumPy array and vice versa is very easy
# a = torch.ones(5)
# print(a)

# # torch to numpy with .numpy()
# b = a.numpy()
# print(b)
# print(type(b))

# # Carful: If the Tensor is on the CPU (not the GPU),
# # both objects will share the same memory location, so changing one
# # will also change the other
# a.add_(1)
# print(a)
# print(b)

# # numpy to torch with .from_numpy(x)
# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a)
# print(b)

# # again be careful when modifying
# a += 1
# print(a)
# print(b)

# # by default all tensors are created on the CPU,
# # but you can also move them to the GPU (only if it's available )
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     # z = z.numpy() # not possible because numpy cannot handle GPU tenors
#     # move to CPU again
#     z.to("cpu")       # ``.to`` can also change dtype together!
#     # z = z.numpy()
