# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# ファイルやディレクトリの操作を行うための標準ライブラリ
import os


# import pytorch library
import torch

# import variable from pytorch library for back propagation
from torch.autograd import Variable


# ..は親ディレクトリを表す
print(os.listdir("../data"))


# numpy array
array = [[1, 2, 3], [4, 5, 6]]
first_array = np.array(array)
print("Array Type: {}".format(type(first_array)))
print("Array Shape: {}".format(np.shape(first_array)))
print(first_array)

# pytorch array
tensor = torch.Tensor(array)
print("Array Type: {}".format(tensor.type))
print("Array Shape: {}".format(tensor.shape))
print(tensor)

# numpy ones
print("Numpy {}\n".format(np.ones((2,3))))

# pytorch ones
print(torch.ones((2,3)))

# random numpy array
array = np.random.rand(2,2)
print("{} {}\n".format(type(array),array))

# from numpy to tensor
from_numpy_to_tensor = torch.from_numpy(array)
print("{}\n".format(from_numpy_to_tensor))

# from tensor to numpy
tensor = from_numpy_to_tensor
from_tensor_to_numpy = tensor.numpy()
print("{} {}\n".format(type(from_tensor_to_numpy),from_tensor_to_numpy))


# create tensor 
tensor = torch.ones(3,3)
print("\n",tensor)

# Resize
print("{}{}\n".format(tensor.view(9).shape,tensor.view(9)))

# Addition
print("Addition: {}\n".format(torch.add(tensor,tensor)))

# Subtraction
print("Subtraction: {}\n".format(tensor.sub(tensor)))

# Element wise multiplication
print("Element wise multiplication: {}\n".format(torch.mul(tensor,tensor)))

# Element wise division
print("Element wise division: {}\n".format(torch.div(tensor,tensor)))

# Mean
tensor = torch.Tensor([1,2,3,4,5])
print("Mean: {}".format(tensor.mean()))

# Standart deviation (std)
print("std: {}".format(tensor.std()))


# define variable
# 最新pytorchでは Variable を明示しなくても使える。非推奨
# requires_grad はテンソルに対する勾配の追跡の設定 var.gradに保存される
var = Variable(torch.ones(3), requires_grad = True)
var


# lets make basic backward propagation
# we have an equation that is y = x^2
array = [2, 4]
tensor = torch.Tensor(array)
x = Variable(tensor, requires_grad=True)
y = x**2
print("y = ", y)

# recap o equation o = 1/2+sum(y)
o = (1/2)*sum(y)
print("o = ",o)

# backward
o.backward() # calculates gradients

print("gradients:", x.grad)