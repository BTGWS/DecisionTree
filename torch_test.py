import math

import torch
import numpy as np

tensor = torch.ones(4, 4)
tensor[:, 1] = 0  # 将第1列(从0开始)的数据全部赋值为0
print(tensor)
# 逐个元素相乘结果
print(f"tensor.mul(tensor): \n {tensor.mul(tensor)} \n")
# 等价写法:
print(f"tensor * tensor: \n {tensor * tensor}")

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
a = 8 / 14
b = 6 / 14
c = -math.log2(a) * a - math.log2(b) * b
print(c)
