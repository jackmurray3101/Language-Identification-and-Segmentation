import torch

x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
print(x.size())
y = x[0, 0:3]
print(y)