import torch

a = [1, 2, 3, 4, 5, 6, 7, 8]

b = torch.utils.data.DataLoader(dataset=a)

print(next(iter(b)))
