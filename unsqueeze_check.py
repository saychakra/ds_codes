import torch

score = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(score)

print(score.unsqueeze(0))
print(score.unsqueeze(1))
print(score.unsqueeze(2))