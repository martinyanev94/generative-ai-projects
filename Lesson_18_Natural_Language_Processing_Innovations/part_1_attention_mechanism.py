import torch
import torch.nn.functional as F

def attention(Q, K, V):
    scores = torch.matmul(Q, K.T) / (Q.size(-1) ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights

# Sample input representation
Q = F.embedding(torch.tensor([1, 2]), torch.randn(3, 4)) # Query
K = F.embedding(torch.tensor([1, 2]), torch.randn(3, 4)) # Key
V = F.embedding(torch.tensor([1, 2]), torch.randn(3, 4)) # Value

output, weights = attention(Q, K, V)
print(output)
print(weights)
