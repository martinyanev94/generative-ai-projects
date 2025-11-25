class MiniBatchDiscrimination(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MiniBatchDiscrimination, self).__init__()
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.randn(input_dim, num_classes))

    def forward(self, x):
        # Calculate distances between samples in mini-batch
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        distance = torch.exp(-torch.sum(diff ** 2, dim=-1))
        output = torch.sum(distance, dim=-1)
        return output
