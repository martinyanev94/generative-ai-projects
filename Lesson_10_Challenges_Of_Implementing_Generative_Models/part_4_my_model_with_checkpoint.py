class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(100, 256)
        self.layer2 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = checkpoint(self.layer2, x)  # Gradient checkpointing
        return x
