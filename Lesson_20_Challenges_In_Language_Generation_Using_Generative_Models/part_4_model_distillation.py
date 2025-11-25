import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# A hypothetical teacher model
teacher_model = SimpleModel(input_dim=100, output_dim=10)

# A student model (distilled version)
student_model = SimpleModel(input_dim=100, output_dim=10)

# Distillation process
def distill(teacher_model, student_model, data_loader):
    teacher_model.eval()  # Teacher in evaluation mode
    student_model.train()  # Student in training mode

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(student_model.parameters())

    for data in data_loader:
        with torch.no_grad():
            teacher_output = teacher_model(data)

        student_output = student_model(data)
        loss = criterion(student_output, teacher_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Distillation complete.")
