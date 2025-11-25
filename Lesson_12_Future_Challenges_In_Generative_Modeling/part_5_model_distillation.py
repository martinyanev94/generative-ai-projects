class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Linear(100, 256)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(100, 128)  # Smaller output dimension

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def distill(teacher, student, data_loader, temperature=2.0):
    for real_images, _ in data_loader:
        teacher_outputs = teacher(real_images) / temperature
        student_outputs = student(real_images) / temperature
        loss = nn.KLDivLoss()(F.log_softmax(student_outputs), F.softmax(teacher_outputs))
        # Update student model using the computed loss
