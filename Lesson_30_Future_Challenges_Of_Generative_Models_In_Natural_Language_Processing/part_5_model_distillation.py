class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Define layers for a complex teacher model
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Define layers for a simpler student model
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

def distillation_loss(student_logits, teacher_logits, T=2.0):
    return nn.KLDivLoss()(F.log_softmax(student_logits / T, dim=1), 
                          F.softmax(teacher_logits / T, dim=1)) * (T * T)

def train_distillation(teacher, student, train_loader, num_epochs):
    optimizer_student = optim.Adam(student.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for data, labels in train_loader:
            teacher_output = teacher(data)
            student_output = student(data)
            loss = distillation_loss(student_output, teacher_output)

            optimizer_student.zero_grad()
            loss.backward()
            optimizer_student.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
