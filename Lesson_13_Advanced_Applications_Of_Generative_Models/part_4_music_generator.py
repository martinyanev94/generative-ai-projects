import torch.nn.functional as F

class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicGenerator, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

music_generator = MusicGenerator(input_size=notes_count, hidden_size=256, output_size=notes_count).to(device)
