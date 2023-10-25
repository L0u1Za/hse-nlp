from torch import nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5),
            nn.Conv1d(4, 8, kernel_size=5)
        )
        self.fc = nn.Sequential(
            nn.Linear(8, 2),
            nn.Dropout()
        )

    def forward(self, inputs):
        outputs = self.embed(inputs).unsqueeze(1).flatten(2, 3)

        outputs = self.conv(outputs)

        outputs, _ = outputs.max(dim=2)

        outputs = self.fc(outputs)

        return outputs

class RNN(nn.Module):
    def __init__(self, hidden_size=256):
        pass

    def forward(self, inputs):
        pass

class LSTM(nn.Module):
    def __init__(self):
        pass
    def forward(self, inputs):
        pass

if __name__ == "__main__":
    import torch
    model = CNN(vocab_size=3, embed_size=64)

    batch = torch.randint(low=0, high=2, size=(1, 3))

    print(model(batch))