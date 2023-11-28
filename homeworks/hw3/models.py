from torch import nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, kernel_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.conv = nn.Sequential(
            nn.ZeroPad2d((kernel_size - 1, 0, 0, 0)),
            nn.Conv1d(embed_size, hidden_size, kernel_size=kernel_size, padding=0)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.Dropout()
        )

    def forward(self, inputs):
        outputs = self.embed(inputs)

        outputs = outputs.transpose(1, 2)
        outputs = self.conv(outputs)

        outputs = outputs.transpose(1, 2)
        outputs = self.fc(outputs)

        return outputs