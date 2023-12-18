from torch import nn
import torch

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


class RNNWrapper(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, hidden_size=256, num_layers=1, bidirectional=False, type='rnn'):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(vocab_size, embed_size)

        if (type == 'rnn'):
            self.model = nn.RNN(embed_size, hidden_size, num_layers)
        elif (type == 'lstm'):
            self.model = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        embeds = self.embed(inputs)
        outputs = torch.zeros(inputs.shape[0], inputs.shape[1], self.output_size).to(inputs.device)

        hidden=None
        for sequence_i in range(embeds.shape[1]): # iterate through sequence
            embed = embeds[:, sequence_i, :].squeeze(1)
            output, hidden = self.model(embed, hidden)
            output = self.fc(output)
            for i in range(embeds.shape[0]):
                outputs[i][sequence_i] = output[i]

        return outputs

