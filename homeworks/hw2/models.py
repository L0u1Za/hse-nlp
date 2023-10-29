from torch import nn
import torch

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
    def __init__(self, input_size, output_size, hidden_size=256, device='cpu'):
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size

        self.w_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, output_size)

        self.a_h = nn.Tanh()
        self.a_o = nn.Softmax(dim=0)

    def init_hidden(self):
        return torch.rand(self.hidden_size).to(self.device)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden()
        inputs = torch.cat([hidden, inputs]).to(self.device)
        hidden = self.a_h(self.w_h(inputs))
        outputs = self.a_o(self.w_o(hidden))
        return outputs, hidden

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size=256, bidirectional=False):
        self.hidden = torch.rand(hidden_size)
        self.c = torch.rand(hidden_size)

        self.w_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_c_hat = nn.Linear(input_size + hidden_size, hidden_size)

        self.activation_gates = nn.Sigmoid()
        self.activation = nn.Tanh()

    def forward(self, inputs):
        inputs = torch.cat([self.hidden, inputs])
        f = self.activation_gates(self.w_f(inputs))
        i = self.activation_gates(self.w_i(inputs))
        o = self.activation_gates(self.w_o(inputs))
        c_hat = self.activation(self.w_c_hat(inputs))

        self.c = f * self.c + i * c_hat
        self.hidden = o * self.activation(self.c)

        return self.hidden

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=1, bidirectional=False, device='cpu'):
        self.lstm = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.lstm.append(LSTMLayer(input_size, hidden_size, bidirectional))
            else:
                self.lstm.append(LSTMLayer(hidden_size, hidden_size, bidirectional))
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    def forward(self, inputs):
        pass

class RNNWrapper(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, hidden_size=256, num_layers=1, bidirectional=False, type='rnn', device='cpu'):
        super().__init__()

        self.output_size = output_size

        self.embed = nn.Embedding(vocab_size, embed_size)

        if (type == 'rnn'):
            self.model = RNN(embed_size, output_size, hidden_size, device=device)
        elif (type == 'lstm'):
            self.model = LSTM(embed_size, output_size, hidden_size, num_layers, bidirectional, device=device)

    def forward(self, inputs):
        embeds = self.embed(inputs)
        outputs = torch.zeros(len(embeds), self.output_size)
        for i, embeds_b in enumerate(embeds):
            hidden=None
            for embed in embeds_b:
                output, hidden = self.model(embed, hidden)
            print(output)
            outputs[i] = output
        return outputs


if __name__ == "__main__":
    import torch
    model = RNNWrapper(vocab_size=3, embed_size=16, output_size=2)

    batch = torch.randint(low=0, high=2, size=(1, 3))

    print(model(batch))