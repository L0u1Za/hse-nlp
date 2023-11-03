from torch import nn
import torch
from time import time
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.conv = nn.Sequential(
            nn.Conv1d(embed_size, embed_size, kernel_size=5)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 2),
            nn.Dropout()
        )

    def forward(self, inputs, seq_lengths):
        outputs = self.embed(inputs).transpose(1, 2)

        outputs = self.conv(outputs)

        outputs, _ = outputs.max(dim=2)

        outputs = self.fc(outputs)

        return outputs

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, device='cpu'):
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size

        self.w_h = nn.Linear(input_size + hidden_size, hidden_size)

        self.a_h = nn.Tanh()

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(self.device)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(inputs.shape[0])
        inputs = torch.cat([hidden, inputs], dim=1).to(self.device)

        hidden = self.a_h(self.w_h(inputs))

        return hidden, hidden

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size=256, bidirectional=False, device='cpu'):
        super().__init__()

        self.hidden_size = hidden_size
        self.device = device

        self.w_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_c_hat = nn.Linear(input_size + hidden_size, hidden_size)

        self.activation_gates = nn.Sigmoid()
        self.activation = nn.Tanh()
    def init_hidden_c(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(self.device), torch.zeros(batch_size, self.hidden_size).to(self.device)

    def forward(self, inputs, hidden_c=None):
        #print("==(+Layer)==")
        #print(inputs.shape)
        if (hidden_c is None):
            hidden, c = self.init_hidden_c(inputs.shape[0])
        else:
            hidden, c = hidden_c
        inputs = torch.cat([hidden, inputs], dim=1).to(self.device)
        f = self.activation_gates(self.w_f(inputs))
        i = self.activation_gates(self.w_i(inputs))
        o = self.activation_gates(self.w_o(inputs))
        c_hat = self.activation(self.w_c_hat(inputs))

        c = f * c + i * c_hat
        hidden = o * self.activation(c)
        #print("==(-Layer)==")
        return hidden, (hidden, c)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, bidirectional=False, device='cpu'):
        super().__init__()

        self.lstm = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lstm.append(LSTMLayer(input_size, hidden_size, bidirectional, device=device))
            else:
                self.lstm.append(LSTMLayer(hidden_size, hidden_size, bidirectional,  device=device))

    def forward(self, inputs, hidden=None):
        #print("==(+LSTM)==")
        #print(inputs.shape)
        if (hidden is None):
            hidden = [None] * len(self.lstm)
        output = inputs
        #print("Num layers", len(self.lstm))
        for i, layer in enumerate(self.lstm):
            #print("Layer", i + 1)
            hidden_state = hidden[i]
            output, hidden_state = layer(output, hidden_state)
            hidden[i] = hidden_state
        #print(output.shape)
        #print("==(-LSTM)==")
        return output, hidden

class RNNWrapper(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, hidden_size=256, num_layers=1, bidirectional=False, type='rnn', device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(vocab_size, embed_size)

        if (type == 'rnn'):
            self.model = RNN(embed_size, hidden_size, device=device)
        elif (type == 'lstm'):
            self.model = LSTM(embed_size, hidden_size, num_layers, bidirectional, device=device)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs, seq_lengths):
        embeds = self.embed(inputs)
        outputs = torch.zeros(len(embeds), self.hidden_size).to(self.device)

        hidden=None
        for sequence_i in range(embeds.shape[1]): # iterate through sequence
            embed = embeds[:, sequence_i, :].squeeze(1)
            output, hidden = self.model(embed, hidden)

            for i in range(len(seq_lengths)):
                if sequence_i + 1 == seq_lengths[i]:
                    outputs[i] = output[i]

        outputs = self.fc(outputs)
        return outputs


if __name__ == "__main__":
    import torch
    model = RNNWrapper(vocab_size=3, embed_size=16, output_size=2)

    batch = torch.randint(low=0, high=2, size=(1, 3))

    print(model(batch))