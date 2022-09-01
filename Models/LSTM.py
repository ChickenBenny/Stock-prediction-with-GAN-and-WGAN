import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop_out, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.dropout = nn.Dropout(drop_out)
        if activation == 'Linear':
            self.fc = nn.Linear(hidden_dim, output_dim)
        elif activation == 'Sigmoid':
            self.fc = nn.Sequential(hidden_dim, output_dim)
        elif activation == 'Relu':
            self.fc = nn.ReLU(hidden_dim, output_dim)
        elif activation == 'Elu':
            self.fc = nn.ELU(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

        