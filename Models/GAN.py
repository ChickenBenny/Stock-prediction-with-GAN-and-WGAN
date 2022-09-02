import torch
import torch.nn as nn

class generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop_out):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.act(out)
        return out


class discriminator(nn.Module):
    def __init__(self, config, batch_size):
        super().__init__()

        self.batch_size = batch_size
        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                nn.Conv1d(config[i - 1], config[i], kernel_size = 3, stride = 1, padding = 'same'),
                nn.LeakyReLU(),
                nn.BatchNorm1d(config[i])
            )
        )
        self.conv = nn.Sequential(*modules)
      
        modules = []
        modules.append(
            nn.Sequential(
                nn.Linear(config[-1], 200),
                nn.LeakyReLU(),
                nn.BatchNorm1d(200),
                nn.Linear(200, 200),
                nn.LeakyReLU(),
                nn.Linear(200, 1),
                nn.Sigmoid()
            )
        )
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        out_conv = self.conv(x)
        out_conv = out_conv.view(self.batch_size, -1)
        out = self.fc(out_conv)
        return out