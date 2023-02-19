import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc_mean = nn.Linear(
            in_features=self.num_directions*hidden_size,
            out_features=latent_dim
        )

        self.fc_log_var = nn.Linear(
            in_features=self.num_directions*hidden_size,
            out_features=latent_dim
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var*epsilon
        return z

    def forward(self, x):
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return z, mean, log_var


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 latent_dim,
                 window_size,
                 num_layers=1,
                 biderectional=True
                 ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.biderectional = biderectional
        self.window_size = window_size
        self.num_directions = 2 if self.biderectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_to_output = nn.LSTM(
            input_size=self.num_directions * hidden_size,
            hidden_size=input_size,
            batch_first=True
        )

    def forward(self, z):
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm_to_hidden(latent_z)
        out, _ = self.lstm_to_output(out)
        return out


class RVE(nn.Module):

    def __init__(self, encoder, decoder=None, reconstruct=False):
        super(RVE, self).__init__()
        self.decode_mode = reconstruct
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder
        self.encoder = encoder
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat

        return y_hat, z, mean, log_var
