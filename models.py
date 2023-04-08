import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, dropout=0, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
            in_features=self.num_directions*hidden_size,
            out_features=latent_dim)
        )


        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
            in_features=self.num_directions*hidden_size,
            out_features=latent_dim)
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
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        else:
            h = h_n[-1, -1, :, :]
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
                 bidirectional=True
                 ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.window_size = window_size
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
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

    def __init__(self, encoder, decoder=None, reconstruct=False, dropout=0):
        super(RVE, self).__init__()
        self.decode_mode = reconstruct
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder
        self.encoder = encoder
        self.p = dropout
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 200),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat

        return y_hat, z, mean, log_var


class SimpleRVE(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=0, num_layers=1):
        super(SimpleRVE, self).__init__()
        self.decode_mode = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.p = dropout
        self.num_layers = num_layers
        if self.bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=self.bidirectional)
        self.mu = nn.Linear(in_features=self.D * hidden_size, out_features=2)
        self.sigma = nn.Linear(in_features=self.D * hidden_size, out_features=2)

        self.regressor = nn.Sequential(nn.Linear(2, 200), nn.Tanh(), nn.Dropout(self.p), nn.Linear(200, 1))

    def forward(self, x):
        batch = x.shape[0]
        _, (hn, _) = self.lstm(x)
        """
        hidden.shape = (num_layers*num_directions, batch, hidden_size)
        layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        So you shouldn’t simply do hidden[-1] but first do a view() to separate the num_layers and num_directions (1 or 2). If you do

        hidden = hidden.view(num_layers, 2, batch, hidden_size) # 2 for bidirectional
        last_hidden = hidden[-1]
        then last_hidden.shape = (2, batch, hidden_size) and you can do

        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]
        """

        hn = hn.view(self.num_layers, self.D, batch, self.hidden_size)

        last_hidden = hn[-1]
        if self.bidirectional:
            out = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        else:
            out = last_hidden[0]

        mu = self.mu(out)
        sigma = self.sigma(out)

        eps = torch.randn_like(sigma)
        z = mu + eps * torch.exp(0.5 * sigma)

        y_hat = self.regressor(z)

        return y_hat, z, mu, sigma
