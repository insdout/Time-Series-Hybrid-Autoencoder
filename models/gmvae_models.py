import torch
from torch import nn
import logging

log = logging.getLogger(__name__)


class Qy_x(nn.Module):
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.encoder = encoder
        self.qy_logit = nn.Linear(enc_out_dim, k)
        self.qy = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = self.encoder(x)
        qy_logit = self.qy_logit(h1)
        qy = self.qy(qy_logit)
        y = torch.nn.functional.gumbel_softmax(qy_logit, hard=True)
        return qy_logit, qy, y


class Qz_xy(nn.Module):
    def __init__(self, k, encoder, enc_out_dim, hidden_size, latent_dim):
        super(Qz_xy, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.h2 = nn.Sequential(
            nn.Linear(enc_out_dim + k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
            )
        self.z_mean = nn.Linear(hidden_size, latent_dim)
        self.zlogvar = nn.Linear(hidden_size, latent_dim)

    def gaussian_sample(self, z_mean, z_logvar):
        z_std = torch.sqrt(torch.exp(z_logvar))

        eps = torch.randn_like(z_std)
        z = z_mean + eps*z_std

        return z

    def forward(self, x, y):
        h1 = self.encoder(x)    # dim: Batch, hidden_size
        xy = torch.cat((h1, y), dim=1)
        h2 = self.h2(xy)
        # q(z|x, y)
        z_mean = self.z_mean(h2)
        zlogvar = self.zlogvar(h2)
        z = self.gaussian_sample(z_mean, zlogvar)
        return z, z_mean, zlogvar


class Px_z(nn.Module):
    def __init__(self, decoder, k):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.decoder_hidden = self.decoder.hidden_size
        self.latent_dim = self.decoder.latent_dim
        self.z_mean = nn.Linear(k, self.latent_dim)
        self.zlogvar = nn.Linear(k, self.latent_dim)

    def forward(self, z, y):
        # p(z|y)
        z_mean = self.z_mean(y)
        zlogvar = self.zlogvar(y)

        # p(x|z)
        x_hat = self.decoder(z)
        return z_mean, zlogvar, x_hat


class EncoderLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout_lstm,
                 dropout_layer=0.0,
                 num_layers=1,
                 bidirectional=False
                 ):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = dropout_lstm
        self.p_dropout_layer = dropout_layer

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer)

    def forward(self, x):
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        """
        hidden.shape = (num_layers*num_directions, batch, hidden_size)
        layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        So you shouldn’t simply do hidden[-1] but first do a view() to separate the num_layers and num_directions (1 or 2). If you do

        hidden = hidden.view(num_layers, 2, batch, hidden_size) # 2 for bidirectional
        last_hidden = hidden[-1]
        then last_hidden.shape = (2, batch, hidden_size) and you can do

        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]

        TODO: check if it same as
        # Pass the input through the LSTM
        output, (h_n, c_n) = lstm(input_data, (h0, c0))
        Extract the last forward and backward outputs
        last_forward_output = output[:, -1, :hidden_size]
        last_backward_output = output[:, 0, hidden_size:]

        """
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        else:
            h = h_n[-1, -1, :, :]
        h = self.dropout_layer(h)
        return h


class DecoderLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 latent_dim,
                 window_size,
                 dropout_lstm,
                 dropout_layer,
                 num_layers=1,
                 bidirectional=False
                 ):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.window_size = window_size
        self.p_lstm = dropout_lstm
        self.p_dropout_layer = dropout_layer
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer)

        self.lstm_to_output = nn.LSTM(
            input_size=self.num_directions * hidden_size,
            hidden_size=input_size,
            batch_first=True
        )

    def forward(self, z):
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm_to_hidden(latent_z)
        out = self.dropout_layer(out)
        out, _ = self.lstm_to_output(out)
        return out


class GMVAE(nn.Module):

    def __init__(self, k, Qy_x_net, Qz_xy_net, Px_z_net, regression_dims, regression_dropout=0.0):
        """_summary_

        Args:
            k (_type_): _description_
            Qy_x_net (_type_): _description_
            Qz_xy_net (_type_): _description_
            Px_z_net (_type_): _description_
        """
        super(GMVAE, self).__init__()
        self.k = k
        self.qy_x = Qy_x_net
        self.qz_xy = Qz_xy_net
        self.px_z = Px_z_net
        self.p = regression_dropout
        self.regression_dims = regression_dims
        self.regressor = nn.Sequential(
            nn.Linear(self.qz_xy.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def infer(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        k = self.k
        batch_size = x.shape[0]
        qy_logit, qy = self.qy_x(x)
        y_hat = torch.argmax(qy, dim=-1)

        # Create tensor with 1s at specified indices
        y_ = torch.zeros(batch_size, k).to(y_hat.device)
        y_ = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
        z_hat, *_ = self.qz_xy(x, y_)
        *_, x_hat = self.px_z(z_hat, y_)
        rul_hat = self.regressor(z_hat)
        out_infer = {
            "latent_y": y_hat,
            "z": z_hat,
            "x_hat": x_hat,
            "rul_hat": rul_hat
            }
        return out_infer

    def forward(self, x):
        k = self.k
        batch_size = x.shape[0]
        qy_logit, qy, y = self.qy_x(x)
        z, zm, zv = self.qz_xy(x, y)
        zm_prior, zv_prior, px = self.px_z(z, y)
        rul_hat = self.regressor(z)


        out_dict = {
            "z_latent": z,
            "zm": zm,
            "zv": zv,
            "zm_prior": zm_prior,
            "zv_prior": zv_prior,
            "qy_logit": qy_logit,
            "qy": qy,
            "px": px,
            "rul_hat": rul_hat,
            "x_hat": px,
            "y": y
            }

        return out_dict
