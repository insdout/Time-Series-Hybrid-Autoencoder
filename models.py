import torch
import torch.nn as nn
import numpy as np
from hydra.utils import instantiate


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, dropout_lstm, dropout=0, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = dropout_lstm
        self.p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * hidden_size,
                out_features=latent_dim)
        )

        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * hidden_size,
                out_features=latent_dim)
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

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
        """
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
                 dropout_lstm,
                 dropout_layer,
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


class RVE(nn.Module):

    def __init__(self, encoder, decoder=None, reconstruct=False, dropout_regressor=0, regression_dims=200):
        super(RVE, self).__init__()
        self.decode_mode = reconstruct
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder
        self.encoder = encoder
        self.p = dropout_regressor
        self.regression_dims = regression_dims
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat

        return y_hat, z, mean, log_var



class RVEAttention_MH(nn.Module):

    def __init__(self,  encoder, attention_embed_dim, attention_num_heads, attention_dropout, decoder=None,
                 reconstruct=False, dropout_regressor=0, regression_dims=200):
        super(RVEAttention_MH, self).__init__()
        self.decode_mode = reconstruct
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder
        self.encoder = encoder
        self.p = dropout_regressor
        self.regression_dims = regression_dims
        self.self_attention = nn.MultiheadAttention(embed_dim=attention_embed_dim, num_heads=attention_num_heads,
                                                    dropout=attention_dropout, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(attention_embed_dim)
  
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        """
        self attention input_size dims: N, L, E,
        where   N batch size
                L is the target sequence length,
                E is the query embedding dimension embed_dim
        x input_size: N, L, F,
        where   N batch size
                L is window size,
                E number of sensors
        For feature-wise attention x should be transposed: (N, E, L)
        """
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.self_attention(x, x, x)
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat

        return y_hat, z, mean, log_var


class MultiplicativeAttention(nn.Module):

    def __init__(self, values_embedding, queries_embedding):
        super().__init__()
        self.values_embedding = values_embedding
        self.queries_embedding = queries_embedding
        self.W = torch.nn.Parameter(torch.FloatTensor(
            self.queries_embedding, self.values_embedding).uniform_(-0.1, 0.1), requires_grad=True)

    def forward(self,
        query,  # [Batch size, N queries, queries_dim]
        values # [Batch size, N values, values_dim]
    ):

        weights = query @ self.W @ values.permute(0, 2, 1)  # [Batch size, N queries, N values]
        weights /= np.sqrt(self.queries_embedding)
        out = weights @ values # [Batch size, N queries, values_dim]
        return out  
    

class RVEAttention_MP(nn.Module):

    def __init__(self,  encoder, attention_values_embedding, attention_queries_embedding, decoder=None,
                 reconstruct=False, dropout_regressor=0, regression_dims=200):
        super(RVEAttention_MP, self).__init__()
        self.decode_mode = reconstruct
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder
        self.encoder = encoder
        self.p = dropout_regressor
        self.regression_dims = regression_dims
        self.self_attention = MultiplicativeAttention(values_embedding=attention_values_embedding, queries_embedding=attention_queries_embedding)
        self.batchnorm = nn.BatchNorm1d(attention_values_embedding, affine=False)
  
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        """
        self attention input_size dims: N, L, E,
        where   N batch size
                L is the target sequence length,
                E is the query embedding dimension embed_dim
        x input_size: N, L, F,
        where   N batch size
                L is window size,
                E number of sensors
        For feature-wise attention x should be transposed: (N, E, L)
        """
        x = torch.permute(x, (0, 2, 1))
        x = self.self_attention(x, x)
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat

        return y_hat, z, mean, log_var