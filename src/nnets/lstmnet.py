import torch
import numpy as np

from src.sampler.distributions import sample_latent_noise


# Define the LSTM RNN one-to-many model RNN.
class LSTMNet(torch.nn.Module):

    def __init__(
        self,
        x_shape,
        length_seq=50,
        h_dim=-1,
        out_shape=50,
        c_dim=0,
        z_dim=0,
        c_embed_dim=5,
        lstm_layers=4,
        device="cuda",
    ):
        """
        x_shape: tuple.
            Shape of the input data (excluding batch size).
        length_seq: int.
            Length of the sequence.
        out_shape: int.
            Shape of the output data (excluding batch size and time/sequence dim).
        c_dim: int.
            Dimension of the context flattened.
        c_embed_dim: int.
            Dimension of the context embedding.
        lstm_layers: int.
            Number of layers of the stacked LSTM.
        device: str.
        """
        super(LSTMNet, self).__init__()
        self.x_shape = x_shape
        self.x_dim = np.prod(x_shape)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.c_embed_dim = c_embed_dim if c_dim > 0 else 0
        self.x_in_size = int(self.x_dim + self.z_dim + c_embed_dim)
        self.h_dim = h_dim if h_dim > 0 else self.out_dim

        self.out_shape = out_shape
        self.out_dim = int(np.prod(out_shape))
        self.length_seq = length_seq

        self.params_dim = c_dim
        self.h_layers = lstm_layers

        self.lstm = torch.nn.LSTM(
            input_size=self.x_in_size,
            hidden_size=self.h_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.c_layer = None
        if c_dim > 0:
            self.c_layer = torch.nn.Linear(c_dim, self.c_embed_dim)

        self.act_fun = torch.nn.LeakyReLU()
        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(
                self.h_dim * self.length_seq, self.out_dim * self.length_seq
            ),
            self.act_fun,
            torch.nn.Linear(
                self.out_dim * self.length_seq, self.out_dim * self.length_seq
            ),
        )
        # self.final_layer = torch.nn.Linear(self.out_dim * self.length_seq, self.out_dim*self.length_seq)
        self.device = device
        self.to(device)

    def forward(self, x, context=None):
        batch_size = x.size(0)
        # If x has time/sequence dimension, then x.shape=(N,D,L)
        # One-to-many to many-to-many
        if (x.shape[-1] - self.z_dim) != self.length_seq:
            # repeat last dimension to match out_dim
            out = (
                x.unsqueeze(1)
                .repeat(1, self.length_seq, 1)
                .flatten(start_dim=2)
            )  # x.shape = (N,L,D)
        else:
            if x.ndim < 3:
                # x.shape = (N, L) -> (N, 1, L)
                out = x.unsqueeze(1)
            else:
                out = x.flatten(start_dim=1, end_dim=-2)  # x.shape = (N,D,L)
            if self.z_dim > 0:
                # x.shape = (N,D,[L,Z]))
                z = out[:, 0, -self.z_dim :]  # z.shape = (N,Z)
                out = out[:, :, : -self.z_dim]  # out.shape = (N,D,L)
                z = z.unsqueeze(2).repeat(1, 1, self.length_seq)
                out = torch.cat([out, z], dim=1)  # out.shape = (N,D+Z,L)

            out = out.permute(0, 2, 1)  # x.shape = (N,L,D)

        # out here must: out.shape = (N,L,D)

        # Concatenate the input with the context embedding.
        if self.c_layer is not None and context is not None:
            c = self.c_layer(
                context.view(batch_size, -1)
            )  # (n, c_dim) -> (n, x_input)
            c = self.act_fun(c)
            # repeat context to match the sequence/temporal length
            c = c.unsqueeze(1).repeat(1, self.length_seq, 1)
            # Concatenate the context with the input
            out = torch.cat([out, c], dim=-1)

        # x here must: x.shape = (N,L,D)
        out, (h_l, c_l) = self.lstm(out)  # out.shape = (N,L,H_dim)
        out = out.flatten(start_dim=1)
        out = self.final_layers(out)
        out = out.view(batch_size, *self.out_shape, self.length_seq)
        if self.out_dim == 1:
            out = out.squeeze(1)
        return out

    def predict(
        self,
        X,
        context=None,
        z_samples=10,
        z_type_dist="gauss",
        z_std=0.1,
    ):
        if self.z_dim < 1:
            return self(X, context=context)

        batch_size = X.shape[0]
        # Repeat for noise samples
        Z = sample_latent_noise(
            n_samples=X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            z_std=z_std,
            device=X.device,
        )
        if X.shape[-1] != self.length_seq:
            sims = X.reshape((batch_size, 1) + self.x_shape)
        else:
            sims = X.reshape(
                (batch_size, 1) + self.x_shape + (self.length_seq,)
            )
        sims = sims.repeat(1, z_samples, *([1] * len(sims.shape[2:])))
        params = (
            context.reshape(batch_size, 1, -1)
            .repeat(1, z_samples, 1)
            .flatten(start_dim=0, end_dim=1)
        )

        # Z.shape = (n_samples, z_samples, z_dim)
        if X.shape[-1] == self.length_seq:
            Z = Z.reshape(
                (
                    batch_size,
                    z_samples,
                    *([1] * len(sims.shape[2:-1])),
                    self.z_dim,
                )
            )
            Z = Z.repeat(1, 1, *sims.shape[2:-1], 1)
        XZ = torch.cat([sims, Z], dim=-1)
        XZ = XZ.flatten(start_dim=0, end_dim=1)
        if X.shape[-1] == self.length_seq:
            XZ = XZ.view(
                batch_size * z_samples, -1, self.length_seq + self.z_dim
            )
        pred = self(XZ, params)
        return pred


if __name__ == "__main__":
    nrows = 40
    x_shape = (1,)
    t_dim = 50
    z_dim = 0
    c_dim = 2

    device = torch.device("cuda")
    model = LSTMNet(
        x_shape=x_shape,
        length_seq=t_dim,
        out_shape=x_shape,
        c_dim=c_dim,
        z_dim=z_dim,
        device=device,
    )
    if z_dim > 0:
        x_shape = x_shape[:-1] + (x_shape[-1] + z_dim,)
    x = torch.randn(nrows, *x_shape).to(
        device
    )  # Batch size of n_samples, 1 channel, 20x50 image
    c = torch.randn((nrows, c_dim)).to(device)
    print(x.shape)
    output = model(x, context=c)
    print(output.shape)  # Should print torch.Size([5, 1])
    if z_dim > 0:
        pred = model.predict(x[:, :-z_dim], context=c)
    else:
        pred = model.predict(x, context=c)
    print(pred.shape)
