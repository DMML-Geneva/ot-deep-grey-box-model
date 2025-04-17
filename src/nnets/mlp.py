import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from src.sampler.distributions import Bernoulli


def weights_init_mlp(m, activation="leaky_relu"):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        # check if the activation function is relu or leaky_relu
        if activation == "relu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="relu"
            )
        elif activation == "leaky_relu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
        elif activation == "elu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
        elif activation == "selu":
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="selu"
            )
        elif activation == "silu":
            nn.init.xavier_normal(m.weight)
        elif activation == "tanh":
            nn.init.xavier_normal_(m.weight)
        elif activation == "softplus":
            nn.init.xavier_normal_(m.weight)


def sample_lat_space(
    n_samples, z_dim=1, z_size=4, type="gauss", z_std=0.1, device="cuda"
):
    if type.lower() == "gauss":
        Z = torch.randn(n_samples, z_size, z_dim, device=device) * z_std
    elif type.lower() == "uniform":
        Z = torch.rand(n_samples, z_size, z_dim, device=device)
    elif type.lower() == "bernoulli":
        bernoulli = Bernoulli(
            p=0.5,
            head=torch.ones(z_dim, device=device),
            tail=torch.zeros(z_dim, device=device),
            device=device,
        )
        Z = bernoulli.sample(n_samples * z_size).reshape(
            n_samples, z_size, z_dim
        )
    else:
        raise ValueError(f"Unknown noise latent distribution {type}")
    return Z


class MLPConditionalGenerator(nn.Module):
    def __init__(
        self,
        x_dim,
        c_dim=0,
        c_embed_dim=None,
        z_dim=None,
        output_dim=None,
        h_layers=[
            100,
            100,
            100,
            100,
        ],  # 3 hidden layers with 100 neurons each as default
        z_std=0.1,
        activation="relu",
        with_drop_out=False,
        with_layer_norm=False,
        init_weights=weights_init_mlp,
        device="cuda",
    ):
        super(MLPConditionalGenerator, self).__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.c_embed_dim = c_embed_dim
        self.params_dim = c_dim
        self.h_layers = h_layers
        if z_dim is None:
            self.z_dim = x_dim
        else:
            self.z_dim = z_dim
        self.z_std = z_std
        self.output_dim = output_dim
        if self.output_dim is None:
            self.output_dim = x_dim

        self.activation_str = activation
        if self.activation_str == "relu":
            self.activation = F.relu
        elif self.activation_str == "leaky_relu":
            self.activation = F.leaky_relu
        elif self.activation_str == "elu":
            self.activation = F.elu
        elif self.activation_str == "selu":
            self.activation = F.selu
        elif self.activation_str == "tanh":
            self.activation = F.tanh
        elif self.activation_str == "softplus":
            self.activation = F.softplus
        elif self.activation_str == "silu":
            self.activation = F.silu

        self.init_weights = init_weights
        self.with_layer_norm = with_layer_norm
        self.with_drop_out = with_drop_out
        if self.with_drop_out:
            self.dropout_layers = [
                nn.Dropout(0.2).to(device) for _ in range(len(h_layers))
            ]
        else:
            self.dropout_layers = None
        if self.with_layer_norm:
            self.batch_norm_layers = [
                nn.LayerNorm(h_layers[i]).to(device)
                for i in range(len(h_layers))
            ]

        self.device = device

        x_blocks = [nn.Linear(x_dim + z_dim, h_layers[0])]

        if c_dim > 0:
            context_blocks = [nn.Linear(c_dim, h_layers[0])]

        # Add the hidden layers
        for i in range(1, len(h_layers)):
            x_blocks.append(nn.Linear(h_layers[i - 1], h_layers[i]))
            if c_dim > 0:
                context_blocks.append(nn.Linear(c_dim, h_layers[i]))

        if output_dim is not None:
            x_blocks.append(nn.Linear(h_layers[-1], output_dim))
        else:
            x_blocks.append(nn.Linear(h_layers[-1], x_dim))
        self.x_blocks = nn.ModuleList(x_blocks)
        self.context_blocks = nn.ModuleList(context_blocks)

        self.to(device)

        if self.init_weights is not None:
            init_weight = lambda m: self.init_weights(
                m, activation=self.activation_str
            )
            self.apply(init_weight)

    def forward(self, x, context=None):
        # Concatenate the input with the noise
        batch_size = x.shape[0]
        out = x.reshape(batch_size, -1)

        # for loop until the last layer
        for i, block in enumerate(self.x_blocks[:-1]):
            out = block(out)

            if self.with_layer_norm:
                out = self.batch_norm_layers[i](out)

            out = self.activation(out)

            if self.with_drop_out:
                out = self.dropout_layers[i](out)

            # Conditional block
            if context is not None and i < len(self.context_blocks):
                c = context.view(batch_size, -1)
                c = self.context_blocks[i](c)
                out = out + self.activation(c)

        out = self.x_blocks[-1](out)  # Last layer without activation
        return out

    def predict(self, X, context=None, z_samples=10, z_type_dist="gauss"):
        x_dim = X.shape[-1]
        params_dim = np.prod(context.shape[1:])

        # Repeat for noise samples
        sims = X.reshape(-1, 1, x_dim).repeat(1, z_samples, 1)
        params = context.reshape(-1, 1, params_dim).repeat(1, z_samples, 1)

        if self.z_dim < 1:
            XZ = sims
        else:
            Z = sample_lat_space(
                X.shape[0],
                z_dim=self.z_dim,
                z_size=z_samples,
                type=z_type_dist,
                z_std=self.z_std,
                device=X.device,
            )
            XZ = torch.cat([sims, Z], dim=2)

        # Prediction
        XZ = XZ.flatten(start_dim=0, end_dim=1)
        params = params.flatten(start_dim=0, end_dim=1)
        pred = self(XZ, context=params)

        return pred.reshape(-1, z_samples, self.output_dim)


def get_mlp_discriminator(
    x_dim, h_layers=[100, 100, 100], sigmoid=False, device="cuda"
):
    # Initial Linear layer
    networks = [
        nn.Linear(x_dim, h_layers[0]),
        nn.LeakyReLU(),
    ]  # nn.Dropout(0.2)

    # Hidden layers
    for i in range(1, len(h_layers)):
        networks.append(nn.Linear(h_layers[i - 1], h_layers[i]))
        networks.append(nn.LeakyReLU())  #
        # networks.append(nn.Dropout(0.2))
    # Last layer
    networks.append(nn.Linear(h_layers[-1], 1))

    if sigmoid:
        networks.append(nn.Sigmoid())
    activation_str = "sigmoid" if not sigmoid else "leaky_relu"
    f = nn.Sequential(*networks).to(device)
    init_weight = lambda m: weights_init_mlp(m, activation=activation_str)
    f.apply(init_weight)
    # for p in f.parameters():
    #    factor = x_dim  #
    #    p.data = p.data / factor
    return f
