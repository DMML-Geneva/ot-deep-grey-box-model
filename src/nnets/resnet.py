import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from src.nnets.mlp import sample_lat_space


class ResNet_D(nn.Module):
    "Discriminator ResNet architecture from https://github.com/harryliew/WGAN-QC"

    def __init__(
        self, size=64, nc=3, nfilter=64, nfilters_max=512, res_ratio=0.1
    ):
        """
        Args:
            size (int): image size
            nc (int): number of channels
            nfilter (int): number of filters
            nfilters_max (int): maximum number of filters
            res_ratio (float): residual ratio
        """
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilters_max
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(nc, 1 * nf, 3, padding=1)
        self.act_fun = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.act_fun((self.conv_img(x)))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)

        return out


class ResNet_G(nn.Module):
    "Generator ResNet architecture from https://github.com/harryliew/WGAN-QC"

    def __init__(
        self,
        z_dim,
        size,
        nc=3,
        nfilter=64,
        nfilters_max=512,
        bn=True,
        res_ratio=0.1,
        **kwargs,
    ):
        """
        Args:
            z_dim (int): latent dimension
            size (int): image size
            nc (int): number of channels
            nfilter (int): number of filters
            nfilters_max (int): maximum number of filters
            bn (bool): batch normalization
            res_ratio (float): residual ratio
        """
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilters_max
        self.bn = bn
        self.z_dim = z_dim
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** (nlayers + 1))

        self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.act_fun = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2**i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2),
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, nc, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        out = self.fc(z)
        if self.bn:
            out = self.bn1d(out)
        out = self.act_fun(out)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class ResNetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(
            self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias
        )
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias
        )
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.act_fun = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.act_fun(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.act_fun(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s


class Conv1DResidualBlock(nn.Module):
    def __init__(
        self,
        fin,  #   input channels
        fout,  #   output channels
        fhidden=None,  #   hidden channels
        bn=True,
        res_ratio=0.1,
    ):
        super().__init__()

        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout

        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv1d(
            self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias
        )

        if self.bn:
            self.bn1d_0 = nn.BatchNorm1d(self.fhidden)

        self.conv_1 = nn.Conv1d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias
        )
        if self.bn:
            self.bn1d_1 = nn.BatchNorm1d(self.fout)

        if self.learned_shortcut:
            self.conv_s = nn.Conv1d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )
            if self.bn:
                self.bn1d_s = nn.BatchNorm1d(self.fout)
        self.act_fun = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn1d_0(dx)
        dx = self.act_fun(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn1d_1(dx)
        out = self.act_fun(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn1d_s(x_s)
        else:
            x_s = x
        return x_s


"""
    1D Convolutional Residual Network for Discriminator
"""


class Conv1DResidualNet(nn.Module):
    def __init__(
        self,
        x_dim,
        out_dim=1,
        context_dim=0,
        z_dim=0,
        nlayers=4,
        nfilters=10,
        nfilters_max=128,
        kernel_size=3,
        res_ratio=0.1,
        with_avg_pool=True,
        sigmoid=False,  # for GANs discriminator
        device="cuda",
    ):
        super().__init__()
        self.x_dim = x_dim
        self.out_dim = out_dim
        self.context_dim = context_dim
        self.nlayers = nlayers
        self.nfilters = nfilters
        self.nfilters_max = nfilters_max
        self.kernel_size = kernel_size
        self.with_avg_pool = with_avg_pool
        self.z_dim = z_dim
        self.sigmoid_flag = sigmoid
        self.device = device

        # First layer of the network with a simple convolution (no residual)
        self.conv_input = nn.Conv1d(
            1, 1 * nfilters, kernel_size=self.kernel_size, padding=1
        )
        self.act_fun = nn.LeakyReLU(0.2, inplace=True)
        self.activation_str = "leaky_relu"

        nf0 = min(nfilters, nfilters_max)
        nf1 = min(nfilters * 2, nfilters_max)

        if context_dim > 0:
            self.fc_context_input = nn.Linear(context_dim, nfilters)

        # First blocks of residual, no downsampling with avgpool
        blocks = [
            Conv1DResidualBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            Conv1DResidualBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nfilters * 2**i, nfilters_max)
            nf1 = min(nfilters * 2 ** (i + 1), nfilters_max)
            if self.with_avg_pool:
                blocks += [
                    nn.AvgPool1d(
                        kernel_size=self.kernel_size, stride=2, padding=1
                    ),
                ]
            blocks += [
                Conv1DResidualBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
                Conv1DResidualBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]
            self.last_block_out_dim = nf1

        self.resnet = nn.Sequential(*blocks)

        # Flatten the output of the last block
        # \frac{x_dim}{out} = 2^{nlayers}
        if self.x_dim % 10 == 0:
            self.resnet_out_flat_dim = (int)(
                np.ceil((self.x_dim / (2**self.nlayers)))
            )
        else:
            self.resnet_out_flat_dim = (int)(
                np.ceil((self.x_dim / (2**self.nlayers - 1)))
            )

        self.resnet_out_flat_dim = (
            self.last_block_out_dim * self.resnet_out_flat_dim
        )
        self.fc = nn.Linear(self.resnet_out_flat_dim, self.out_dim)
        if self.context_dim > 0:
            self.fc_context_out = nn.Linear(
                context_dim, self.resnet_out_flat_dim
            )

        # to device
        self.to(device)

    def forward(
        self,
        inputs,
        context=None,
    ):
        batch_size = inputs.size(0)

        # First layer of the network with a simple convolution (no residual)
        out = inputs.reshape(
            batch_size, 1, -1
        )  # (batch_size, 1, x_dim) for 1D conv, with 1 channel input
        out = self.act_fun(
            (self.conv_input(out))
        )  # (batch_size, nfilters, x_dim)

        if self.context_dim > 0:
            out_context = self.fc_context_input(context.view(batch_size, -1))
            out_context = self.act_fun(out_context)
            # match the dimensions of the context with the output of the first layer
            out_context = out_context.unsqueeze(2).expand(-1, -1, out.shape[2])
            out = out + out_context

        out = self.resnet(out)

        out = out.view((batch_size, -1))

        if self.context_dim > 0:
            out_context = self.fc_context_out(context.view(batch_size, -1))
            out_context = self.act_fun(out_context)
            out = out + out_context

        out = self.fc(out)
        if self.sigmoid_flag:
            out = torch.sigmoid(out)

        return out

    def predict(self, X, context, z_samples, z_type_dist):
        x_dim = np.prod(X.shape[1:])
        params_dim = np.prod(context.shape[1:])
        if self.z_dim < 1:
            return self(X, context=context)

        # Repeat for noise samples
        sims = X.reshape(-1, 1, x_dim).repeat(1, z_samples, 1)
        params = context.reshape(-1, 1, params_dim).repeat(1, z_samples, 1)
        Z = sample_lat_space(
            X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            device=X.device,
        )
        XZ = torch.cat([sims, Z], dim=-1)

        # Prediction
        XZ = XZ.flatten(start_dim=0, end_dim=1)
        params = params.flatten(start_dim=0, end_dim=1)
        pred = self(XZ, context=params)

        return pred.reshape(-1, z_samples, self.out_dim)


# Test the Simple ResNet with input shape (batch_size, 1, 20, 50)
if __name__ == "__main__":
    input_dim = (1, 50)
    z_dim = 4
    kernel_size = 3
    c_dim = 1
    out_dim = 50
    model = Conv1DResidualNet(
        x_dim=np.prod(input_dim) + z_dim,
        out_dim=out_dim,
        context_dim=c_dim,
        nfilters=8,
        nlayers=4,
    )
    device = torch.device("cpu")
    model.to(device)
    x = torch.randn(
        (55,) + input_dim,
    ).to(
        device
    )  # Batch size of n_samples, 1 channel, 20x50 image
    z = torch.randn((55, input_dim[0], z_dim)).to(device)
    c = torch.randn((55, c_dim)).to(device)
    if z_dim > 0:
        output = model(torch.concat((x, z), dim=-1), context=c)
    else:
        output = model(x, context=c)
    print(output.shape)  # Should print torch.Size([5, 1])
    assert output.shape == torch.Size([55, 50])
