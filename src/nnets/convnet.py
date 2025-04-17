import torch
import torch.nn as nn
import numpy as np

from src.nnets.utils import get_conv1d_output_dim


class Conv1DNet(nn.Module):
    def __init__(
        self,
        x_dim,
        out_dim=1,
        n_layers=3,
        nfilters=3,
        kernel_size=3,
        with_pooling=True,
        sigmoid=False,
        device="cuda",
    ):

        super().__init__()
        self.x_dim = x_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.with_pooling = with_pooling
        self.sigmoid_flag = sigmoid
        self.device = device

        # First layer of the network with a simple convolution
        self.comp_hlayers = nn.ModuleList()

        self.padding = 1
        self.conv_input = nn.Conv1d(
            1, nfilters, kernel_size=self.kernel_size, padding=self.padding
        )
        # self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(self.kernel_size)

        self.h_out_dims = []
        self.h_out_dims.append(
            get_conv1d_output_dim(
                x_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=1,
                dilation=1,
            )
        )

        for i in range(n_layers - 1):
            self.comp_hlayers.append(
                nn.Conv1d(
                    nfilters,
                    nfilters,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                )
            )
            h_dim = get_conv1d_output_dim(
                self.h_out_dims[-1],
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=1,
                dilation=1,
            )
            self.h_out_dims.append(h_dim)

        if with_pooling:
            h_dim = h_dim // (self.kernel_size)
            self.h_out_dims[-1] = h_dim
        self.out_layer = nn.Linear(nfilters * self.h_out_dims[-1], out_dim)
        self.act_fun = nn.LeakyReLU()
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(
            batch_size, 1, -1
        )  # (batch_size, 1, x_dim) for 1D conv, with 1 channel input

        out = self.conv_input(out)
        out = self.act_fun(out)

        for i in range(self.n_layers - 1):
            out = self.comp_hlayers[i](out)
            out = self.act_fun(out)

        if self.with_pooling:
            out = self.avg_pool(out)
        out = out.view(batch_size, -1)
        out = self.out_layer(out)
        if self.sigmoid_flag:
            out = torch.sigmoid(out)
        return out


if __name__ == "__main__":
    input_dim = (50,)
    out_dim = 1
    nrows = 70
    device = torch.device("cuda")
    model = Conv1DNet(
        x_dim=np.prod(input_dim),
        out_dim=out_dim,
        sigmoid=True,
    )
    print(model.h_out_dims)
    x = torch.randn(
        ((nrows,) + input_dim),
    ).to(
        device
    )  # Batch size of n_samples, 1 channel, 20x50 image
    # c = torch.randn((10, c_dim)).to(device)
    output = model(x)
    print(output.shape)  # Should print torch.Size([5, 1])
    assert output.shape == torch.Size([nrows, 1])
