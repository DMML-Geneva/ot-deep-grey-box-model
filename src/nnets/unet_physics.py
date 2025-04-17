# This file contains the implementation of the UNet model for the reaction-diffusion system
# Adapted from: Wehenkel, Apple ....

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.sampler.distributions import sample_latent_noise


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Block2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Block3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
            x = block(x)
            # print(x.shape)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Encoder2d(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block2d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
            x = block(x)
            # print(x.shape)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Encoder3d(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block3d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
            x = block(x)
            # print(x.shape)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), cond_size=0):
        super().__init__()
        self.chs = chs
        cond_sizes = [0] * len(chs)
        cond_sizes[0] = cond_size
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    chs[i] + cond_sizes[i], chs[i + 1], 2, stride=2
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, 1])(
            enc_ftrs.unsqueeze(3)
        ).squeeze(3)
        return enc_ftrs


class Decoder2d(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), cond_size=0):
        super().__init__()
        self.chs = chs
        cond_sizes = [0] * len(chs)
        cond_sizes[0] = cond_size
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    chs[i] + cond_sizes[i], chs[i + 1], 2, stride=2
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block2d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        # x is the smallest feature map
        # encoder_features are the larger feature maps
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)  # Align dimensions
            x = torch.cat([x, enc_ftrs], dim=1)  # Skip connection.
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(
            enc_ftrs
        )  # .squeeze(3)
        return enc_ftrs


class Decoder3d(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), cond_size=0):
        super().__init__()
        self.chs = chs
        cond_sizes = [0] * len(chs)
        cond_sizes[0] = cond_size
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    chs[i] + cond_sizes[i], chs[i + 1], 2, stride=2
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block3d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        # x is the smallest feature map
        # encoder_features are the larger feature maps
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            x = torch.cat([x, enc_ftrs], dim=1)  # Skip connection.
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_class=1,
        retain_dim=True,
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv1d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid()
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class UNet2d(nn.Module):
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_class=1,
        retain_dim=True,
    ):
        super().__init__()
        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.final_act = nn.Sigmoid()
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        enc_chs=(2, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_class=2,
        retain_dim=True,
        cond_dim=0,
        final_act=None,
    ):
        super().__init__()
        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs, cond_size=cond_dim)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.final_act = (
            nn.Sigmoid() if final_act is not None else nn.Identity()
        )
        self.retain_dim = retain_dim

    def forward(self, x, context=None):
        enc_ftrs = self.encoder(x)
        if context is not None:
            context = context.unsqueeze(2).expand(
                -1, -1, enc_ftrs[-1].shape[2]
            )
            enc_ftrs[-1] = torch.cat((context, enc_ftrs[-1]), 1)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])
        return out


class UNetReactionDiffusion(nn.Module):
    def __init__(
        self,
        enc_chs=(2, 16, 32, 64, 128),
        dec_chs=(128, 64, 32, 16),
        num_class=2,
        cond_dim=0,
        z_dim=0,
        t_dim=0,
        retain_dim=True,
        final_act=None,
        device="cuda",
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.t_dim = t_dim
        self.state_dim = num_class
        if t_dim > 0 and z_dim > 0:
            enc_chs = (enc_chs[0] * (t_dim + z_dim),) + enc_chs[1:]
        elif t_dim > 0:
            enc_chs = (enc_chs[0] * t_dim,) + enc_chs[1:]
        elif z_dim > 0:
            enc_chs = (enc_chs[0] * z_dim,) + enc_chs[1:]

        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs, cond_size=cond_dim)
        self.head = nn.Conv2d(dec_chs[-1], self.state_dim, 1)
        self.final_act = (
            nn.Sigmoid() if final_act is not None else nn.Identity()
        )
        self.retain_dim = retain_dim
        self.activation_str = "relu"
        self.device = device
        self.to(device)

    def forward(self, x, context=None):
        # Reshape from (batch, 2, grid, grid, (time + z_dim)) to
        #              (batch, 2*(time + z_dim), grid, grid)
        if self.t_dim > 0:
            b, c, h, w, t = x.shape
            x_input = x.permute(0, 4, 1, 2, 3)
            x_input = x_input.reshape(b, c * t, h, w)
            if context is not None:
                context = (
                    context.unsqueeze(1)
                    .repeat(1, t, 1)
                    .view(-1, self.cond_dim)
                )
        else:
            x_input = x

        # X_input shape (batch, 2 *time*z_dim, grid, grid)

        # Standard U-Net processing
        enc_ftrs = self.encoder(x_input)
        # Add context to the deepest feature map
        if context is not None:
            context = (
                context.unsqueeze(2)
                .unsqueeze(2)
                .expand(-1, -1, enc_ftrs[-1].shape[2], enc_ftrs[-1].shape[3])
            )
            # Concatenate context to the deepest feature map
            enc_ftrs[-1] = torch.cat((context, enc_ftrs[-1]), 1)

        # Decode with U-Net decoder, reverse the encoder features
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)

        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])

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

        # Repeat for noise samples
        Z = sample_latent_noise(
            n_samples=X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            z_std=z_std,
            device=X.device,
        )

        sims = X.unsqueeze(1).repeat(
            1, z_samples, 1, 1, 1, 1
        )  # (n_samples x z_size x 2 x grid x grid x time)

        grid_size = X.shape[2]
        # Z shape from (n_samples, z_size, z_dim) to (n_samples, z_size, state, grid, grid, z_dim)
        Z = (
            Z.unsqueeze(2)
            .unsqueeze(2)
            .unsqueeze(2)
            .repeat(1, 1, self.state_dim, grid_size, grid_size, 1)
        )
        # [(n_samples x z_size x 2 x grid x grid x time), (n_samples x z_size x 2 x grid x grid x z_dim)]
        XZ = torch.cat([sims, Z], dim=-1)
        XZ = XZ.flatten(
            start_dim=0, end_dim=1
        )  # (n_samples * z_size) x 2 x grid x grid x (time + z_dim)
        params = (
            context.unsqueeze(1)
            .repeat(1, z_samples, 1)
            .flatten(start_dim=0, end_dim=1)
        )  # (n_samples + z_size) x c_dim
        pred = self(XZ, params)
        return pred


class UNetReaction3DDiffusion(nn.Module):
    def __init__(
        self,
        enc_chs=(2, 16, 32, 64, 128),
        dec_chs=(128, 64, 32, 16),
        num_class=2,
        cond_dim=0,
        z_dim=0,
        final_act=None,
        device="cuda",
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.state_dim = num_class
        if z_dim > 0:
            enc_chs = (enc_chs[0] + z_dim,) + enc_chs[1:]

        self.encoder = Encoder3d(enc_chs)
        self.decoder = Decoder3d(dec_chs, cond_size=cond_dim)
        self.head = nn.Conv3d(dec_chs[-1], self.state_dim, 1)
        self.final_act = (
            nn.Sigmoid() if final_act is not None else nn.Identity()
        )
        self.activation_str = "relu"
        self.device = device
        self.to(device)

    def forward(self, x, context=None):

        # x shape (batch, 2, grid, grid, time+z_dim)
        if self.z_dim > 0:
            z = x[:, 0, :, :, -self.z_dim :].clone()
            x_input = x[:, :, :, :, : -self.z_dim]
            z = (
                z.permute(0, 3, 1, 2)
                .unsqueeze(-1)
                .repeat(1, 1, 1, 1, x_input.shape[-1])
            )
            x_input = torch.cat((x_input, z), dim=1)
        else:
            x_input = x

        # Standard U-Net processing
        enc_ftrs = self.encoder(x_input)
        # Add context to the deepest feature map
        if context is not None:
            context = (
                context.unsqueeze(2)
                .unsqueeze(2)
                .unsqueeze(2)
                .expand(
                    -1,
                    -1,
                    enc_ftrs[-1].shape[2],
                    enc_ftrs[-1].shape[3],
                    enc_ftrs[-1].shape[4],
                )
            )
            # Concatenate context to the deepest feature map
            enc_ftrs[-1] = torch.cat((context, enc_ftrs[-1]), 1)

        # Decode with U-Net decoder, reverse the encoder features
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = self.final_act(out)
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

        # Repeat for noise samples
        Z = sample_latent_noise(
            n_samples=X.shape[0],
            z_dim=self.z_dim,
            z_size=z_samples,
            type=z_type_dist,
            z_std=z_std,
            device=X.device,
        )

        sims = X.unsqueeze(1).repeat(
            1, z_samples, 1, 1, 1, 1
        )  # (n_samples x z_size x 2 x grid x grid x time)

        grid_size = X.shape[2]
        # Z shape from (n_samples, z_size, z_dim) to (n_samples, z_size, state, grid, grid, z_dim)
        Z = (
            Z.unsqueeze(2)
            .unsqueeze(2)
            .unsqueeze(2)
            .repeat(1, 1, self.state_dim, grid_size, grid_size, 1)
        )
        # [(n_samples x z_size x 2 x grid x grid x time), (n_samples x z_size x 2 x grid x grid x z_dim)]
        XZ = torch.cat([sims, Z], dim=-1)
        XZ = XZ.flatten(
            start_dim=0, end_dim=1
        )  # (n_samples * z_size) x 2 x grid x grid x (time + z_dim)
        params = (
            context.unsqueeze(1)
            .repeat(1, z_samples, 1)
            .flatten(start_dim=0, end_dim=1)
        )  # (n_samples + z_size) x c_dim
        pred = self(XZ, params)
        return pred


from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # Input shape: (batch, embed_dim, time)
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        x = x.permute(2, 0, 1)  # (time, batch, embed_dim)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (batch, embed_dim, time)
        return x


class UNetReactionAttentionDiffusion(nn.Module):
    def __init__(
        self,
        enc_chs=(2, 16, 32, 64, 128),
        dec_chs=(128, 64, 32, 16),
        num_class=2,
        cond_dim=0,
        z_dim=0,
        t_dim=0,
        retain_dim=True,
        final_act=None,
        device="cuda",
        embed_dim=2 * 32 * 32,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.t_dim = t_dim
        self.state_dim = num_class

        if z_dim > 0:
            enc_chs = (enc_chs[0] + z_dim,) + enc_chs[1:]

        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs, cond_size=cond_dim)
        self.head_dec = nn.Conv2d(dec_chs[-1], self.state_dim, 1)
        self.temporal_attention = TemporalAttention(
            embed_dim, num_heads, num_layers
        )

        self.final_act = (
            nn.Sigmoid() if final_act is not None else nn.Identity()
        )
        self.retain_dim = retain_dim
        self.device = device
        self.activation_str = "relu"
        self.to(device)

    def forward(self, x, context=None):
        # Handle latent noise
        if self.z_dim > 0:
            z = x[:, 0, :, :, -self.z_dim :].clone()
            x = x[:, :, :, :, : -self.z_dim]
            z = (
                z.permute(0, 3, 1, 2)
                .unsqueeze(-1)
                .repeat(1, 1, 1, 1, x.shape[-1])
            )
            x = torch.cat((x, z), dim=1)

        # Reshape for temporal attention
        b, c, h, w, t = x.shape
        x_input = x.permute(0, 4, 1, 2, 3).reshape(b * t, c, h, w)

        # Encoder-decoder processing
        enc_ftrs = self.encoder(x_input)

        if context is not None:
            context = (
                context.unsqueeze(1).repeat(1, t, 1).view(-1, self.cond_dim)
            )
            context = (
                context.unsqueeze(2)
                .unsqueeze(2)
                .expand(-1, -1, enc_ftrs[-1].shape[2], enc_ftrs[-1].shape[3])
            )
            enc_ftrs[-1] = torch.cat((context, enc_ftrs[-1]), 1)

        dec_out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        dec_out = self.head_dec(
            dec_out
        )  # (b*t, num_class, h, w) = (b, t, 2, 32, 32)
        # Temporal attention processing
        temporal_features = dec_out.view(b, t, -1).permute(0, 2, 1)

        temporal_features = self.temporal_attention(temporal_features)
        out = temporal_features.permute(0, 2, 1).reshape(b * t, -1, h, w)

        # Final head and activation
        # out = self.head(temporal_features)
        out = self.final_act(out)

        # Reshape back to temporal dimensions
        out = out.view(b, t, *out.shape[1:]).permute(0, 2, 3, 4, 1)
        if self.retain_dim:
            out = F.interpolate(out, size=(h, w, t))

        return out


class UNetC2STReactionDiffusion(nn.Module):
    def __init__(
        self,
        enc_chs=(2, 16, 32),
        dec_chs=(32, 16),
        num_class=2,
        cond_dim=2,
        z_dim=0,
        t_dim=16,
        retain_dim=True,
        final_act=None,
        device="cuda",
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.t_dim = t_dim
        self.state_dim = num_class
        self.grid_size = 32
        if t_dim > 0 and z_dim > 0:
            enc_chs = (enc_chs[0] * (t_dim + z_dim),) + enc_chs[1:]
        elif t_dim > 0:
            enc_chs = (enc_chs[0] * t_dim,) + enc_chs[1:]
        elif z_dim > 0:
            enc_chs = (enc_chs[0] * z_dim,) + enc_chs[1:]

        self.encoder = Encoder2d(enc_chs)
        self.decoder = Decoder2d(dec_chs, cond_size=cond_dim)
        self.head = nn.Conv2d(dec_chs[-1], self.state_dim, 1)
        self.final_layer = nn.Linear(
            self.state_dim * self.t_dim * self.t_dim, 1
        )
        self.final_act = nn.Sigmoid()

        self.retain_dim = retain_dim
        self.activation_str = "relu"
        self.device = device
        self.to(device)

    def forward(self, x, context=None):
        if x.ndim == 2:
            if self.cond_dim > 0:
                context = x[:, -self.cond_dim :]
                x = x[:, : -self.cond_dim]
            x = x.view(
                x.shape[0],
                self.state_dim,
                self.grid_size,
                self.grid_size,
                self.t_dim,
            )
        # x shape (batch, 2, grid, grid, time+z_dim)
        # Reshape from (batch, 2, grid, grid, (time + z_dim)) to
        #              (batch, 2* (time + z_dim), grid, grid)
        b = x.shape[0]
        if self.t_dim > 0:
            x_input = x.reshape(
                (b, self.state_dim, self.grid_size, self.grid_size, -1)
            )
            b, c, h, w, t = x_input.shape
            x_input = x.permute(0, 4, 1, 2, 3)
            x_input = x_input.reshape(b, c * t, h, w)

        else:
            x_input = x

        # Standard U-Net processing
        enc_ftrs = self.encoder(x_input)
        if context is not None:
            context = (
                context.unsqueeze(2)
                .unsqueeze(2)
                .expand(-1, -1, enc_ftrs[-1].shape[2], enc_ftrs[-1].shape[3])
            )
            enc_ftrs[-1] = torch.cat((context, enc_ftrs[-1]), 1)

        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)

        if self.retain_dim:
            out = F.interpolate(out, x.shape[-1])

        out = out.view(b, -1)
        out = self.final_layer(out)
        out = self.final_act(out)

        return out


if __name__ == "__main__":
    nrows = 70
    x_dim = (2, 32, 32, 16)
    z_dim = 3
    c_dim = 2
    if z_dim > 0:
        x_dim = x_dim[:-1] + (x_dim[-1] + z_dim,)

    device = torch.device("cpu")
    model = UNetReaction3DDiffusion(z_dim=z_dim, cond_dim=c_dim).to(device)
    x = torch.randn(nrows, *x_dim).to(
        device
    )  # Batch size of n_samples, 1 channel, 20x50 image
    c = torch.randn((nrows, c_dim)).to(device)
    print(x.shape)
    output = model(x, context=c)
    print(output.shape)  # Should print torch.Size([5, 1])
