import torch
from torch import nn
from torch.nn import Flatten

from src.models.vae.architecture.nn import normalization, init_weights


class Encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 ch,
                 num_layers,
                 latent_size,
                 num_classes,
                 num_target_emb_channels,
                 img_size,
                 dims,
                 activation,
                 dropout: float,
                 ):
        assert dims == 3, "Only dims == 3 is supported so far"
        super().__init__()

        self.num_classes = num_classes
        dummy = torch.zeros(1, in_channels, img_size, img_size, img_size)
        if num_classes:
            in_channels += num_target_emb_channels
            self.target_embedding = nn.Embedding(num_classes, num_target_emb_channels)
            dummy = torch.cat([dummy, torch.zeros(1, num_target_emb_channels, img_size, img_size, img_size)], dim=1)

        layers = []
        in_ch = in_channels
        out_ch = ch if num_layers > 0 else in_channels
        for i in range(num_layers):
            layers += [
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_ch),
                activation.get_act()
            ]
            in_ch = out_ch
            out_ch *= 2

        layers += [
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            activation.get_act()
        ]
        self.CNN = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_out = self.CNN(dummy)
            self.bottleneck_shape = dummy_out.shape[1:]
            self.flattened_size = dummy_out.numel()

        layers.append(Flatten())
        self.CNN = nn.Sequential(*layers)

        self.linear_means = nn.Linear(self.flattened_size, latent_size)
        self.linear_log_var = nn.Linear(self.flattened_size, latent_size)

    def forward(self, x, target=None):
        _, _, D, H, W = x.shape
        if self.num_classes:
            assert target is not None
            target_embed = self.target_embedding(target)
            target_embed = target_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target_embed = target_embed.expand(-1, -1, D, H, W)
            x = torch.cat((x, target_embed), dim=1)

        x = self.CNN(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars
