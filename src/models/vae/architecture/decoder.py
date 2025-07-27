import torch
from torch import nn

from src._types import Activation


class Decoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_layers,
                 latent_size,
                 num_classes,
                 dims,
                 bottleneck_shape,
                 activation: Activation,
                 last_act: Activation,
                 dropout: float,
                 ):
        assert dims == 3, "Only dims == 3 is supported so far"
        super().__init__()

        input_size = latent_size
        self.num_class = num_classes

        if num_classes:
            self.target_embedding = nn.Embedding(num_classes, input_size)
            input_size *= 2

        layers = [
            nn.Linear(input_size, int(torch.prod(torch.tensor(bottleneck_shape)))),
            nn.Unflatten(1, bottleneck_shape),
        ]

        in_ch = out_ch = bottleneck_shape[0]
        for i in range(num_layers):
            in_ch = out_ch
            out_ch //= 2

            layers += [
                nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.InstanceNorm3d(out_ch),
                activation.get_act()
            ]
            layers += [nn.Dropout3d(p=dropout)]
        in_ch = out_ch
        layers += [
            nn.ConvTranspose3d(in_channels=in_ch, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
            last_act.get_act()
        ]

        self.CNN = nn.Sequential(*layers)

    def forward(self, z, target=None):
        if self.num_class:
            assert target is not None
            target_embed = self.target_embedding(target)
            z = torch.cat((z, target_embed), dim=1)

        x = self.CNN(z)

        return x
