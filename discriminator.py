"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""
import argparse
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3) -> None:
        super().__init__()
        
        layers = [nn.Conv2d(args.image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                        2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="discriminator")
    parser.add_argument('--image_channels', type=int, default=3, help='')

    args = parser.parse_args()
    
    d = Discriminator(args)
    
    x = torch.randn(1, 3, 256, 256)
    print(d(x).shape)
    import pdb;pdb.set_trace()
    print()