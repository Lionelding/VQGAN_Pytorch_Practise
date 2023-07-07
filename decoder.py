import argparse
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolution = [16]
        num_res_blocks = 3
        resolution = 16
        
        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]
        
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
                
        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="decoder")
    parser.add_argument('--latent_dim', type=int, default=32, help='')
    parser.add_argument('--image_channels', type=int, default=3, help='')

    args = parser.parse_args()
    
    decoder = Decoder(args)
    import pdb;pdb.set_trace()
    print()