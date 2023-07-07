import torch
from helper import NonoLocalBlock, GroupNorm

nlb = NonoLocalBlock(32)
# y = nlb(x)

gn = GroupNorm(32)

x = torch.randn(1, 32, 50, 50)

y = gn(x)
import pdb;pdb.set_trace()

print(y)

