from torch import nn
import torch, pypose as pp
from pypose.optim import LM
from pypose.optim.strategy import Constant
from pypose.optim.scheduler import StopOnPlateau

class InvNet(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        init = pp.randn_SE3(*dim)
        self.pose = pp.Parameter(init)

    def forward(self, input):
        error = (self.pose @ input).Log()
        return error.tensor()

device = torch.device("cuda")

input = pp.randn_SE3(2, 2, device=device)
invnet = InvNet(2, 2).to(device)
strategy = Constant(damping=1e-4)
optimizer = LM(invnet, strategy=strategy)
scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

# # 1st option, full optimization
# scheduler.optimize(input=input)

# 2nd option, step optimization
while scheduler.continual():
    loss = optimizer.step(input)
    scheduler.step(loss)

    print('Loss:', loss)
