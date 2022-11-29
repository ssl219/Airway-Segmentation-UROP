import torch 
import torch.nn as nn

class WingsNetGatingSignal(nn.Module):
  def __init__(self, in_channels = 1, out_channels = 2, is_batchnorm=False):
    super(WingsNetGatingSignal, self).__init__()
    self.is_batchnorm = is_batchnorm
    self.out_channels = out_channels
    self.act = nn.ReLU()
    self.conv = nn.Conv3d(in_channels=in_channels, out_channels=self.out_channels, 
                  kernel_size=1, stride=1)
    self.norm = nn.BatchNorm3d(num_features = self.out_channels)

  def forward(self, input):
    x = self.conv(input)
    if self.is_batchnorm:
      x = self.norm(x)
    x = self.act(x)
    return x

class AttnGatingBlock(nn.Module):
  def __init__(self, in_channels_x=1,in_channels_g=1,out_channels=2):
    super(AttnGatingBlock, self).__init__()
    self.act = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    self.out_channels = out_channels
    self.in_channels_x = in_channels_x
    self.in_channels_g = in_channels_g
    self.conv_in = nn.Conv3d(in_channels=self.in_channels_x, out_channels=out_channels, 
                        kernel_size=2, stride=2, padding=0,bias=False)
    self.conv_gate = nn.Conv3d(in_channels=self.in_channels_g, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0,bias=True)
    self.conv_cat = nn.Conv3d(in_channels=out_channels, out_channels=1, 
                    kernel_size=1,stride=1, padding=0)
    self.conv_final = nn.Conv3d(in_channels=self.in_channels_x, out_channels=self.in_channels_x, 
                       kernel_size=1, padding='same')
    self.norm =  nn.BatchNorm3d(num_features = self.in_channels_x)
    
  def forward(self, x, g):
    theta = self.conv_in(x)
    phi = self.conv_gate(g)

    phi = F.upsample(phi, size=theta.shape[2:], mode='trilinear')

    y = torch.add(phi, theta)
    y = self.act(y)

    y = self.conv_cat(y)
    y = self.sigmoid(y)

    y = F.upsample(y, size=x.shape[2:], mode='trilinear')
    y = torch.repeat_interleave(y, self.in_channels_x, dim=1)
    y = torch.mul(y, x)

    y = self.conv_final(y)
    y = self.norm(y)
    return y
