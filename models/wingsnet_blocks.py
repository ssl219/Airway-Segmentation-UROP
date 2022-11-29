import torch 
import torch.nn as nn

class SSEConv(nn.Module):
    def __init__(self, in_channel=1, out_channel1=1, out_channel2=2, stride=1,
                 kernel_size=3, padding=1, dilation=1, down_sample=1,
                 bias=True):
        self.in_channel = in_channel
        self.out_channel = out_channel1
        super(SSEConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel1, kernel_size,
                               stride=stride, padding=padding*dilation,
                               bias=bias, dilation=dilation)
        self.conv2 = nn.Conv3d(out_channel1, out_channel2, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channel1)
        self.act = nn.LeakyReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=down_sample,
                                     mode='trilinear', align_corners=True)
        self.conv_se = nn.Conv3d(out_channel1, 1, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.norm_se = nn.Sigmoid()

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm(e0)
        e0 = self.act(e0)
        e_se = self.conv_se(e0)
        e_se = self.norm_se(e_se)
        e0 = e0 * e_se
        e1 = self.conv2(e0)
        e1 = self.up_sample(e1)
        return e0, e1


class SSEConv2(nn.Module):
    def __init__(self, in_channel=1, out_channel1=1, out_channel2=2, stride=1,
                 kernel_size=3, padding=1, dilation=1, down_sample=1,
                 bias=True):
        self.in_channel = in_channel
        self.out_channel = out_channel1
        super(SSEConv2, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel1, kernel_size,
                               stride=stride, padding=padding*dilation,
                               bias=bias, dilation=dilation)
        self.conv2 = nn.Conv3d(out_channel1, out_channel2, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channel1)
        self.act = nn.LeakyReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=down_sample,
                                     mode='trilinear', align_corners=True)
        self.conv_se = nn.Conv3d(out_channel1, 1, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.norm_se = nn.Sigmoid()
        self.conv_se2 = nn.Conv3d(out_channel1, 1, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.norm_se2 = nn.Sigmoid()

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm(e0)
        e0 = self.act(e0)
        e_se = self.conv_se(e0)
        e_se = self.norm_se(e_se)
        e0 = e0 * e_se
        e_se = self.conv_se2(e0)
        e_se = self.norm_se2(e_se)
        e0 = e0 * e_se
        e1 = self.conv2(e0)
        e1 = self.up_sample(e1)
        return e0, e1


class droplayer(nn.Module):
    def __init__(self, channel_num=1, thr=0.3):
        super(droplayer, self).__init__()
        self.channel_num = channel_num
        self.threshold = thr

    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0], self.channel_num, 1, 1, 1).cuda()
            r[r < self.threshold] = 0
            r[r >= self.threshold] = 1
            r = r*self.channel_num/(r.sum()+0.01)
            return x*r
        else:
            return x
