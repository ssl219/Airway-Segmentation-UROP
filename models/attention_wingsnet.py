import torch 
import torch.nn as nn
from wingsnet_blocks import SSEConv, SSEConv2
from attention_blocks import AttnGatingBlock, WingsNetGatingSignal


class AttnWingsNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=1, supervision_mode = 'encode_decode'):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(AttnWingsNet, self).__init__()
        self.ec1 = SSEConv(self.in_channel, 8, self.out_channel2, bias=self.bias)
        self.ec2 = SSEConv(8, 16, self.out_channel2, bias=self.bias)
        self.ec3 = SSEConv(16, 32, self.out_channel2, bias=self.bias, dilation=2)
        
        self.ec4 = SSEConv2(32, 32, self.out_channel2, bias=self.bias, down_sample=2)
        self.ec5 = SSEConv2(32, 32, self.out_channel2, bias=self.bias, dilation=2, down_sample=2)
        self.ec6 = SSEConv2(32, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=2)
        
        self.ec7 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.ec8 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4)
        self.ec9 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, dilation=2, down_sample=4)
    
        self.ec10 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec11 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
        self.ec12 = SSEConv2(64, 64, self.out_channel2, bias=self.bias, down_sample=8)
    
        self.pool0 = nn.MaxPool3d(kernel_size=[2,2,2],stride=[2,2,2],return_indices =False)
        self.pool1 = nn.MaxPool3d(kernel_size=[2,2,2],stride=[2,2,2],return_indices =False)
        self.pool2 = nn.MaxPool3d(kernel_size=[2,2,2],stride=[2,2,2],return_indices =False)

        self.up_sample0 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dc1 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=4)
        self.dc2 = SSEConv2(64, 64, self.out_channel2, bias=self.bias,  down_sample=4)
        self.dc3 = SSEConv2(128, 64, self.out_channel2, bias=self.bias, down_sample=2)
        self.dc4 = SSEConv2(64, 32, self.out_channel2, bias=self.bias,  down_sample=2)
        self.dc5 = SSEConv(64, 32, self.out_channel2, bias=self.bias,  down_sample=1)
        self.dc6 = SSEConv(32, 16, self.out_channel2, bias=self.bias,  down_sample=1)

        self.attn1 = AttnGatingBlock(in_channels_x=64,in_channels_g=64,out_channels=64)
        self.attn2 = AttnGatingBlock(in_channels_x=64,in_channels_g=64,out_channels=64)
        self.attn3 = AttnGatingBlock(in_channels_x=32,in_channels_g=32,out_channels=32)

        self.signal_gate1 =  WingsNetGatingSignal(in_channels = 64, out_channels = 64, is_batchnorm=True)
        self.signal_gate2 =  WingsNetGatingSignal(in_channels = 64, out_channels = 64, is_batchnorm=True)
        self.signal_gate3 =  WingsNetGatingSignal(in_channels = 32, out_channels = 32, is_batchnorm=True)

        
        if supervision_mode == 'deep':
          self.dropout1 = droplayer(channel_num=2, thr=0.3)
          self.dropout2 = droplayer(channel_num=2, thr=0.3)
          self.dropout3 = droplayer(channel_num=2, thr=0.3)
          self.dc0_0 = nn.Sequential(
                nn.Conv3d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))
          self.dc0_1 = nn.Sequential(
                nn.Conv3d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))
          self.dc0_2 = nn.Sequential(
                nn.Conv3d(2, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))
        
        elif supervision_mode in ['encode_decode','decode']:
          self.dropout1 = droplayer(channel_num=24, thr=0.3)
          self.dropout2 = droplayer(channel_num=12, thr=0.3)
          self.dc0_0 = nn.Sequential(
                nn.Conv3d(24, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))
          self.dc0_1 = nn.Sequential(
                nn.Conv3d(12, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))

        self.act = nn.Sigmoid()

        self.supervision_mode = supervision_mode

                
    def forward(self, x):
        e0, s0 = self.ec1(x)
        e1, s1 = self.ec2(e0)
        e1, s2 = self.ec3(e1) 
    
        e2 = self.pool0(e1)
        e2, s3 = self.ec4(e2)
        e3, s4 = self.ec5(e2)
        e3, s5 = self.ec6(e3)  

        e4 = self.pool1(e3)
        e4, s6 = self.ec7(e4) 
        e5, s7 = self.ec8(e4)
        e5, s8 = self.ec9(e5)
      
        e6 = self.pool2(e5)
        e6, s9 = self.ec10(e6)
        e7, s10 = self.ec11(e6) 
        e7, s11 = self.ec12(e7) 
        
        g1 = self.signal_gate1(e7)
        at1 = self.attn1(e5,g1)

        e8 = self.up_sample0(e7)
        d0, s12 = self.dc1(torch.cat((e8,at1),1))
        d0, s13 = self.dc2(d0)

        g2 = self.signal_gate2(d0)
        at2 = self.attn2(e3,g2)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1,at2),1))
        d1, s15 = self.dc4(d1)

        g3 = self.signal_gate3(d1)
        at3 = self.attn3(e1,g3)
        
        d2 = self.up_sample2(d1)
        d2, s16 = self.dc5(torch.cat((d2,at3),1))
        d2, s17 = self.dc6(d2)
        
        if self.supervision_mode == 'encode_decode':
          if not self.training:
            pred1 = self.dc0_1(self.dropout2(torch.cat((s12,s13,s14,s15,s16,s17),1)))
            return self.act(pred1)
          else:
            pred1 = self.dc0_1(self.dropout2(torch.cat((s12,s13,s14,s15,s16,s17),1)))
            pred0 = self.dc0_0(self.dropout1(torch.cat((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11),1)))
            return pred0, pred1
        elif self.supervision_mode == 'decode':
          pred1 = self.dc0_1(self.dropout2(torch.cat((s12,s13,s14,s15,s16,s17),1)))
          if not self.training:
            return self.act(pred1)
          else:
            return pred1
        elif self.supervision_mode == 'deep':
          if not self.training:
            pred2 = self.dc0_2(self.dropout3(s17))
            return self.act(pred2)
          else:
            pred0 = self.dc0_0(self.dropout1(s13))
            pred1 = self.dc0_1(self.dropout2(s15))
            pred2 = self.dc0_2(self.dropout3(s17))
            return pred0,pred1,pred2