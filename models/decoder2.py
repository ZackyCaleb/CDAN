import torch
import torch.nn as nn
from .modules import SpatialAttention, ResnetBlock, Normalize, nonlinearity, Upsample


class decoder2(nn.Module):
    def __init__(self, z_channels=256, block_in=512):
        super(decoder2, self).__init__()
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, padding=1)
        self.attn = SpatialAttention()
        self.l0_r0 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=0.0)
        self.l0_r1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=0.0)

        self.l1_r0 = ResnetBlock(in_channels=512, out_channels=512, temb_channels=0, dropout=0.0)
        self.l1_r1 = ResnetBlock(in_channels=512, out_channels=512, temb_channels=0, dropout=0.0)
        # self.l1_r2 = ResnetBlock(in_channels=512, out_channels=512, temb_channels=0, dropout=0.0)
        # self.l1_r3 = ResnetBlock(in_channels=512, out_channels=512, temb_channels=0, dropout=0.0)
        self.l1_up = Upsample(512, True)

        self.l2_r0 = ResnetBlock(in_channels=512, out_channels=256, temb_channels=0, dropout=0.0)
        self.l2_r1 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        # self.l2_r2 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        # self.l2_r3 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        self.l2_up = Upsample(256, True)

        self.l3_r0 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        self.l3_r1 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        # self.l3_r2 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        # self.l3_r3 = ResnetBlock(in_channels=256, out_channels=256, temb_channels=0, dropout=0.0)
        self.l3_up = Upsample(256, True)

        self.l4_r0 = ResnetBlock(in_channels=256, out_channels=128, temb_channels=0, dropout=0.0)
        self.l4_r1 = ResnetBlock(in_channels=128, out_channels=128, temb_channels=0, dropout=0.0)
        # self.l4_r2 = ResnetBlock(in_channels=128, out_channels=128, temb_channels=0, dropout=0.0)
        # self.l4_r3 = ResnetBlock(in_channels=128, out_channels=128, temb_channels=0, dropout=0.0)
        self.l4_up = Upsample(128, True)

        self.norm_out = Normalize(128)
        self.color_out = nn.Sequential(*[torch.nn.Conv2d(128, 3, kernel_size=3,stride=1, padding=1),
                                         nn.Tanh()])
        self.mask_out = nn.Sequential(*[torch.nn.Conv2d(128, 1, kernel_size=3, stride=1,padding=1),
                                        nn.Sigmoid()])
    def  forward(self, z):
        temb = None
        h = self.conv_in(z)

        h = self.l0_r0(h, temb)
        h = self.l0_r1(h, temb)
        h = self.attn(h)

        h = self.l1_r0(h, temb)
        h = self.l1_r1(h, temb)
        # h = self.l1_r2(h, temb)
        # h = self.attn(h)

        h = self.l2_r0(h, temb)
        h = self.l2_r1(h, temb)
        # h = self.l2_r2(h, temb)
        # h = self.attn(h)
        h = self.l2_up(h)

        h = self.l3_r0(h, temb)
        h = self.l3_r1(h, temb)
        # h = self.l3_r2(h, temb)
        # h = self.attn(h)
        h = self.l3_up(h)


        h = self.l4_r0(h, temb)
        h = self.l4_r1(h, temb)
        # h = self.l4_r2(h, temb)
        # h = self.attn(h)
        h = self.l4_up(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        mask = self.mask_out(h)
        # color = self.color_out(h)
        # return mask, color
        return mask