import torch
import torch.nn as nn

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class SpatialAttention(nn.Module):
    def __init__(self, conv_kernel_size=7, pool_kes=7):
        super(SpatialAttention, self).__init__()

        assert conv_kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if conv_kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, conv_kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.pk = pool_kes

    def forward(self, x):
        b, c, h, w = x.shape
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        avg_out = nn.Unfold(kernel_size=self.pk, dilation=1, padding=0, stride=self.pk)(avg_out)
        avg_out_mean = torch.mean(avg_out, dim=1, keepdim=True)
        avg_out_out = torch.tile(avg_out_mean, dims=(1, avg_out.shape[1], 1))
        avg_out_out = nn.Fold(output_size=(h, w), kernel_size=self.pk, dilation=1, padding=0, stride=self.pk)(avg_out_out)

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out = nn.Unfold(kernel_size=self.pk, dilation=1, padding=0, stride=self.pk)(max_out)
        max_out_mean, _ = torch.max(max_out, dim=1, keepdim=True)
        max_out_out = torch.tile(max_out_mean, dims=(1, avg_out.shape[1], 1))
        max_out_out = nn.Fold(output_size=(h,w), kernel_size=self.pk, dilation=1, padding=0, stride=self.pk)(max_out_out)

        x = torch.cat([avg_out_out, max_out_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*input

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4,stride=2,padding=1,bias=False)
    def forward(self, x):
        # x = self.conv1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        return x