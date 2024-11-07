import torch
import torch.nn as nn

class FeatureDiscriminator(nn.Module):
    def __init__(self, input_nc=3, aus_nc=6, image_size=112, ndf=64, n_layers=6):
        super(FeatureDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ]

        cur_dim = ndf
        for n in range(1, n_layers):
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)
        # patch discriminator top
        self.dis_top = nn.Conv2d(cur_dim, 1, kernel_size=kw-1, stride=1, padding=padw, bias=False)
    def forward(self, img):
        embed_features = self.model(img)
        pred_map = self.dis_top(embed_features)
        return pred_map.squeeze()

class AuDiscriminator(nn.Module):
    def __init__(self, input_nc=3, aus_nc=6, image_size=112, ndf=64, n_layers=6):
        super(AuDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ]

        cur_dim = ndf
        for n in range(1, n_layers):
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)

        # AUs classifier top
        k_size = int(image_size / (2 ** n_layers))
        self.aus_top = nn.Conv2d(cur_dim, aus_nc, kernel_size=k_size, stride=1, bias=False)


    def forward(self, img):
        embed_features = self.model(img)
        pred_aus = self.aus_top(embed_features)
        return pred_aus.squeeze()