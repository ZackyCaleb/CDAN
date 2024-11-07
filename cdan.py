import torch
import torch.nn as nn
from models.losses import LPIPS, TVLoss
from models.encoder import encoder
from models.decoder1 import decoder1
from models.decoder2 import decoder2
from models.Discriminator import FeatureDiscriminator, AuDiscriminator

class CDAN(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 num_class=7,
                 ):
        super(CDAN, self).__init__()
        self.num_class = num_class
        self.encoder = encoder()
        self.decoder1 = decoder1()
        self.decoder2 = decoder2()
        self.quant_conv = torch.nn.Conv2d(256, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 256, 1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.ou_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LPIPS().eval()
        # self.identity_model = LightCNN_29Layers_v2().eval()
        # self.identity_model = torch.nn.DataParallel(self.identity_model).cuda()
        # checkpoint = torch.load(r'./dataset/oulu_image/LightCNN_29Layers_V2_checkpoint.pth.tar')
        # self.identity_model.load_state_dict(checkpoint['state_dict'])
        # for param in self.identity_model.parameters():
        #     param.requires_grad = False
        # self.discriminator = NLayerDiscriminator(input_nc=3,
        #                                          n_layers=3,
        #                                          use_actnorm=False,
        #                                          ndf=64
        #                                          ).apply(weights_init)
        # self.discriminator = SplitDiscriminator()
        self.feature_dis = FeatureDiscriminator()
        self.AU_dis = AuDiscriminator()
        self.discriminator_weight = 0.8
        self.disc_factor = 1.0
        self.tv_loss = TVLoss()
        '''加载AU模型'''
        # self.au_mdoel = au_discriminator()
        # self.au_mdoel = MEFARG()
        # check_au = torch.load(r'.\OpenGprahAU-SwinT_second_stage.pth')
        # state_dict = check_au['state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if 'module.' in k:
        #         k = k[7:]   # remove `module.`
        #     new_state_dict[k] = v
        # self.au_mdoel.load_state_dict(state_dict, strict=False)
        # # for param in self.au_mdoel.parameters():
        # #     param.requires_grad = False


    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(torch.relu(1. - logits_real))
        loss_fake = torch.mean(torch.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def color_branch(self, input):
        x = self.post_quant_conv(input)
        x = self.decoder1(x)
        return x

    # def mask_branch(self, input, mask):
    def mask_branch(self, input):
        # x = self.decoder2(input, mask)
        mask = self.decoder2(input)
        return mask

    def calculate_adaptive_weight(self, nll_loss, g_loss):
        decoder1_layer_weight = self.decoder1.conv_out.weight
        decoder2_layer_weight = self.decoder2.conv_out.weight
        d1_grads = torch.autograd.grad(nll_loss, decoder1_layer_weight, retain_graph=True)[0]
        d2_grads = torch.autograd.grad(nll_loss, decoder2_layer_weight, retain_graph=True)[0]
        g1_grads = torch.autograd.grad(g_loss, decoder1_layer_weight, retain_graph=True)[0]
        g2_grads = torch.autograd.grad(g_loss, decoder2_layer_weight, retain_graph=True)[0]
        d_weight = (torch.norm(d1_grads) + torch.norm(d2_grads)) / (torch.norm(g1_grads) + torch.norm(g2_grads)+ 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward_loss(self, inputs, reconstructions, global_step):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + p_loss
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # identity_loss = self.identity_loss(inputs, reconstructions)
        logits_real = self.discriminator(inputs)
        logits_fake = self.discriminator(reconstructions)
        g_loss = -torch.mean(logits_fake)
        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss)
        disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=1.0)

        loss = nll_loss + d_weight * disc_factor * g_loss
        d_loss = disc_factor * self.hinge_d_loss(logits_real, logits_fake)

        return loss, d_loss

    def forward(self, input):
        h = self.encoder(input)
        h = self.quant_conv(h)
        # 输出预测表情预测
        # quant_mask, emb_loss, _ = self.quantier(h)
        # pred = self.au_line(quant_mask)

        # 输出au掩码，图片掩码，合成图片
        # mask_map = self.mask_branch(h, quant_mask)
        mask_map = self.mask_branch(h)
        color_map = self.color_branch(h)
        mask_input = mask_map*input
        mask_color = (1-mask_map)*color_map

        # mask_input = (1-mask_map)*input
        # mask_color = mask_map*color_map
        xrec = mask_input+mask_color

        return xrec, color_map, mask_input, mask_color, mask_map


    def dis_loss(self, inputs, xrec, mask_input, color_map, au):
        # logits_real, au_real = self.discriminator(inputs)
        logits_real = self.feature_dis(inputs)
        au_real = self.AU_dis(inputs)
        # logits_fake, _ = self.discriminator(xrec.detach())
        logits_fake = self.feature_dis(xrec.detach())

        # _, mask_input_au = self.discriminator(mask_input.detach())
        # mask_input_au = self.AU_dis(mask_input.detach())
        # _, color_map_au = self.discriminator(color_map.detach())
        # color_map_au = self.AU_dis(color_map.detach())

        # AU_loss = self.ou_loss(au_real, au) + self.ou_loss(color_map_au, torch.ones_like(au))
        AU_loss = self.ou_loss(au_real, au)

        NL_loss = self.disc_factor * self.hinge_d_loss(logits_real, logits_fake)

        # return 1e-1*NL_loss+30*AU_loss
        # return 1e-1*NL_loss
        return 1e-1*NL_loss, 0*AU_loss

    def gen_loss(self, inputs, xrec, mask_input, color_map, mask_map, au):
        rec_loss = self.l1_loss(xrec, inputs)
        p_loss = self.perceptual_loss(inputs, xrec)
        rec_loss = rec_loss + p_loss
        nll_loss = torch.mean(rec_loss)
        # logits_fake, _ = self.discriminator(xrec)
        logits_fake = self.feature_dis(xrec)
        g_loss = torch.mean(torch.relu(1. - logits_fake))

        # _, mask_input_au = self.discriminator(mask_input)
        mask_input_au = self.AU_dis(mask_input)
        # _, color_map_au = self.discriminator(color_map)
        color_map_au = self.AU_dis(color_map)

        au_loss = self.ou_loss(mask_input_au, au) + self.ou_loss(color_map_au, torch.zeros_like(au))
        # au_loss = self.ou_loss(mask_input_au, au) + torch.relu(color_map_au).sum()
        # au_loss = self.ou_loss(mask_input_au, au)

        tv_loss = 2e-5 * self.tv_loss(mask_input) + 1e-1*torch.mean(mask_input)
        # return 3e-1*nll_loss, 3e-1*g_loss, 60*au_loss, tv_loss
        return 3e-1*nll_loss, 3e-1*g_loss, au_loss, tv_loss


    def test(self, input):
        h = self.encoder(input)
        h = self.quant_conv(h)

        # 输出预测表情预测
        quant_mask, emb_loss, _ = self.quantier(h)
        # pred = self.au_line(quant_mask)

        # 输出au掩码，图片掩码，合成图片
        # mask_map = self.mask_branch(h, quant_mask)
        mask_map = self.mask_branch(h)
        color_map = self.color_branch(h)
        mask_input = mask_map*input
        mask_color = (1-mask_map)*color_map
        xrec = mask_input+mask_color

        # mask_map = self.aug(self.img_cvt(mask_input))
        # mask_map_au = self.au_mdoel(mask_map)

        # return mask_map, mask_map_au
        return mask_map, color_map, xrec, mask_input, mask_color





