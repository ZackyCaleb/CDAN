import torch
from tqdm import tqdm
import numpy as np
from cdan import CDAN
import os
import glob
from torchvision import utils as vutils
from models.Discriminator import FeatureDiscriminator, AuDiscriminator

class Solver(object):
    def __init__(self, dataset_loader, config):
        self.data_loader = dataset_loader
        self.cdan = CDAN().to(config.device)
        # self.nl_model = NLayerDiscriminator(input_nc=3,
        #                                          n_layers=3,
        #                                          use_actnorm=False,
        #                                          ndf=64
        #                                          ).apply(weights_init).to(opt.device)
        #
        # self.au_mdoel = au_discriminator()

        # self.dis_model = SplitDiscriminator()
        self.Feature_dis = FeatureDiscriminator()
        self.Au_dis = AuDiscriminator()
        # self.opt_vq, self.opt_nl, self.opt_au = self.configure_optimizers(opt)
        # self.opt_vq, self.opt_dis = self.configure_optimizers(opt)
        self.opt_vq, self.opt_dis_fea, self.opt_dis_au = self.configure_optimizers(config)
        self.train(config)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_ae = torch.optim.Adam(list(self.cdan.encoder.parameters()) +
                                  list(self.cdan.decoder1.parameters()) +
                                  list(self.cdan.decoder2.parameters()) +
                                  # list(self.cdan.quantier.parameters()) +
                                  # list(self.cdan.classifier.parameters()) +
                                  list(self.cdan.quant_conv.parameters()) +
                                  list(self.cdan.post_quant_conv.parameters()),
                                  lr=lr, betas=(args.beta1, args.beta2))
        # opt_nl = torch.optim.Adam(self.nl_model.parameters(),
        #                             lr=lr, betas=(0.5, 0.999))
        #
        # opt_au = torch.optim.Adam(self.au_mdoel.parameters(),
        #                             lr=lr, betas=(0.5, 0.999))
        # opt_dis = torch.optim.Adam(self.dis_model.parameters(),
        # opt_dis = torch.optim.Adam(self.dis_model.parameters(),
        #                             lr=lr, betas=(0.5, 0.999))
        opt_dis_fea = torch.optim.Adam(self.Feature_dis.parameters(),
                                    lr=lr, betas=(args.beta1, args.beta2))
        opt_dis_au = torch.optim.Adam(self.Au_dis.parameters(),
                                    lr=lr, betas=(args.beta1, args.beta2))
        # opt_dis = torch.optim.SGD(self.dis_model.parameters(),lr=0.01, momentum=0.9, weight_decay=0.01)
        # return opt_ae, opt_nl, opt_au
        return opt_ae, opt_dis_fea, opt_dis_au

    def set_requires_grad(self, parameters, requires_grad=False):
        if not isinstance(parameters, list):
            parameters = [parameters]
        for param in parameters:
            if param is not None:
                param.requires_grad = requires_grad


    def train(self, args):
        train_dataset = self.data_loader
        # train_dataset, test_dataset=load_data(args.root_path, args)
        for epoch in range(args.epochs):
            acc, nums, dis = 0, 0, 0
            lower_gen_loss, lower_dis_loss = 10.0, 10.0
            all_gen_loss, all_dis_loss = 0, 0
            # try:
            #     images, au = next(train_dataset)
            # except:
            #     data_iter = iter(train_dataset)
            #     images, au = next(data_iter)
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (images, au) in enumerate(tqdm(train_dataset)):
                    imgs = images.to(device=args.device)
                    # imgs = images.cuda()
                    # labels = labels.to(device=args.device)
                    au = torch.squeeze(au.to(device=args.device).type(torch.float32), dim=1)
                    # au = torch.squeeze(au.cuda().type(torch.float32), dim=1)
                    # mask_map, color_map, xrec, au_loss, vq_loss, discloss = self.cdan(imgs)
                    # mask_map, color_map, xrec, mask_input, mask_color, au_loss, vq_loss, discloss = self.cdan(imgs)
                    # mask_map, color_map, xrec, vq_loss, emb_loss, discloss = self.cdan(imgs, labels, au)
                    xrec, color_map, mask_input, mask_color, mask_map = self.cdan(imgs)
                    # NL_loss, AU_loss = self.cdan.dis_loss(imgs, xrec, mask_input, color_map, au, args.epochs)
                    # dis_loss = self.cdan.dis_loss(imgs, xrec, mask_input, mask_color, au)
                    NL_loss, AU_loss = self.cdan.dis_loss(imgs, xrec, mask_input, mask_color, au)
                    dis_loss = NL_loss+AU_loss
                    self.set_requires_grad(self.Feature_dis, True)
                    self.opt_dis_fea.zero_grad()
                    # NL_loss.backward(retain_graph=True)
                    NL_loss.backward()
                    # self.opt_dis_fea.step()

                    self.set_requires_grad(self.Au_dis, True)
                    self.opt_dis_au.zero_grad()
                    # NL_loss.backward(retain_graph=True)
                    AU_loss.backward()
                    self.opt_dis_au.step()

                    #
                    # self.set_requires_grad(self.nl_model, True)
                    # self.opt_nl.zero_grad()
                    # # NL_loss.backward(retain_graph=True)
                    # NL_loss.backward()
                    # self.opt_nl.step()
                    #
                    # self.set_requires_grad(self.au_mdoel, True)
                    # self.opt_au.zero_grad()
                    # # AU_loss.backward(retain_graph=True)
                    # AU_loss.backward()
                    # self.opt_au.step()

                    if i % args.num_iters == 0:
                        # self.set_requires_grad(self.nl_model, False)
                        # self.set_requires_grad(self.au_mdoel, False)
                        # self.set_requires_grad(self.dis_model, False)
                        self.set_requires_grad(self.Feature_dis, False)
                        self.set_requires_grad(self.Au_dis, False)

                        self.opt_vq.zero_grad()
                        # gen_loss = self.cdan.gen_loss(imgs, xrec, mask_input, color_map, mask_map, au, args.epochs)
                        # gen_loss = self.cdan.gen_loss(imgs, xrec, mask_input, mask_color, au)
                        nll_loss, g_loss, au_loss, tv_loss = self.cdan.gen_loss(imgs, xrec, mask_input, mask_color, mask_map, au)
                        gen_loss = nll_loss+g_loss+au_loss+tv_loss
                        gen_loss.backward()
                        self.opt_vq.step()
                        # info  = 'epoch / epoches'.format(epoch, epoch)
                        # pbar.set_postfix(
                        #     # au_loss=np.round(AU_loss.cpu().detach().numpy().item(), 5),
                        #     # NL_loss=np.round(NL_loss.cpu().detach().numpy().item(), 5),
                        #     NL_loss=np.round(dis_loss.cpu().detach().numpy().item(), 5),
                        #     gen_loss=np.round(gen_loss.cpu().detach().numpy().item(), 5),
                        # )
                        pbar.set_postfix(
                            epoch=epoch,
                            dis_loss=np.round(dis_loss.cpu().detach().numpy().item(), 8),
                            NL_loss=np.round(NL_loss.cpu().detach().numpy().item(), 8),
                            AU_loss=np.round(AU_loss.cpu().detach().numpy().item(), 8),
                            gen_loss=np.round(gen_loss.cpu().detach().numpy().item(), 8),
                            nll_loss=np.round(nll_loss.cpu().detach().numpy().item(), 8),
                            g_loss=np.round(g_loss.cpu().detach().numpy().item(), 8),
                            au_loss=np.round(au_loss.cpu().detach().numpy().item(), 8),
                            tv_loss=np.round(tv_loss.cpu().detach().numpy().item(), 8),
                        )
                        pbar.update(1)
                    # print('epoch:', epoch, 'dis_loss:', dis_loss, 'NL_loss:', NL_loss, 'AU_loss:', AU_loss, 'gen_loss:', gen_loss,
                    #       'nll_loss:', nll_loss, 'g_loss:', g_loss, 'au_loss:', au_loss, 'tv_loss:', tv_loss)
                    all_dis_loss += dis_loss
                    all_gen_loss += gen_loss
            all_dis_loss = all_dis_loss/len(train_dataset)
            all_gen_loss = all_gen_loss/len(train_dataset)
            print('{}/{}'.format(epoch, args.epochs))
            '''
            save samples
            '''
            with torch.no_grad():
                mask_map = mask_map.repeat(1, 3, 1, 1)
                real_fake_images = torch.cat((imgs[:4], color_map[:4], mask_map[:4], mask_input[:4], mask_color[:4], xrec[:4]))
                vutils.save_image(real_fake_images, os.path.join("results", f"all_{epoch}.jpg"), nrow=4)
                # vutils.save_image(mask_map[:4], os.path.join("results", f"mask_{epoch}.jpg"), nrow=4)

            if epoch > 300 and lower_dis_loss > all_dis_loss and lower_gen_loss > all_gen_loss:
            # if epoch==args.epochs-1:
                for i in glob.glob(os.path.join(args.save_check, '*.pth')):
                # for i in glob.glob(os.path.join(args.save_check, '*.pth')):
                # for i in glob.glob(os.path.join(args.save_check, '*.pth')):
                    os.remove(i)
                torch.save(self.cdan.state_dict(), os.path.join(args.save_check, f"cdan_affect_HaSa_NoAuSp_{epoch}.pth"))
                # torch.save(self.cdan.state_dict(), os.path.join(args.save_check, f"cdan_2_{epoch}.pth"))
                # torch.save(self.cdan.state_dict(), os.path.join(args.save_check, f"cdan_affect_{epoch}.pth"))
                lower_dis_loss = all_dis_loss
                lower_gen_loss = all_gen_loss

    def rec(self, args):
        model_path = args.save_check
        self.cdan.load_state_dict(torch.load(model_path+f"cdan_affect_HaSa_50.pth"))
        self.cdan.to(args.device)
        img_names = self.data_loader[1]
        data_loader = self.data_loader[0]
        with torch.no_grad():
            for i, (img, _) in enumerate(data_loader):
                xrec, color_map, mask_input, mask_color, mask_map = self.cdan(img)
                imgn = img_names[i]
                imgn = os.path.basename(imgn[0])
                imgn = imgn.split('.')[0]
                # mask_map_color = 1-mask_map
                # real_fake_images = torch.cat((color_map[:4], xrec[:4], mask_input[:4], mask_color[:4], ))
                # vutils.save_image(img[:4], os.path.join(args.save_path, imgn+'_input.jpg'), nrow=4)
                # vutils.save_image(xrec[:4], os.path.join(args.save_path, imgn+'.jpg'), nrow=4)
                # vutils.save_image(mask_input[:4], os.path.join(args.save_path, imgn+'.jpg'), nrow=4)

                vutils.save_image(xrec[:4], os.path.join(args.save_rec, imgn+'_xrec.jpg'), nrow=4)
                # vutils.save_image(mask_map_color[:4], os.path.join(args.save_path,, imgn+'_mask_map_color.jpg'), nrow=4)
                vutils.save_image(mask_input[:4], os.path.join(args.save_rec, imgn+'_mask_input.jpg'), nrow=4)
                vutils.save_image(mask_map[:4], os.path.join(args.save_rec, imgn+'_mask_map.jpg'), nrow=4)
                # vutils.save_image(mask_map[:4], os.path.join(args.save_path, os.path.basename(path)), nrow=4)
                vutils.save_image(mask_color[:4], os.path.join(args.save_rec, imgn+'_mask_color.jpg'), nrow=4)
                vutils.save_image(color_map[:4], os.path.join(args.save_rec, imgn+'_color_map.jpg'), nrow=4)

