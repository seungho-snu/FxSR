import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from .base_model import BaseModel
import models.modules.RRDBNet_arch as RRDBNet_arch

logger = logging.getLogger('base')

class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.rank = -1  # non dist training
        opt_net = opt['network_G']
        self.netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
        self.netG = self.netG.to(self.device)

        self.print_network()
        self.load()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ

        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def test(self, opt, logger, img_name):
        self.netG.eval()

        with torch.no_grad():
            self.texture_gain = opt.T_ctrl
            image_size = self.var_L.shape
            w = image_size[2]
            h = image_size[3]
            gain_channel = torch.ones([1, 1, w, h])

            for i in range(0, 1):
                t = gain_channel[i, 0, :, :]*self.texture_gain
                gain_channel[i, 0, :, :] = t

            self.fake_H = self.netG((self.var_L, gain_channel.to(self.device)))

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
