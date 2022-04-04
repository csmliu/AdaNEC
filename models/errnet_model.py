import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class ERRNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if 'idx' in data:
            self.idx = data['idx'].to(device=self.gpu_ids[0])
            self.idx_vec = torch.eye(self.opt.nModel)[self.idx].to(device=self.gpu_ids[0], dtype=torch.bool)

        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])                
        
        self.input = input
        
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True
        
        if target_t is not None:            
            self.target_edge = self.edge_map(self.target_t)         
            
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_i = tensor2im(self.output_i)
            target = tensor2im(self.target_t)

            if self.aligned:
                res = index.quality_assess(output_i, target)
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 'transmission_layer', str(self._count)+'.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'blended', str(self._count)+'.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return 
        
        with torch.no_grad():
            output_i = self.forward()
            output_i = tensor2im(output_i)
                # if os.path.exists(join(savedir, name,'t_output.png')):
                #     i = 2
                #     while True:
                #         if not os.path.exists(join(savedir, name,'t_output_{}.png'.format(i))):
                #             Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output_{}.png'.format(i)))
                #             break
                #         i += 1
                # else:
                #     Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output.png'))
            if self.data_name is not None and savedir is not None:                
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class Predictor(nn.Module):
    """ This is the feature extractor in the paper. """
    def __init__(self, feats=64, in_c=3, inter_c=32, n_layers=5):
        super().__init__()

        layers = []
        prev_c, curr_c = in_c, inter_c
        for i in range(1, n_layers+1):
            layers.append(('conv%d'%i, nn.Conv2d(prev_c, curr_c, 4, 2, 1)))
            layers.append(('relu%d'%i, nn.ReLU(True)))
            prev_c, curr_c = curr_c, curr_c*2
        
        layers.append(('avg', nn.AdaptiveAvgPool2d(1)))
        layers.append(('flatten', nn.Flatten()))
        layers.append(('FC', nn.Linear(prev_c, feats)))

        self.model = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        return self.model(x)


class Attention(nn.Module):

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, q, k, v, attn_=None):
        ##########################    Input shape    ##########################
        # q = [N n_head 1 d_k]
        # k = [N n_head len_k d_k]    # len_k is the number of k's
        # v = [N len_k 3 H W]
        ##########################   k * v -> attn   ##########################
        # q =                           [N n_head 1 d_k]
        #                                      @
        # k = [N n_head len_k d_k]  --> [N n_head d_k len_k]
        #                                      ↓
        # attn =                        [N n_head 1 len_k]
        #######################################################################

        attn = (q / self.temperature) @ (k.transpose(2, 3))
        if attn_ is None:
            attn_ = F.softmax(attn, dim=-1)
        else:
            attn_ = torch.tensor([[[attn_]]], dtype=torch.float32, device=q.device)

        v_ = v.permute(0, 2, 3, 4, 1).unsqueeze(4)
        attn_ = attn_.permute(0, 2, 3, 1)[:, None, None]
        res_ = v_ @ attn_
        res = res_[..., 0, :].permute(0, 4, 1, 2, 3)

        C = attn.shape[-1]
        return res, attn.view(-1, C)


class CDAM(nn.Module):

    def __init__(self, n_head=1, d_model=64, d_k=64):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)

        self.attention = Attention(temperature=d_k**0.5)
    
    def forward(self, q, k, v, attn_=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k = q.size(0), q.size(1), k.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        res, attn = self.attention(q, k, v, attn_)
        return res, attn


class ERRNetModel(ERRNetBase):
    def name(self):
        return 'errnet'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_network(self):
        pass
        # print('--------------------- Model ---------------------')
        # print('##################### NetG #####################')
        # networks.print_network(self.net_i)
        # if self.isTrain and self.opt.lambda_gan > 0:
        #     print('##################### NetD #####################')
        #     networks.print_network(self.netD)

    def _eval(self):
        try:
            self.net_i_list.eval()
            self.net_p_list.eval()
            self.net_p.eval()
            self.net_A.eval()
        except:
            pass

    def _train(self):
        try:
            self.net_i_list.train()
            self.net_p_list.train()
            self.net_p.train()
            self.net_A.train()
        except:
            pass

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if self.isTrain:
            assert self.opt.nModel == len(self.opt.icnn_path)
        self.net_i_list = nn.ModuleList()
        self.net_p_list = nn.ModuleList()

        in_channels = 3
        self.vgg = None
        
        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        
        if not self.isTrain and self.opt.nModel == 1:
            self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
            networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        else:
            for idx in range(self.opt.nModel):
                net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
                networks.init_weights(net_i, init_type=opt.init_type)
                self.net_i_list.append(net_i)

                netP = Predictor().to(self.device)
                self.net_p_list.append(netP)
            
            self.net_p = Predictor().to(self.device)
            self.net_A = CDAM().to(self.device)

        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss
            self.loss_dic['ce'] = nn.CrossEntropyLoss()

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            param_list = [self.net_p_list.parameters(), self.net_p.parameters(), self.net_A.parameters()]
            self.optimizer_G = torch.optim.Adam(itertools.chain(*param_list),
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_ce = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
        
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        if self.opt.lambda_ce > 0:
            attn_shape, idx_shape = self.attn.shape[0], self.idx.shape[0]
            if attn_shape != idx_shape:
                assert attn_shape % idx_shape == 0
                idx = self.idx.repeat_interleave(attn_shape//idx_shape, 0)
            else:
                idx = self.idx
            self.loss_ce = self.loss_dic['ce'](self.attn, idx)
            self.loss_G += self.loss_ce * self.opt.lambda_ce

        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        if not self.isTrain and self.opt.nModel == 1:
            output_i = self.net_i(input_i)
            self.output_i = output_i
            return self.output_i

        with torch.no_grad():
            self.v = torch.stack([net(input_i) for net in self.net_i_list], dim=1)
            N, X, C, H, W = self.v.shape
        self.q = self.net_p(self.input).unsqueeze(1)  # [N 1 C]
        self.k = torch.stack([net(self.input) for net in self.net_p_list], dim=1)  # [N X C]
        if self.isTrain:
            _, self.attn = self.net_A(self.q, self.k, self.v)         # for IDE Loss
            
            self.v = self.v[~self.idx_vec].reshape(N, X-1, C, H, W)   # [N X-1 C H W]
            self.k = self.k[~self.idx_vec].reshape(N, X-1, -1)        # [N X-1 C]
            self.attn_res, _ = self.net_A(self.q, self.k, self.v)     # for v_i^C $\mathbf{v}_\mathit{i}^\complement
        else:  # use all experts in the testing phase
            self.attn_res, self.attn = self.net_A(self.q, self.k, self.v, self.opt.avg)
            if self.opt.show_expertise_level:
                print(self.attn)
        self.output_i = self.attn_res.reshape(N, -1, H, W)
        
        return self.output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()
        
        if self.loss_ce is not None:
            ret_errors['CE'] = self.loss_ce.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if model.opt.nModel == 1 and icnn_path[0] is None:
            model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i_list[0].load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            if len(icnn_path) == 1 and model.opt.nModel == 1:
                state_dict = torch.load(icnn_path[0])
                model.net_i.load_state_dict(state_dict['icnn'])
            elif len(icnn_path) == 1 and model.opt.nModel != 1:
                state_dict = torch.load(icnn_path[0])
                model.net_i_list.load_state_dict(state_dict['icnn'])
                if hasattr(model, 'net_p_list'):
                    model.net_p_list.load_state_dict(state_dict['net_p_list'])
                if hasattr(model, 'net_p'):
                    model.net_p.load_state_dict(state_dict['net_p'])
                if hasattr(model, 'net_A'):
                    model.net_A.load_state_dict(state_dict['net_A'])
                model.epoch = state_dict['epoch']
                model.iterations = state_dict['iterations']
            else:
                assert len(icnn_path) == model.opt.nModel
                for idx, ckpt in enumerate(icnn_path):
                    state_dict = torch.load(ckpt)
                    model.net_i_list[idx].load_state_dict(state_dict['icnn'])
                    model.epoch = state_dict['epoch']
                    model.iterations = state_dict['iterations']

        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i_list.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }
        
        if hasattr(self, 'net_p_list'):
            state_dict.update({'net_p_list': self.net_p_list.state_dict()})
        if hasattr(self, 'net_p'):
            state_dict.update({'net_p': self.net_p.state_dict()})
        if hasattr(self, 'net_A'):
            state_dict.update({'net_A': self.net_A.state_dict()})

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict


class NetworkWrapper(ERRNetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_network(self):
        print('--------------------- NetworkWrapper ---------------------')
        networks.print_network(self.net)

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[31]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
                
            else:
                raise NotImplementedError            
            
            self.loss_dic['t_cx'] = cxloss

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

            # define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
            self._init_optimizer([self.optimizer_D])
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)
        
    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False
                    
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
                
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            # self.loss_G += self.loss_icnn_pixel
            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
            # self.loss_G += self.loss_fm * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        raise NotImplementedError
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
