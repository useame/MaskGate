from ..config import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import re


def show_mask(mask, index=1):
    """
    Given mask produce a plt graph for visualization.

    if channel num is larger than 1, please choose one layer to print
    by setting index. Batch size over 1 is not acceptable please set
    the index in the first dimension
    """
    # feature_map = 1-feature_map
    mask = mask.squeeze(0)
    mask = mask.cpu().numpy()
    feature_map_num = mask.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    # for index in range(1, feature_map_num+1):
    # plt.subplot(row_num, row_num, index)
    plt.imshow(mask[index - 1], cmap='gray')
    plt.axis('off')
    plt.show()
    # plt.savefig(cfg.vis_dir + '/featuremap'+'.jpg')
    plt.close()


class LearnableBernoulliRounding(torch.autograd.Function):
    """
    Generate a 0-1 mask obey Bernoulli distribution

    You can force add skip_rate to maintain a lower bound,
    accuracy drops if you set that too high.
    Negative allowed for skip_rate parameter if you want better accuracy.
    """
    @staticmethod
    def forward(ctx, input, skip_rate=0):
        y = input + skip_rate
        y = torch.clip(y, min=0.0, max=1.0)
        y = torch.bernoulli(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # print("====== Go BernoulliRounding backward")
        return grad_output, None


class BernoulliRounding(torch.autograd.Function):
    """
    Generate a 0-1 mask obey Bernoulli distribution
    """
    @staticmethod
    def forward(ctx, input):
        y = input.clone()
        y = torch.clip(y, min=0.0, max=1.0)
        y = torch.bernoulli(y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # print("====== Go BernoulliRounding backward")
        return grad_output, None


class HardSoftmax(torch.autograd.Function):
    """
    Generate a 0-1 mask simply by rounding
    """
    @staticmethod
    def forward(ctx, input):
        y_hard = input.clone()
        y_hard = y_hard.zero_()
        y_hard[input >= 0.5] = 1
        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GumbelSigmoid(torch.nn.Module):
    """
    Implementation of gumbel softmax for a binary case using gumbel sigmoid.
    """
    def __init__(self):
        super(GumbelSigmoid, self).__init__()
        self.running_mode = None
        self.sigmoid = nn.Sigmoid()

    def normalize_to_one(self, logits):
        logits_max = torch.max(logits)
        logits_min = torch.min(logits)
        gap = logits_max - logits_min
        logits = torch.div(logits - logits_min, gap)
        return logits

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumbel_samples_tensor = -torch.log(
            eps - torch.log(uniform_samples_tensor + eps)
        )
        return gumbel_samples_tensor

    def gumbel_sigmoid_sample(self, logits, temperature):
        """
        Adds noise to the logits and takes the sigmoid. No Gumbel noise during inference.
        """
        if self.running_mode == RunningMode.GatePreTrain or self.running_mode == RunningMode.FineTuning:
            gumbel_samples_tensor = self.sample_gumbel_like(logits.data)
            gumbel_trick_log_prob_samples = logits + gumbel_samples_tensor.data
        else:
            gumbel_trick_log_prob_samples = logits
        # soft_samples = self.sigmoid(gumbel_trick_log_prob_samples / temperature)
        soft_samples = self.normalize_to_one(gumbel_trick_log_prob_samples / temperature)

        return soft_samples

    def gumbel_sigmoid(self, logits, temperature=2 / 3, hard=False):
        out = self.gumbel_sigmoid_sample(logits, temperature)
        if hard:
            out = HardSoftmax.apply(out)
        else:
            out = BernoulliRounding.apply(out)
            # out = LearnableBernoulliRounding.apply(out, self.skip_rate)
        return out

    def forward(self, logits, running_mode, temperature=2 / 3):
        self.running_mode = running_mode
        if self.running_mode == RunningMode.GatePreTrain or self.running_mode == RunningMode.FineTuning:
            return self.gumbel_sigmoid(logits, temperature=temperature, hard=False)
        else:
            return self.gumbel_sigmoid(logits, temperature=temperature, hard=True)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class MaskGate(nn.Module):
    """
    A Mask Gate generate a Masked x to decide which patch can be skipped. Hardware required to accelerate.
    """
    def __init__(self, patch_size, patch_num, input_channel, output_channel, running_mode=RunningMode.FineTuning, dim=256, depth=4, kernel_size=9, fc_dim=4096, fc_depth=0):
        """
        Initialize mask gate.
        :param patch_size: the size of patch in this layer.
        :param patch_num: the total number of patches
        :param input_channel: input channel
        :param output_channel: output channel of the convolution need acceleration, used to calculate MAC reduction
        :param running_mode: indicates the running mode, please import RunningMode as well, Default RunningMode.FineTuning
        :param dim: hyper parameter, namely the input/output channels of inner convolution layer. Default 256
        :param depth: the depth of the convmixer structure. Default 4
        :param kernel_size: kernel size of convolutions. Default 9
        :param fc_dim: fully connected layer input/output channels. Default 4096
        :param fc_depth: fully connected layer number. Default 0, but two FC layers are already contained
        """
        super(MaskGate, self).__init__()
        # output should be patch_num * patch_num tensor mask
        self.vis = False
        self.output_channel = output_channel
        self.input_channel =  input_channel
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.convmixer = nn.Sequential(
            nn.Conv2d(self.input_channel, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                                   nn.GELU(),
                                                   nn.BatchNorm2d(dim)
                                                   )),
                            nn.Conv2d(dim, dim, kernel_size=1),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, fc_dim),
            *[nn.Linear(fc_dim, fc_dim) for i in range(fc_depth)],
            nn.Linear(fc_dim, self.patch_num)
        )
        self.sigmoid = GumbelSigmoid()
        self.set_running_mode(running_mode)

    def set_running_mode(self, running_mode):
        self.running_mode = running_mode
        if self.running_mode == RunningMode.FineTuning:
            for name, p in self.named_parameters():
                p.requires_grad = False
        if self.running_mode == RunningMode.GatePreTrain:
            for name, p in self.named_parameters():
                p.requires_grad = True


    def forward(self, now_input):
        mask_mode = m_cfg.mask_mode
        # current_input = torch.cat((now_input, previous_input), 1)
        mask = self.convmixer(now_input)
        mask = self.sigmoid(mask, self.running_mode)
        index = torch.nonzero(mask)
        h, w = now_input.shape[1:]
        b = now_input.shape[0]
        mask_h = h // self.patch_size
        mask_w = w // self.patch_size

        # mask = torch.reshape(mask, (mask.shape[0], 1, mask_h, mask_w))

        # mask = torch.zeros_like(mask)
        # mask = torch.ones_like(mask)
        # mask = torch.bernoulli(mask * 0.02)
        # mask = 1 - mask

        if mask_mode == MaskMode.Positive:
            current_skip = torch.count_nonzero(mask)
            current_total_patch = self.patch_num * b
            current_skip_rate = current_skip / current_total_patch
            h_out = h
            w_out = w
            current_mac = b * h_out * w_out * self.output_channel
            m_cfg.total_mac += current_mac
            m_cfg.skipped_mac += current_mac * current_skip_rate
            # mac_gates = n * h_out * w_out * c_in * 1 * k_h * k_w (n = batch_size, h_out = (h + 2*p - k)/s + 1 = h + 0 - 1 / 1
            m_cfg.skipped_patch += current_skip
            m_cfg.total_patch += current_total_patch

        if self.vis:
            m_cfg.mask = mask.clone().detach()
        # mask = mask.expand(now_input.shape[0], mask_h, mask_w)

        # m = torch.nn.Upsample(scale_factor=self.patch_size, mode='nearest')
        # mask = m(mask)
        # if self.vis:
        #     show_mask(mask.detach())
        # 1 jiu 0 xin
        # need 0keep 1remove
        x_masked = torch.mul(now_input, 1 - mask)

        # ids_restore =
        # if mask_mode == MaskMode.Positive:
        #     now_input = torch.mul(previous_input, mask) + torch.mul(now_input, 1 - mask)
        # if mask_mode == MaskMode.Negative:
        #     now_input = torch.mul(now_input, mask) + torch.mul(previous_input, 1 - mask)

        return x_masked,mask

    def init_weights(self):
        """
        This function is designed for initializing our mask gate when first constructed
        in your model. If pretrained model is provided, parameters will be loaded.
        Freeze mask gate during fine-tuning. If you want to train our mask gate, please
        do not forget to freeze your backbone during pretraining.
        """
        for m in self.convmixer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.001)
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        # nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.bias, 0.6)
            if isinstance(m, Residual):
                for j in m.fn.modules():
                    if isinstance(j, nn.Conv2d):
                        nn.init.normal_(j.weight, std=0.001)
                        if hasattr(j, 'bias'):
                            if j.bias is not None:
                                nn.init.constant_(j.bias, 0)
                    if isinstance(j, nn.BatchNorm2d):
                        nn.init.constant_(j.weight, 1)
                        nn.init.constant_(j.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        # self.load_model()
        if self.pretrain_flag == RunningMode.FineTuning:
            for p in self.parameters():
                p.requires_grad = False
        if self.pretrain_flag == RunningMode.GatePreTrain:
            for p in self.parameters():
                p.requires_grad = True

    def save_model(self, epoch=0):
        """
        This function is designed to save this gate parameters separately. Only Use this
        function after pretraining because fine-tuning do not update gate parameters
        :param epoch: current epoch number, default 0, but we recommend you to assign epoch
        number to save model in each epoch during one run
        """
        file_name = osp.join(m_cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        state = {
            'epoch': epoch,
            'gate': self.state_dict()
        }
        torch.save(state, file_name)

    def load_model(self):
        """
        This function is designed to load a pretrained model for mask gate. please put the
        model .pth.tar file to PASS/output/model_dump/ (snapshot_0.pth.tar for example)
        the model will be loaded automatically if it contains gate parameter in network state.
        We recommend you to also establish a similar partial model loading function to your
        own backbone as you will change the model structure when you insert our module.
        """
        ckpt = None
        try:
            model_file_list = glob.glob(osp.join(m_cfg.model_dir, '*.pth.tar'))
            cur_epoch = max(list(
                [int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                 model_file_list]), default=0)
            ckpt = torch.load(osp.join(m_cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar'))
        except IOError:
            if self.pretrain_flag != RunningMode.GatePreTrain and self.pretrain_flag != RunningMode.BackboneTest:
                assert 0, "Error: No gate available."
            else:
                print("Warning: no model loaded, train from start")
        if ckpt is None:
            if self.pretrain_flag == RunningMode.GatePreTrain \
                    or self.pretrain_flag == RunningMode.BackboneTrain\
                    or self.pretrain_flag == RunningMode.BackboneTest:
                return
        if 'gate' in ckpt.keys():
            try:
                self.load_state_dict(ckpt['gate'])
            except RuntimeError:
                if self.pretrain_flag != RunningMode.GatePreTrain and self.pretrain_flag != RunningMode.BackboneTrain:
                    assert 0, "Error: Model structure changed, please retrain your mask gate model or delete the wrong model file"
                else:
                    print("Warning: model not loaded, train from start")
        elif 'network' in ckpt.keys():
            keys = ckpt['network'].keys()
            is_loaded = False
            parameters = {}
            for k in keys:
                if 'Gate' in k:
                    a = r'Gate\.(.*?)$'
                    parameter = re.findall(a, k)
                    if len(parameter)>0:
                        parameters[parameter[0]]=ckpt['network'][k].data
            for k, p in self.named_parameters():
                if k in parameters.keys():
                    p.data = parameters[k]
                    is_loaded = True
            if not is_loaded:
                if self.pretrain_flag != RunningMode.GatePreTrain and self.pretrain_flag != RunningMode.BackboneTest:
                    assert 0, "Error: No gate available, please retrain your mask gate model"
                else:
                    print("Warning: no model loaded, train from start")
    def set_vis(self, vis=True):
        self.vis = True
