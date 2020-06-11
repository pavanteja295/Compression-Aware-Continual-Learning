import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
from models.customnet_SVD import SVD_Conv2d
import pdb

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg16_bn_cifar100'
]

class Sequential_Debug(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, size=1):
        super(VGG, self).__init__()
        self.features = features
        self.last = nn.Linear(512, num_classes)

        self._initialize_weights()

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        # import pdb; pdb.set_trace()
        x = x.view(-1, 512)
        x = self.logits(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def GetNs(self):
        Ns = []
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamN is not None:
                Ns.append(m.ParamN)
        return Ns

    def GetCs(self):
        Cs = []
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamC is not None:
                Cs.append(m.ParamC)
        return Cs

    def GetSigmas(self):
        Sigmas = []
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                Sigmas.append(m.ParamSigma)
        return Sigmas

    def print_modules(self):
        i = 0
        FLOPs = 0
        origin_FLOPs = 0
        for m in self.modules():
            if isinstance(m,SVD_Conv2d):
                #temp_height = current_size[0]
                current_size = m.output_size
                # current_size[0]=int((current_size[0]+2*m.padding-1*(m.kernel_size-1)-1)/m.stride+1)
                # current_size[1]=int((current_size[1]+2*m.padding-1*(m.kernel_size-1)-1)/m.stride+1)
                feature_size = current_size[2]*current_size[3]
                origin_FLOPs += feature_size*m.kernel_size**2*m.input_channel*m.output_channel
                if m.ParamSigma is not None:
                    rank =  np.count_nonzero(m.ParamSigma.data.cpu().numpy())
                    print("SVD_Conv2d%d:\tinchannel:%d\toutchannel:%d\tkernel_size:%d\tstride:%d\tRank:%d"%(i,m.input_channel,m.output_channel,m.kernel_size,m.stride,rank))
                    if m.decompose_type == 'channel':
                        FLOPs+=feature_size*m.kernel_size**2*m.input_channel*rank
                        FLOPs+=feature_size*1*rank*m.output_channel
                    else:
                        feature_size1 = feature_size*m.stride
                        FLOPs+=feature_size1*m.kernel_size*1*m.input_channel*rank
                        FLOPs+=feature_size*m.kernel_size*1*rank*m.output_channel
                else:
                    print("Normal_Conv2d%d:\tinchannel:%d\toutchannel:%d\tkernel_size:%d\tstride:%d"%(i,m.input_channel,m.output_channel,m.kernel_size,m.stride))
                    FLOPs+=origin_FLOPs
                i+=1
        print("FLOPs:%fM\tOrigin FLOPs:%fM\tSpeedUp: %fx"%(FLOPs/1e6,origin_FLOPs/1e6,origin_FLOPs/FLOPs))

    def pruning(self):
        
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                # import pdb; pdb.set_trace()
                valid_idx = torch.arange(m.Sigma.size(0))[m.Sigma!=0]

                valid_idx = torch.arange(m.Sigma.size(0))[m.Sigma!=0]
                # import pdb; pdb.set_trace()
                nm_t = m.Sigma
                rem_rank = nm_t.size(0) - len(valid_idx)
                m.N.data = torch.cat([m.N[:,valid_idx].cpu(), torch.empty(m.output_channel,rem_rank)], dim = 1).cuda()
                # import pdb; pdb.set_trace()
                m.C.data = torch.cat([m.C[valid_idx,:].cpu(), torch.empty(rem_rank, m.C.shape[1])], dim=0).cuda()
                m.Sigma.data = torch.cat([m.Sigma[valid_idx].cpu(), torch.empty(rem_rank)]).cuda()

    def get_rank(self,sensitivity):
        rank = 0
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                if sensitivity is not None:
                    energy_sort_singular,_ = torch.sort(m.ParamSigma**2, descending=True)
                    current_energy = 0.0
                    current_total_energy = torch.sum(energy_sort_singular)
                    idx = 0
                    while current_energy/current_total_energy<1-sensitivity:
                        current_energy+=energy_sort_singular[idx]
                        idx+=1
                    rank+=idx
                else:
                    rank+=m.ParamSigma.size()[0]
        return rank

    def init_from_normal_conv(self,conv_module):
        Ws = []
        Ns = []
        Ss = []
        Cs = []
        Bs = []
        strides = []
        paddings = []
        kernels = []
        for m in conv_module.modules():
            if isinstance(m,torch.nn.Conv2d):
                weight = m.weight.view(m.out_channels,m.in_channels*m.kernel_size[0]*m.kernel_size[1])
                N,S,C = torch.svd(weight, some=True)
                Ws.append(weight)
                Ns.append(N)
                Ss.append(S)
                Cs.append(C)
                Bs.append(m.bias)
                strides.append(m.stride[0])
                paddings.append(m.padding[0])
                kernels.append(m.weight.size(2))
        i = 0
        for m in self.modules():
            if isinstance(m,SVD_Conv2d):
                if m.ParamSigma is not None:
                    m.N.data = Ns[i]
                    m.Sigma.data = Ss[i]
                    m.C.data = Cs[i].transpose(0,1)
                    m.bias = Bs[i]
                    print(torch.sum((m.N.mm(m.Sigma.diag()).mm(m.C)-Ws[i])**2))
                else:
                    m.conv2d.weight = Ws[i]
                    m.conv2d.bias = Bs[i]
                m.stride = strides[i]
                m.padding = paddings[i]
                m.kernel_size = kernels[i]
                i+=1
            

    def standardize_svd(self):
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                NC = m.N.matmul(m.Sigma.abs().diag()).matmul(m.C)
                N,S,C = torch.svd(NC,some = True)
                C = C.transpose(0,1)
                m.N.data = N
                m.C.data = C
                m.Sigma.data = S


def make_layers_cifar100(cfg, batch_norm=False, SVD_=False, rank=[]):
    layers = []
    in_channels = 3
    if SVD_:
            conv_mod_ = SVD_Conv2d
    else:
            conv_mod_ = torch.nn.Conv2d
    ind_ = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if SVD_:
                conv2d = conv_mod_(in_channels, v, kernel_size=3, padding=1, bias=False, rank=rank[ind_] if rank else None)
                ind_ = ind_ + 1
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    #Remove for now lets try one experiment with and without this
    # layers += [
    #     View(-1, 512),
    #     nn.Linear(512, 4096),
    #     nn.ReLU(True),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(True)
    # ]

    return Sequential_Debug(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def vgg16_bn_cifar100(pretrained=False, dataset_history=[], dataset2num_classes={}, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    return VGG(make_layers_cifar100(cfg['D'], batch_norm=True), dataset_history, dataset2num_classes, **kwargs)


def vgg16_bn_cifar100_SVD(pretrained=False, rank=[]):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # import pdb; pdb.set_trace()
    return VGG(make_layers_cifar100(cfg['D'], batch_norm=True, SVD_=True, rank=rank), num_classes=10)