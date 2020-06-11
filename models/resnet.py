import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import torch 
from models.customnet_SVD import SVD_Conv2d

# class SVD_Conv2d(torch.nn.Module):
#     def __init__(self,input_channel,output_channel,kernel_size,stride = 1,padding = 0,
#                 bias = False,SVD_only_stride_1 = False,decompose_type = 'channel'):
#         """
#         stride is fixed to 1 in this module
#         """
#         super(SVD_Conv2d, self).__init__()
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.only_stride_1 = SVD_only_stride_1
#         self.decompose_type = decompose_type
#         self.output_size = None
#         if not SVD_only_stride_1 or self.stride==1:
#             if self.decompose_type == 'channel':
#                 r =  int((input_channel*kernel_size*kernel_size*output_channel) / (output_channel + (input_channel*kernel_size*kernel_size)) ) #min(output_channel,input_channel*kernel_size*kernel_size)
#                 print(f'Now : {r},  Then :  {output_channel,input_channel*kernel_size*kernel_size}' )
#                 self.N = torch.nn.Parameter(torch.empty(output_channel,r))#Nxr
#                 self.C = torch.nn.Parameter(torch.empty(r,input_channel*kernel_size*kernel_size))#rxCHW
#                 self.Sigma = torch.nn.Parameter(torch.empty(r))#rank = r
#             else:#spatial decompose--VH-decompose
#                 r = min(input_channel*kernel_size,output_channel*kernel_size)
#                 self.N = torch.nn.Parameter(torch.empty(input_channel*kernel_size,r))#CHxr
#                 self.C = torch.nn.Parameter(torch.empty(r,output_channel*kernel_size))#rxNW
#                 self.Sigma = torch.nn.Parameter(torch.empty(r))#rank = r
#             self.bias = None
#             if bias:
#                 self.bias = torch.nn.Parameter(torch.empty(output_channel))
#                 self.register_parameter('bias',self.bias)
#                 torch.nn.init.constant_(self.bias,0.0)
#             self.register_parameter('N',self.N)
#             self.register_parameter('C',self.C)
#             self.register_parameter('Sigma',self.Sigma)
#             torch.nn.init.kaiming_normal_(self.N)
#             torch.nn.init.kaiming_normal_(self.C)
#             torch.nn.init.normal_(self.Sigma)
#         else:
#             self.conv2d = nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding,bias = bias)


    # def train(self, mode=True):
    #     self.training = mode
    #     for module in self.children():
    #         print(module)
    #         module.train(mode)
    #     return self

#     def forward(self,x):
#         if not self.only_stride_1 or self.stride==1:
#             if self.training:
#                 r = self.Sigma.size()[0]#r = min(N,CHW)
#                 # C = self.C[:r, :]#rxCHW
#                 # N = self.N[:, :r].contiguous()#Nxr
#                 Sigma = self.Sigma.abs()
#                 C = torch.mm(torch.diag(torch.sqrt(Sigma)), self.C)
#                 N = torch.mm(self.N,torch.diag(torch.sqrt(Sigma)))
#             else:
#                 # import pdb; pdb.set_trace()
#                 valid_idx = torch.arange(self.Sigma.size(0))[self.Sigma!=0]
#                 N = self.N[:,valid_idx].contiguous()
#                 C = self.C[valid_idx,:]
#                 Sigma = self.Sigma[valid_idx].abs()
#                 r = Sigma.size(0)
#                 C = torch.mm(torch.diag(torch.sqrt(Sigma)), C)
#                 N = torch.mm(N,torch.diag(torch.sqrt(Sigma)))
#             if self.decompose_type == 'channel':
#                 #C = C.view(r,self.input_channel,self.kernel_size,self.kernel_size)
#                 NC_ = N@C 
#                 NC_ = NC_.view(self.output_channel, self.input_channel, self.kernel_size,self.kernel_size)
#                 #N = N.view(self.output_channel,r,1,1)
#                 y = torch.nn.functional.conv2d(input = x, weight = NC_,  bias=self.bias,  stride = self.stride,padding = self.padding)
# #                y = torch.nn.functional.conv2d(input = x,weight = C,bias = None,stride = self.stride,padding = self.padding)
# #                y = torch.nn.functional.conv2d(input = y,weight = N,bias = self.bias,stride = 1,padding = 0)
#             else:#spatial decompose
#                 N = N.view(self.input_channel,1,self.kernel_size,r).permute(3,0,2,1)#V:rxcxHx1
#                 C = C.view(r,self.output_channel,self.kernel_size,1).permute(1,0,3,2)#H:Nxrx1xW
#                 y = torch.nn.functional.conv2d(input = x,weight = N,bias = None,stride = [self.stride,1],padding = [self.padding,0])
#                 y = torch.nn.functional.conv2d(input = y,weight = C,bias = self.bias,stride = [1,self.stride],padding = [0,self.padding])

            

#         else:
#             y = self.conv2d(x)
#         self.output_size = y.size()
#         return y
    
#     @property
#     def ParamN(self):
#         if not self.only_stride_1 or self.stride==1:
#             return self.N
#         else:
#             return None
    
#     @property
#     def ParamC(self):
#         if not self.only_stride_1 or self.stride==1:
#             return self.C
#         else:
#             return None

#     @property
#     def ParamSigma(self):
#         if not self.only_stride_1 or self.stride==1:
#             return self.Sigma
#         else:
#             return None

def conv3x3(in_planes, out_planes, stride=1,SVD_only_stride_1 = False,decompose_type = 'channel'):
    "3x3 convolution with padding"
    return SVD_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding = 1,bias=False,SVD_only_stride_1=SVD_only_stride_1,decompose_type = decompose_type)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, droprate=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.drop = nn.Dropout(p=droprate) if droprate>0 else None
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SVD_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
    @property
    def Ns(self):
        N = []
        if self.conv1.ParamN is not None:
            N.append(self.conv1.ParamN)
        if self.conv2.ParamN is not None:
            N.append(self.conv2.ParamN)
        return N

    @property
    def Cs(self):
        C = []
        if self.conv1.ParamC is not None:
            C.append(self.conv1.ParamC)
        if self.conv2.ParamC is not None:
            C.append(self.conv2.ParamC)
        return C

    @property
    def Sigmas(self):
        Sigma = []
        if self.conv1.ParamSigma is not None:
            Sigma.append(self.conv1.ParamSigma)
        if self.conv2.ParamSigma is not None:
            Sigma.append(self.conv2.ParamSigma)
        return Sigma



class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, droprate=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        last_planes = 512*block.expansion

        self.conv1 = conv3x3(in_channels, 64)
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.bn_last(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.logits(x.view(x.size(0), -1))
        return x


class PreActResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10, droprate=0):
        super(PreActResNet_cifar, self).__init__()
        self.in_planes = 16
        last_planes = filters[2]*block.expansion

        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1, droprate=droprate)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2, droprate=droprate)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2, droprate=droprate)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        """

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, droprate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def no_tbn(self):
    #     for m in self.modules():
    #         # if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.features(x)
        out = F.relu(self.bn_last(out))
        out = F.avg_pool2d(out, 8)
        out = self.logits(out.view(out.size(0), -1))
        return out
    
    
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
# ResNet for Cifar10/100 or the dataset with image size 32x32

def ResNet20_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def ResNet56_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [9 , 9 , 9 ], [16, 32, 64], num_classes=out_dim)

def ResNet110_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def ResNet29_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBottleneck, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def ResNet164_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBottleneck, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def WideResNet_28_2_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

def WideResNet_28_2_drop_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim, droprate=0.3)

def WideResNet_28_10_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [160, 320, 640], num_classes=out_dim)

# ResNet for general purpose. Ex:ImageNet

def ResNet10(out_dim=10):
    return PreActResNet(PreActBlock, [1,1,1,1], num_classes=out_dim)

def ResNet18S(out_dim=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=out_dim, in_channels=1)

def ResNet18(out_dim=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=out_dim)

def ResNet34(out_dim=10):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes=out_dim)

def ResNet50(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=out_dim)

def ResNet101(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes=out_dim)

def ResNet152(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes=out_dim)