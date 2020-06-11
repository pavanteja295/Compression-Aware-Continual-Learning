import torch
from utils.metric import compute_conv_output_size
from models.customnet_SVD import SVD_Conv2d
import torch.nn as nn

class AlexNet(torch.nn.Module):

    def __init__(self, str=None):
        super(AlexNet, self).__init__()

        self.ncha,size,_=3, 84, 84
        # self.taskcla=args.taskcla
        # self.latent_dim = args.latent_dim

        # if args.experiment == 'cifar100':
        # hiddens = [64, 128, 256, 1024, 1024, 512]

        # elif args.experiment == 'miniimagenet':
        hiddens = [64, 128, 256, 512, 512, 512]

            # ----------------------------------
        # elif args.experiment == 'multidatasets':
        #     hiddens = [64, 128, 256, 1024, 1024, 512]

        # else:
        #     raise NotImplementedError

        if str:
            conv_mod_ = SVD_Conv2d
        else:
            conv_mod_ = torch.nn.Conv2d
        # import pdb; pdb.set_trace()
        self.conv1= conv_mod_(self.ncha,hiddens[0],kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=conv_mod_(hiddens[0],hiddens[1],kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=conv_mod_(hiddens[1],hiddens[2],kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        # self.maxpool=torch.nn.MaxPool2d(2)
        # self.relu=torch.nn.ReLU()

        # self.drop1=torch.nn.Dropout(0.2)
        # self.drop2=torch.nn.Dropout(0.5)

        # self.fc1=torch.nn.Linear(hiddens[2]*s*s,hiddens[3])
        # self.fc2=torch.nn.Linear(hiddens[3],hiddens[4])
        self.last=torch.nn.Linear(hiddens[4],hiddens[5])

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x_s):
        
        x_s = x_s.view_as(x_s)
        # h = self.features(x_s)
        # import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        h = self.maxpool(self.drop1(self.relu(self.conv1(x_s))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        
        h = h.view(x_s.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        h = self.drop2(self.relu(self.fc3(h)))

        
        h = h.view(x_s.size(0), -1)

        h = self.logits((h))
        # import pdb; pdb.set_trace()
        return h

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


def AlexNet_SVD():
    return AlexNet(str='SVD')
def AlexNet_g():
    return AlexNet()