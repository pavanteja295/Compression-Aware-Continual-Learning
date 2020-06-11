import torch
import torch.nn as nn
from utils import Regularization
from .default import NormalNN
#change these import into dynamic later
# from models.resnet_SVD import SVD_Conv2d
from models.customnet_SVD import SVD_Conv2d
import numpy as np
from utils.metric import AverageMeter
import utils.viz as viz
import sys




def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

class SVDNet(NormalNN):
    """
		Factorized Resnet
    """
    def __init__(self, agent_config):
        super(SVDNet, self).__init__(agent_config)
        self.agent_config = agent_config
#        if self.agent_config['model_name'] == 'Net':

        # else:
        #     from models.resnet import SVD_Conv2d

        self.task_param = {}
        self.task_count = 0
        # change this for multi gpu support
        use_cuda =  True if agent_config['gpuid'][0] >= 0 and torch.cuda.is_available() else False
        self.device = torch.device("cuda" if use_cuda else 'cpu')
        self.DEBUG = False
        self.no_svd = False
        self.trigger = 0
        self.batchnorm_stats = {}
        self.sample_ind = 0
        self.chann_used = {}
        self.all_rank = []
        self.current_rank = 0
        # this is for customn Network only
        self.original_rank = self.get_ranks()
        self.task_names_task_count = {}
        self.dataset = agent_config['dataset']
    
    def learn_batch(self, train_loader, val_loader=None, task_name=None):
        
        # 1.Learn the parameters for current task
        print('Single Task Scenario')
        if self.agent_config['debug']:
            if self.agent_config['single_tasks']:
                    if self.sample_ind:
                        self.model = self.create_model()
                        self.cuda()
                    super(SVDNet, self).learn_batch(train_loader, val_loader)
                    # self.save_model(f'saves/{self.agent_config["exp_name"]}/unpruned_task_sparse_wt_{wt}_task_num{self.task_count}')
                    if self.agent_config['save_running_stats']:
                        self.batchnorm_stats[self.task_count] = self.save_bn_stats()

                    self.task_param[self.task_count] = self.arrange_model()
                    self.chann_used[self.sample_ind ] = self.task_param[self.task_count]
                    self.task_names_task_count[task_name] = self.task_count
                    self.task_count = 0
                    self.sample_ind += 1

            elif self.agent_config['sp_wt_search']:
                chan_lst = []
                acc_lst = []
                # train the model with the given sparsity and perp loss
                # prune it and see how much efficiency it has achieved

                sparse_wt =  np.linspace(self.agent_config['sp_wt_search'][0], self.agent_config['sp_wt_search'][1], 10)
                for wt_ind, wt in enumerate(sparse_wt):
                    self.agent_config['sparse_wt'] = wt
                    print('######################################################################')
                    print(f'Training with sparsity weight {wt}')
                    # initialize the model
                    self.model = self.create_model()

                    self.cuda()
                    super(SVDNet, self).learn_batch(train_loader, val_loader)
                    # self.save_model(f'saves/{self.agent_config["exp_name"]}/unpruned_task_sparse_wt_{wt}_task_num{self.task_count}')
                    if self.agent_config['save_running_stats']:
                        self.batchnorm_stats[self.task_count] = self.save_bn_stats()

                    self.task_param[self.task_count] = self.arrange_model()
                    self.chann_used[self.task_count] = self.task_param[self.task_count]
                    print(f'Ranks after training task {self.sample_ind} is {self.task_param[self.task_count]}')
                    id_ = 0
                    self.model = self.prepare_model_test(self.model, id_)
                    print(f'Validation error after rearrange for sp_weight {wt} and energy {self.agent_config["sparse_wt"]}')
                    acc, _, _, _ = super(SVDNet, self).validation(val_loader)
                    # import pdb; pdb.set_trace()
                    self.task_param[self.task_count]
                    chan_lst.append(np.array(self.task_param[self.task_count]).sum())
                    acc_lst.append(acc.avg)
                viz.chan_acc(chan_lst, acc_lst, sparse_wt, dir_=self.exp_name)
                sys.exit("Done searching") 

            elif self.agent_config['e_search'] is not None:
                acc_lst = []
                chan_lst = []
                
                if self.agent_config['model_weights'] is None:
                    sys.exit("Provide a model to search") 
                # this should load the model3e-5, 3e-2
                energy_ =  np.linspace(self.agent_config['e_search'][0], self.agent_config['e_search'][1] , 20)

                for en in energy_:
                    self.model = self.create_model()
                    self.cuda()
                    self.agent_config['energy_sv'] = en
                    self.task_param[self.task_count] = self.arrange_model()
                    print(f'Ranks after training task {self.sample_ind} is {self.task_param[self.task_count]}')
                    id_ = 0
                    self.model = self.prepare_model_test(self.model, id_)
                    print(f'Pruning energy {en}')
                    #print(f'Validation error after rearrange for sp_weight {wt} and energy {self.agent_config["sparse_wt"]}')
                    acc, _, _, _ = super(SVDNet, self).validation(val_loader)
                    acc_lst.append(acc.avg)
                    chan_lst.append(np.array(self.task_param[self.task_count]).sum())
                
                viz.chan_acc(chan_lst, acc_lst, energy_, dir_=self.exp_name, sp_wt=False)


        else:

            super(SVDNet, self).learn_batch(train_loader, val_loader)
            if self.agent_config['save_running_stats']:
                self.batchnorm_stats[self.task_count] = self.save_bn_stats()
            
            # import pdb; pdb.set_trace() 
            # # if self.task_count:
#            self.save_model(f'saves/{self.agent_config["exp_name"]}/unpruned_task_{self.task_count}')
            
            self.task_param[self.task_count] = self.arrange_model()

            # if self.task_count:
            #     import pdb; pdb.set_trace()

            self.chann_used[self.task_count] = (np.array(self.task_param[self.task_count]) - np.array(self.task_param[self.task_count - 1])).tolist() \
                                               if self.task_count else self.task_param[self.task_count]
            
            self.task_names_task_count[task_name] = self.task_count
            self.task_count += 1

    def get_ranks(self):
        rnk_ = []
        for m in self.model.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                rnk_.append(m.r)

        return rnk_

    def load_model(self, loadmodel):
        # import pdb; pdb.set_trace()
        self.current_rank = torch.load(f'saves/{loadmodel}/rank.pth')
        if self.agent_config['save_running_stats']:
            self.batchnorm_stats = torch.load(f'saves/{loadmodel}/batchnorm.pth')
        self.task_names_task_count = torch.load(f'saves/{loadmodel}/task_name_count.pth')
        self.task_param = torch.load( f'saves/{loadmodel}/task_identifiers.pth')
        self.chann_used = torch.load(f'saves/{loadmodel}/chann_used.pth')
        self.model = self.create_model(rank=self.current_rank)
        model_param = torch.load(f'saves/{loadmodel}/final_model.pth')
        self.model.load_state_dict(model_param)
        self.model.cuda()
        return 0

    def save_model(self):
        # import pdb; pdb.set_trace()
        super(SVDNet, self).save_model(f'saves/{self.agent_config["exp_name"]}/final_model')
        torch.save(self.task_param, f'saves/{self.agent_config["exp_name"]}/task_identifiers.pth')
        if self.agent_config['save_running_stats']:
            torch.save(self.batchnorm_stats, f'saves/{self.agent_config["exp_name"]}/batchnorm.pth')
        torch.save(self.current_rank, f'saves/{self.agent_config["exp_name"]}/rank.pth')
        torch.save(self.task_names_task_count, f'saves/{self.agent_config["exp_name"]}/task_name_count.pth')
        torch.save(self.chann_used, f'saves/{self.agent_config["exp_name"]}/chann_used.pth')
        print('========== model saved ==========')
        


    def save_bn_stats(self):
        bn_lst_ = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                    bn_lst_.append([m.running_var.clone(), m.running_mean.clone(), m.num_batches_tracked.clone()])
        return bn_lst_

    def update_model(self,  inputs, targets, tasks, i):
        out = self.forward(inputs)
        loss, orth_loss, sp_loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        ind_ = 0
        if not self.no_svd:
            if self.task_count:
                for m in self.model.modules():
                    if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                        m.N.grad[:, :self.task_param[self.task_count - 1][ind_]] = 0
                        m.C.grad[:self.task_param[self.task_count - 1][ind_], :] = 0
                        m.Sigma.grad[:self.task_param[self.task_count - 1][ind_]] = 0
                        ind_ = ind_ + 1

                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.grad[:] = 0
                        m.bias.grad[:] = 0

        self.optimizer.step()
        return loss.detach(), out, orth_loss, sp_loss

    def criterion(self, inputs, targets, tasks):
        loss, _, _ = super(SVDNet, self).criterion(inputs, targets, tasks)

        if self.no_svd:
            return loss, 0, 0

        perp_loss = 0

        for N in self.model.GetNs():
            perp_loss+=Regularization.orthogology_loss(N,self.device)
        
        
        for C in self.model.GetCs():
            perp_loss+=Regularization.orthogology_loss(C,self.device)
        
        
        reg_loss=Regularization.Reg_Loss(self.model.GetSigmas(),self.agent_config['reg_type_svd'])

        loss = loss + reg_loss * self.agent_config['sparse_wt']+perp_loss*self.agent_config['perp_wt']

        return loss, perp_loss, reg_loss

    def prepare_model_test(self, model_temp, id_):
        idx_ = 0
        bn_idx_ = 0
        for m in model_temp.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                valid_idx = self.task_param[id_][idx_]
                rem_rank = m.Sigma.size(0) - valid_idx
                m.N.data = torch.cat([m.N[:,:valid_idx].cpu(), torch.zeros(m.output_channel,rem_rank)], dim = 1).cuda()
                m.C.data = torch.cat([m.C[:valid_idx,:].cpu(), torch.zeros(rem_rank, m.C.shape[1])], dim=0).cuda()
                m.Sigma.data = torch.cat([m.Sigma[:valid_idx].cpu(), torch.zeros(rem_rank)]).cuda()
                idx_ += 1

            if self.agent_config['save_running_stats'] and isinstance(m, nn.BatchNorm2d):
                m.running_var, m.running_mean, m.num_batches_tracked = self.batchnorm_stats[id_][bn_idx_]
                bn_idx_ += 1
            
            else:
                pass

        return model_temp

    def validation(self, val_loader, task_id=''):
        if task_id != '':
            # if task_id != '1':
            #     return AverageMeter()
            # import pdb; pdb.set_trace()
            temp_model = self.create_model(rank=self.current_rank )
            temp_model.load_state_dict(self.model.state_dict())

            id_ = self.task_names_task_count[task_id]
            self.model.load_state_dict(self.prepare_model_test(self.model, id_).state_dict())
 
            print('Validation error after rearrange')
            acc, _, _, _ = super(SVDNet, self).validation(val_loader)

            # back again to the model
            self.model = self.create_model(rank=self.current_rank)
            self.model.load_state_dict(temp_model.state_dict())
            self.model = self.model.cuda()
            return  acc
        
        else:
            return super(SVDNet, self).validation(val_loader)

    def arrange_model(self):

        idx_ = []
        ind_ = 0
        total_energy = 0
        self.current_rank = []
        #remove the code with xx
        for m in self.model.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                # out of the already trained sigmas put aside the sigmas that are already good for before tasks
                # ideally nothing is used here for pruning
                if self.task_count:
                    slice_ = self.task_param[self.task_count - 1][ind_]
                else:
                    slice_ = 0
                # xxx
                # if ind_:
                #     self.task_count = m.Sigma.size(0)
                tensor = m.Sigma[slice_:].data.cpu().numpy()
                #Potential to develop
                # if agent_config['prev_sing']:
                #     current_energy = m.Sigma[slice_:].data.cpu().numpy()
                #     total_energy += np.sum(m.Sigma.data.cpu().numpy())
                # else:
                #     current_energy = 0.0
                #     total_energy += np.sum(np.square(tensor))
                if self.agent_config['energy_sv'] is not None:
                    energy_sort_singular,_ = torch.sort((m.Sigma.data[slice_:])**2, descending=True)
                    if self.agent_config['prev_sing']:
                        # everything till now
                        # if self.task_count:
                        #     import pdb; pdb.set_trace()
                        current_total_energy = torch.sum((m.Sigma.data)**2)
                        current_energy = torch.sum((m.Sigma.data[:slice_])**2)
                    else:
                        current_energy = 0.0
                        current_total_energy = torch.sum(energy_sort_singular)
                    idx = 0
                    while current_energy/current_total_energy < 1-self.agent_config['energy_sv']:
                        current_energy+=energy_sort_singular[idx]
                        idx+=1
                    if idx:
                        threshold = float(torch.sqrt(energy_sort_singular[idx-1]))
                    else:
                        threshold = 0
                else:
                    threshold = self.agent_config['sensitivity']

                new_mask = np.where(abs(tensor) < threshold, 0, tensor)
                checl_mask = np.sum( new_mask != new_mask)
                if checl_mask:
                    import pdb; pdb.set_trace()

                nm_t = torch.Tensor(new_mask)
                valid_idx_t = torch.arange(nm_t.size(0))[nm_t!=0]
                valid_idx = torch.cat([torch.arange(slice_) , valid_idx_t + slice_]) 

                
                idx_.append(valid_idx.size(0))
                

                nm_t = m.Sigma
                rem_rank = nm_t.size(0) - len(valid_idx)
                
                print(f'Remaining rank before addint {rem_rank}')

                if self.config['grow_network'] and self.task_count:
                    # import pdb; pdb.set_trace()
                    # add if the network has zero space
                    avg_space = self.get_avg()
                    if rem_rank - avg_space[ind_] < 0:
                        # add some space here
                        # import pdb; pdb.set_trace()
                        rem_rank = int(avg_space[ind_])

                print(f'Remaining rank adding {rem_rank}')

                self.current_rank.append(rem_rank + len(valid_idx))

                if not rem_rank:
                    f_ = lambda x : x
                    f_s = f_
                else:
                    f_ = torch.nn.init.kaiming_normal_
                    f_s = torch.nn.init.normal_

                # print(m.Sigma[valid_idx])
                # if self.task_count:
                #     import pdb; pdb.set_trace()
                temp_N = f_(torch.empty(m.output_channel,rem_rank))
                temp_C = f_(torch.empty(rem_rank, m.C.shape[1]))
                temp_Sigma = f_s(torch.empty(rem_rank))
                m.N.data = torch.cat([m.N[:,valid_idx].cpu(), temp_N.clone()], dim = 1).cuda()
                m.C.data = torch.cat([m.C[valid_idx,:].cpu(), temp_C.clone()], dim=0).cuda()
                m.Sigma.data = torch.cat([m.Sigma[valid_idx].cpu(), temp_Sigma.clone()]).cuda()
                ind_ += 1
        
        # import pdb; pdb.set_trace()
        print(f'Ranks after training task {self.task_count} is {idx_}')
        return idx_

    def get_avg(self):

        np_tsk_all = np.zeros_like(self.task_param[0])
#        if self.agent_config['grow_metric'] == 'avg':
        for key_, tsk_ in self.task_param.items():
            np_tsk_ = np.array(tsk_)
            np_tsk_all = np_tsk_all + np_tsk_
        
        np_tsk_all = np_tsk_all / len(list(self.task_param.keys()))
        # if self.agent_config['grow_metric'] == 'min':
        #     for key_, tsk_ in self.task_param.items():
        #         np_tsk_ = np.array(tsk_)
        #         np_tsk_all = np_tsk_all + np_tsk_
            
            # np_tsk_all = np_tsk_all / len(list(self.task_param.keys()))

        # np_curr = self.original_rank
        # import pdb; pdb.set_trace()
        # some times increases exponentially so a check to make sure it stays upto rank of the modle
        if self.dataset == 'CIFAR100' or self.dataset == 'miniImageNet': # related tasks reduces size by 1MB
            return np.minimum(np_tsk_all, self.original_rank)    # #
        else:    
            return self.original_rank 

    def mode_comp(self, chann_used):
        cum_chann_used = [ sum(t)  for t in zip(*chann_used.values())]
        new_params = 0
        idx_ = 0
        ll_ = 0
        for m in self.model.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                tot_rank = int((m.input_channel*m.kernel_size*m.kernel_size*m.output_channel) / (m.output_channel + (m.input_channel*m.kernel_size*m.kernel_size)) )
                self.all_rank.append(tot_rank)
                #old_FLOPS += (m.input_channel*m.kernel_size*m.kernel_size*tot_rank) + (m.output_channel*tot_rank)
                new_params += (m.input_channel*m.kernel_size*m.kernel_size*cum_chann_used[idx_] ) + (m.output_channel * cum_chann_used[idx_]) + (cum_chann_used[idx_])
                idx_  = idx_ + 1

        
            if isinstance(m, nn.Linear):
                print(f'Linear layer', ll_)
                new_params += (m.weight.shape[0] * m.weight.shape[1]) + m.bias.shape[0]

        
            if isinstance(m, nn.BatchNorm2d):
                print('BatchNorm')
                new_params += m.weight.shape[0] *4*20

        # add size taken for task identifier
        new_params += self.task_count * idx_
        # conver params into MB (1 paaram = 4 bytes)
        model_size = (new_params * 4 ) / 1000000
        print(model_size)

        
        # # import pdb; pdb.set_trace()
        # print('Original params of the network', old_FLOPS)
        return model_size

                # out of the already trained sigmas put aside the sigmas that are already good for before tasks
                # ideally nothing is used here for pruning
