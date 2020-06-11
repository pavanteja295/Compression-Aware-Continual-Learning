import torch
import random
import models
from .default import NormalNN
from utils.decompose import get_look_up_table, get_look_up_table_full, low_rank_approx, channel_decompose, \
    EnergyThreshold, ValueThreshold, LinearRate

class Combined_Model():
	def __init__(self, agent_config):
		# define the main model here
		# define the temporary model each time learn_batch is called for the first time
		# also send in the nuclear norm or not option here
		self.agent_config = agent_config
		self.rank = 0
		self.task_id = 0
		self.singular_vectors = None
		self.threshold = self.agent_config['threshold_trp'] # this should be a parameter
		self.batch_norm_vecs = {}
		self.lin_vecs = {}

	def learn_batch(self,train_loader, val_loader=None):
		# create a new model here
		# import pdb; pdb.set_trace()
		# import pdb; pdb.set_trace()
		self.tmp_nn = Base_Model(self.agent_config)
		self.tmp_nn.learn_batch(train_loader, val_loader)

		if not self.task_id:
			
			self.look_up_table, self.look_up_table_bn, self.look_up_table_lin  = get_look_up_table_full(self.tmp_nn.model)
			print('Validation error after Decomposition', self.tmp_nn.validation(val_loader))
			self.singular_value_r = {key_ : [] for key_ in self.look_up_table}
			self.singular_vectors_N = {key_ : [] for key_ in self.look_up_table}
			self.singular_vectors_C = {key_ : [] for key_ in self.look_up_table}
			self.singular_vectors_N, self.singular_vectors_C, tmp_singular_value_r, self.tmp_nn.model, self.params_ =  channel_decompose(self.tmp_nn.model, self.look_up_table, \
																							criterion=EnergyThreshold(self.threshold), stride_1_only=False)
			self.singular_value_r = { name : [tmp_singular_value_r[name]] for name in tmp_singular_value_r.keys()}
		else:
			tmp_singular_vectors_N, tmp_singular_vectors_C, tmp_singular_value_r, self.tmp_nn.model, self.params_  = channel_decompose(self.tmp_nn.model, self.look_up_table, \
					criterion=EnergyThreshold(self.threshold), stride_1_only=False)
			self.singular_vectors_N  = { name :  torch.cat( [self.singular_vectors_N[name], tmp_singular_vectors_N[name] ], 1)  for name in self.singular_vectors_N.keys() }
			self.singular_vectors_C  = { name :  torch.cat( [self.singular_vectors_C[name], tmp_singular_vectors_C[name] ], 0)  for name in self.singular_vectors_C.keys() }
			#self.singular_value_r =    { name:   self.singular_value_r[name].append(tmp_singular_value_r[name]) if type(self.singular_value_r[name]) == type([]) else [self.singular_value_r[name], tmp_singular_value_r[name]]  for name in self.singular_value_r.keys()}
			for name in self.singular_value_r.keys():
				self.singular_value_r[name].append(tmp_singular_value_r[name])
		
		print('Params init are :', self.params_[1]  )
		print('Params new are :', self.params_[0]  )
		# store the batch norm for now other intelligent techniques can be implemented later
		# params for batch_norm
		# batch_norm_params = ['track_running_stats', ]
		self.bn_dict = {'weight' : 0, 'bias' : 1, 'running_mean' : 2, 'running_var' : 3, 'num_batches_tracked' : 4 }

		for name, m  in self.tmp_nn.model.named_modules():
			if name in self.look_up_table_bn:
				if not self.task_id:
#					self.batch_norm_vecs[name] = [m.weight.view(-1, 1), m.bias.view(-1, 1), m.running_mean.view(-1, 1), m.running_var.view(-1, 1), [m.num_batches_tracked] ]
					self.batch_norm_vecs[name] = [m.weight, m.bias, m.running_mean, m.running_var] #, [m.num_batches_tracked] ]

				else:
					# self.batch_norm_vecs[name]  =  [torch.cat( [self.batch_norm_vecs[name][0], m.weight.view(-1, 1)], 0), torch.cat( [self.batch_norm_vecs[name][1], m.bias.view(-1, 1)], 1), \
					# 								torch.cat( [self.batch_norm_vecs[name][2], m.running_mean.view(-1, 1)], 1), torch.cat( [self.batch_norm_vecs[name][3], m.running_var.view(-1, 1)], 1), \
					# 								self.batch_norm_vecs[name][4].append(m.num_batches_tracked)]
					self.batch_norm_vecs[name]  =  [torch.cat( [self.batch_norm_vecs[name][0], m.weight], 0), torch.cat( [self.batch_norm_vecs[name][1], m.bias], 0), \
													torch.cat( [self.batch_norm_vecs[name][2], m.running_mean], 0), torch.cat( [self.batch_norm_vecs[name][3], m.running_var], 0), \
													]#self.batch_norm_vecs[name][4].append(m.num_batches_tracked)]
	
		for name, m  in self.tmp_nn.model.named_modules():
			if name in self.look_up_table_lin:
				if not self.task_id:
					self.lin_vecs[name] = [m.weight, m.bias]

				else:
					self.lin_vecs[name]  =  [torch.cat( [self.lin_vecs[name][0], m.weight], 0), torch.cat( [self.lin_vecs[name][1], m.bias], 0) ]



		# batch norm for now can be ignore I guess
		self.task_id += 1

		
	# def save_model(self, )
	def validation(self, val_loader, task_id):
		# use task_id
		bm_ = Base_Model(self.agent_config)
		bm_.model = self.construct_model(task_id, bm_.model)
		return bm_.validation(val_loader)
		

	def construct_model(self, task_id, model):
		#model = models.__dict__[self.agent_config['model_type']].__dict__[self.agent_config['model_name']]()
		
		dict2 = model.state_dict()
		id_ = int(task_id) - 1
		for name in dict2:
			if any(t_conv in name for t_conv in self.look_up_table):
				dim_ = dict2[name].shape
				model_name = name[:-7]
				param = dict2[name]
				
				tmp_indx = self.singular_value_r[model_name][id_]
				strt_indx = sum(self.singular_value_r[model_name][:id_])
				last_indx = strt_indx + tmp_indx
				tmp_sing_N  = self.singular_vectors_N[model_name][:, strt_indx:last_indx]
				tmp_sing_C  = self.singular_vectors_C[model_name][ strt_indx:last_indx, :]
				new_NC = tmp_sing_N @ tmp_sing_C
				new_NC = new_NC.contiguous().view(dim_[0], dim_[1], dim_[2], dim_[3])
				dict2[name].copy_(new_NC)
				
				
			elif any(t_bn in name for t_bn in self.look_up_table_bn):
				param_id = self.bn_dict[name[name.rfind('.')+1:]]
				layer_name_ = name[:name.rfind('.')]
				if name[name.rfind('.')+1:]  ==  'num_batches_tracked':
					pass
					# try:
					# 	dict2[name].copy_(self.batch_norm_vecs[layer_name_][param_id][id_])
					# except:
					# 	import pdb; pdb.set_trace()
				else:
					dim_ = dict2[name].shape
					dict2[name].copy_(self.batch_norm_vecs[layer_name_][param_id][ dim_[0] * (id_) : dim_[0] * (id_ + 1)])
			
			elif any(t_bn in name for t_bn in self.look_up_table_lin):
				layer_name_ = name[:name.rfind('.')]
				param_  = name[name.rfind('.')+1:]

				if param_ == 'weight':
					dim_ = dict2[name].shape
					pass
					#dim_
					dict2[name].copy_(self.lin_vecs[layer_name_][0][dim_[0] * (id_) : dim_[0] * (id_ + 1) , : ])
				elif param_ == 'bias':
					#pass
					dict2[name].copy_(self.lin_vecs[layer_name_][1][dim_[0] * (id_) : dim_[0] * (id_ + 1) ])


		model.load_state_dict(dict2)
		return model
		# construct the model from stored models
		# 

		# once it is trained time to integrate this model to the main_model
		# call the  decompose layers

		# now update the final model
	# def save(self, )


class Base_Model(NormalNN):
		
	def __init__(self, agent_config):
		super(Base_Model, self).__init__(agent_config)
		# import pdb; pdb.set_trace()
		self.look_up_table = get_look_up_table(self.model)
		self.period = self.agent_config['period']
		self.threshold = agent_config['threshold_trp']
		 #may be create model might need to be a bit different


	def update_model(self, inputs, targets, tasks, batch_idx):
		if batch_idx % self.period == 0:
			print_ = False
			if batch_idx == 0:
				print_= False
	
			self.model, sub = low_rank_approx(self.model, self.look_up_table, criterion=EnergyThreshold(self.threshold), print_=print_)
		# import pdb; pdb.set_trace()
		out = self.forward(inputs)
		loss = self.criterion(out, targets, tasks)
		self.optimizer.zero_grad()
		loss.backward()
        
		if self.period is not None and batch_idx % self.period == 0 and batch_idx != 0:
			for name, m in self.model.named_modules():
				if name in self.look_up_table:
					m.weight.grad.data.add_(self.agent_config['nuclear_weight']*sub[name])
		self.optimizer.step()
		return loss.detach(), out
		
		
		








		

