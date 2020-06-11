import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents
from tensorboardX import SummaryWriter
import utils.viz as viz
from dataloaders.wrapper import AppendName
import random
import copy


''' basically the default changed neednot be changed for
    1) Incremental weight addition
        For starters this needs to have two models
        1) final model which gets appended
        2) model for task A
        3) Within the main model we need to have filter indices for each task

    Final model needs to be in USV space

    Task A ---> Train model for A. Do post processing by decomposing each layer into UVS append the main model with these values
    

    regularizer needs to be added if the model needs to trained in the SVD space

    # validation split for CIFAR-100 and others: 0.15
    # validation split for MiniImageNet : 0.02
'''

def get_splits(train_dataset, val_dataset):
    # converting multi dataset into splits
    keys_ = list(train_dataset.keys())
    train_dataset_splits = {}
    val_dataset_splits = {}
    task_output_space = {}
    for idx_ , key_ in enumerate(keys_):
        train_dataset_splits[str(idx_ + 1)] = AppendName(train_dataset[key_], str(idx_ + 1))
        val_dataset_splits[str(idx_ + 1)] = AppendName(val_dataset[key_], str(idx_ + 1))
        task_output_space[str(idx_ + 1)] = train_dataset[key_].number_classes
    # import pdb; pdb.set_trace()
    return train_dataset_splits, val_dataset_splits, task_output_space



def run(args, rand_seed):
    print('Random seeed', rand_seed)
    if args.benchmark:
        print('benchamrked')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        random.seed(rand_seed)

    # sparse_wts = args.sparse_wt
    # for sparse_wt in sparse_wts:
    if args.single_tasks:
        args.exp_name = f'{args.exp_name}_single_tasks'    
    # args.sparse_wt = sparse_wt

    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)
    
    # 5 sequence tasks
    if args.dataset == 'multidataset':
        train_dataset_splits, val_dataset_splits, task_output_space = get_splits(train_dataset, val_dataset)
    
    else:

        if args.n_permutation>0:
            train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                args.n_permutation,
                                                                                remap_class=not args.no_class_remap)
            
        else:
            train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                            first_split_sz=args.first_split_size,
                                                                            other_split_sz=args.other_split_size,
                                                                            rand_split=args.rand_split,
                                                                            remap_class=not args.no_class_remap)
            

    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef, 'exp_name' : args.exp_name, 'nuclear_weight' : args.nuclear_weight, 'period':args.period, 'threshold_trp': args.threshold_trp,
                    'sparse_wt': args.sparse_wt, 'perp_wt': args.perp_wt, 'reg_type_svd': args.reg_type_svd, 'energy_sv':args.energy_sv, 'save_running_stats':args.save_running_stats,
                    'e_search' : args.e_search, 'sp_wt_search' : args.sp_wt_search, 'single_tasks':args.single_tasks, 'prev_sing':args.prev_sing, 'debug':args.debug, 'grow_network':args.grow_network, 'ind_models':args.ind_models, 'dataset':args.dataset}

    if args.ind_models:
        agent_config_lst = {}
        for key_, val in task_output_space.items():
            
            agent_config_lst[key_] = copy.deepcopy(agent_config)
            agent_config_lst[key_]['out_dim'] = {key_:val}
            agent_config_lst[key_]['ind_models'] = True
            # import pdb; pdb.set_trace()
            
        agents_lst = {}
        for key_, val in task_output_space.items():
            agents_lst[key_] = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config_lst[key_])
          
    else:
        print(args.nuclear_weight, args.threshold_trp)
        agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    # import pdb; pdb.set_trace()     
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)

    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                                    
        test_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, test_loader)

        acc_table['All'] = {}
        acc_table['All']['All'] = agent.validation(test_loader)

    else:  # Incremental learning
        save_dict = {}
        # adhering to the Advarsarial Continual learning paper
        if args.dataset  == 'miniImageNet':
            validation_split = 0.02
            
        else:
            validation_split = 0.15

        
        final_accs = OrderedDict()
        if args.loadmodel:
                agent.load_model(args.loadmodel)
                for i in range(len(task_names)):
                    train_name = task_names[i]           
                    test_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                            
                    final_accs[train_name] = agent.validation(test_loader, train_name).avg
                    print(f'/CumAcc/Task{train_name}, {final_accs[train_name]}, {i}')
        else:

            for i in range(len(task_names)):

                train_name = task_names[i]
                if args.ind_models:
                    agent = agents_lst[train_name]
        
                writer = SummaryWriter(log_dir="runs/" + agent.exp_name)
                # # import pdb; pdb.set_trace()
                #print('Final split for ImageNet', int(np.floor(validation_split * len(train_dataset_splits[train_name]))))
                # split = int(np.floor(validation_split * len(train_dataset_splits[train_name])))
                # train_split, val_split = torch.utils.data.random_split(train_dataset_splits[train_name], [len(train_dataset_splits[train_name]) - split, split])
                # train_dataset_splits[train_name] = train_split
                print('====================== Task Num', i + 1,'=======================')
                print('======================',train_name,'=======================')
                train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                test_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

                if args.incremental_class:
                    agent.add_valid_output_dim(task_output_space[train_name])

                # Learn
                # import pdb; pdb.set_trace()
                agent.learn_batch(train_loader, test_loader, task_name=train_name)

                # if single task skip this step:
                # Evaluate
                if not (args.single_tasks or args.ind_models):
                    final_accs = OrderedDict()
                    acc_table[train_name] = OrderedDict()
                    for j in range(i+1):
                        # import pdb; pdb.set_trace()
                        val_name = task_names[j]
                        print('validation split name:', val_name)
                        # import pdb; pdb.set_trace()
                        val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                        test_loader = torch.utils.data.DataLoader(val_data,
                                                                batch_size=args.batch_size, shuffle=False,
                                                                num_workers=args.workers)
                        acc_table[val_name][train_name] = agent.validation(test_loader, val_name)
                        final_accs[val_name] = acc_table[val_name][train_name].avg
                        print(f'/CumAcc/Task{val_name}, {acc_table[val_name][train_name].avg}, {i}')
                        writer.add_scalar('/CumAcc/Task' + val_name, acc_table[val_name][train_name].avg, i)
                    
                    # writer.add_scalar('/CumLoss/Task' + val_name, loss_table[val_name][train_name].avg, i )
                elif args.single_tasks:
                    val_name = task_names[i]
                    final_accs[train_name] = agent.validation(test_loader, val_name).avg

                else:
                    val_name = task_names[i]
                    final_accs[train_name] = agent.validation(test_loader, val_name)[0].avg

            agent.save_model()

        print(final_accs)
        # collect the channels used, accuracies of individual  task, compression ratio of the final model for the
        # compute the size of the model
        avg_acc = sum(list(final_accs.values())) / len(final_accs)
        # import pdb; pdb.set_trace()
        model_size = agent.mode_comp(agent.chann_used) if not args.ind_models else len(agents_lst)*agent.mode_comp()
        save_dict['channels']  = agent.chann_used if not args.ind_models else []
        # import pdb; pdb.set_trace()
        save_dict['acc']  = final_accs
        save_dict['avg_acc'] = avg_acc
        print('Average accuray is', avg_acc)
        print('Model size is', model_size)
        save_dict['all_rank'] = agent.all_rank if not args.ind_models else []
        save_dict['model_size'] = model_size


    return  save_dict, task_names

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--perp_wt', type=float, default=1.0, metavar='D',
                    help='orthogology restrain weight  (default: 1.0)')
    parser.add_argument('--sparse_wt', type=float, nargs="+", default=[0.01], 
                    help='weight decay (default: 0.001)') 
    parser.add_argument('-e_search','--e_search', type=float,  nargs="+", default=[3e-5, 3e-2],
                    help="search for prune energy")
    parser.add_argument('-sp_wt_search','--sp_wt_search', type=float,  nargs="+", default=[0.1, 0.9],
                    help="search for prune energy")
    parser.add_argument('-e','--energy_sv', type=float, default=3e-5,
                    help="the energy of the singular values that should be pruned")
    parser.add_argument('--reg_type_svd', type=str, default='Hoyer', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_type', type=str, default='customnet_SVD', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    
    parser.add_argument('--model_name', type=str, default='Net', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=0, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='SVDNet', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='SVDNet', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=5)
    parser.add_argument('--other_split_size', type=int, default=5)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--single_tasks', dest='single_tasks', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--prev_sing', dest='prev_sing', default=False, action='store_true',
                        help="Allow data augmentation during training")
    
    parser.add_argument('--grow_network', dest='grow_network', default=False, action='store_true',
                        help="Grow Network as we grow")
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--benchmark', dest='benchmark', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--save_running_stats', dest='save_running_stats', default=False, action='store_true',
                        help="Randomize the order of splits")
    #This is not currently supported yet
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=True, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=1, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=3, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    parser.add_argument('--exp_name', dest='exp_name', default='SVDNet_Test', type=str,
                        help="Exp name to be added to the suffix")
    parser.add_argument('--loadmodel', dest='loadmodel', default=None, type=str,
                        help="modelname")
    #args.
    parser.add_argument('--nuclear-weight',  type=float, default=0.001, help='The weight for nuclear norm regularization')
    parser.add_argument('--threshold_trp',  type=float, default=0.99, help='Threshold for pruning')
    parser.add_argument('--period', dest='period', type=int, default=1, help='set the period of TRP')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='set the seed')
    parser.add_argument('--ind_models', dest='ind_models', default=False, action='store_true',help='run individual models')
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    reg_sparse_wt = args.sparse_wt
    

    # The for loops over hyper-paramerters or repeats
    exp_name = args.exp_name
    
    for i, sparse_wt in enumerate(reg_sparse_wt):
        seed = 1
        avg_acc_lst = []
        final_acc_runs = {}
        task_chann_used = {}
        avg_acc = 0
        model_size = 0
        
        args.sparse_wt = sparse_wt
        print(f'==================== Sparse Weight {sparse_wt} ============')
        
        args.exp_name = f'{exp_name}_sparse_wt{args.sparse_wt}'
        # avg_final_acc[reg_coef] = np.zeros(args.repeat)
        # avg_final_acc = []
        sv_dir = f'results/{args.exp_name}'
        for r in range(args.repeat):
            import json
            
            if not os.path.exists(sv_dir):
                os.makedirs(sv_dir)
            
            # Run the experiment
            save_dict_run, task_names_run = run(args, seed)


            for key_, acc_run in save_dict_run['acc'].items():
                if key_ in final_acc_runs:
                    final_acc_runs[key_] = acc_run + final_acc_runs[key_]
                else:
                    final_acc_runs[key_] = acc_run


            for ind_, task_ in enumerate(task_names_run):
                if task_ in task_chann_used:
                    task_chann_used[task_] =  [sum(x) for x in zip(task_chann_used[task_], save_dict_run['channels'][ind_])]   
                else:
                    task_chann_used[task_] = save_dict_run['channels'][ind_]

            avg_acc_lst.append(save_dict_run['avg_acc'])
            avg_acc += save_dict_run['avg_acc']
            model_size += save_dict_run['model_size']

            # viz.chann_tasks(agent.chann_used, args.model_name, agent.all_rank, args.exp_name)

            with open(f'{sv_dir}/output_run{r}.json', 'w') as fp:
                json.dump(save_dict_run, fp)

            seed += 1

        # divide by the runs
        avg_acc = avg_acc / args.repeat
        model_size = model_size / args.repeat
        for key_, val_ in final_acc_runs.items():
            final_acc_runs[key_] = final_acc_runs[key_] / args.repeat

        
        for key_, val in task_chann_used.items():
            task_chann_used[key_] = [t/args.repeat for t in task_chann_used[key_]]
        
        import statistics
        save_dict_final = {}
        save_dict_final['std_acc'] = 0 #statistics.stdev(avg_acc_lst)
        save_dict_final['model_size'] = model_size
        save_dict_final['avg_acc'] = avg_acc
        save_dict_final['acc'] = final_acc_runs
        save_dict_final['chann'] = task_chann_used

        with open(f'{sv_dir}/output_final.json', 'w') as fp:
            json.dump(save_dict_final, fp)

        print(f'============== Average accuracy is {avg_acc}=====================')
        print(f'============== STD of accuracy is {save_dict_final["std_acc"]}=====================')

            # print(acc_table)

            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            # avg_acc_history = [0] * len(task_names)
            # for i in range(len(task_names)):
            #     train_name = task_names[i]
            #     cls_acc_sum = 0
            #     for j in range(i + 1):
            #         val_name = task_names[j]
            #         cls_acc_sum += acc_table[val_name][train_name].avg
            #     avg_acc_history[i] = cls_acc_sum / (i + 1)
            #     print('Task', train_name, 'average acc:', avg_acc_history[i])

            # Gather the final avg accuracy
    #         avg_final_acc[reg_coef][r] = avg_acc_history[-1]

    #         # Print the summary so far
    #         print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
    #         print('The regularization coefficient:', args.reg_coef)
    #         print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
    #         print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
    # for reg_coef,v in avg_final_acc.items():
    #     print('reg_coef:', reg_coef,'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
