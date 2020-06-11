import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

def chan_acc(lchan, lacc, lwt, dir_='default', sp_wt=True):	

	#import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import host_subplot
	import mpl_toolkits.axisartist as AA

	host = host_subplot(111, axes_class=AA.Axes)
	plt.subplots_adjust(bottom=0.2)

	#Add twin y axis with offset from bottom and hide top axis
	par = host.twiny()
	offset = -30

	new_fixed_axis = host.get_grid_helper().new_fixed_axis

	par.axis["bottom"] = new_fixed_axis(loc="bottom",
                                    axes=par,
                                    offset=(0, offset))
	par.axis["top"].set_visible(False)

	plt.plot(lwt, lacc)
	host.set_xlabel("Weight")
	host.set_xticks(lwt.tolist())
	#host.set_xlim(min(lwt), max(lwt))


	par.set_xlabel("Channels")
	par.set_xticks(lchann)
	par.set_xlim(min(lchan), max(lchan))

	# plt.subplot(2,1,1)

	# lwt_str = [str(i) for i in lwt]
	# dict_f = dict(zip(lwt_str, lchan))
	# dict_i = dict(zip( lchan, lwt_str))
	

	# def deg2rad(X):
	# 	return dict_f[str(X)]

	# def rad2deg(X):
	# 	return dict_i[X]

	# fig=plt.figure()
	# ax=fig.add_subplot(111, label="1")
	# # ax2 = ax.twiny()
	# # ax2=fig.add_subplot(111, label="2", frame_on=False)

	# # make_labels = [f'{wt}w/{chan}c' for chan, wt in zip(lchan, lwt)]
	# ax.plot(lwt.tolist(), lacc, color="C0")
	# # ax.set_xlabel("Num Channels", color="C0")
	# ax.set_ylabel("Accuracy", color="C0")
	# ax.tick_params(axis='x', colors="C0")
	# ax.tick_params(axis='y', colors="C0")
	# ax.set_xticks(lwt.tolist())
	# import pdb; pdb.set_trace()
	# secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))
	# secax.set_label('Num Channels')
	# ax.set_xticklabels(lwt.tolist())
	
	# import pdb; pdb.set_trace()
	# # ax2.plot(lwt, lacc, color="C1")
	# # ax2.xaxis.tick_top()
	# # ax2.yaxis.tick_right()
	# if sp_wt:
	# 	ax.set_xlabel('Sparse Weight', color="C0") 
	# else:
	# 	ax.set_xlabel('Energy', color="C0") 

	# ax2.set_ylabel('Accuracy', color="C1")
	# ax2.xaxis.set_label_position('top') 
	# ax2.yaxis.set_label_position('right') 
	# ax2.tick_params(axis='x', colors="C1")
	# ax2.tick_params(axis='y', colors="C1")
	# ax2.xticks(lwt, make_labels)

	# remove this and add one location to create all dirs
	sv_dir = f'graphs/{dir_}'
	if not os.path.exists(sv_dir):
		os.makedirs(sv_dir)
	if sp_wt:
		plt.savefig(os.path.join(sv_dir, 'accuracy_channels.png'))
	else:
		plt.savefig(os.path.join(sv_dir, 'accuracy_channels_energy.png'))
	plt.close()

def chann_tasks(rank_lst, net_name, all_rank, dir_):
	ind = np.arange(len(all_rank))
	cum_chann_used = [ sum(t)  for t in zip(*rank_lst.values())]
	rem_ = (np.array(all_rank) - np.array(cum_chann_used)).tolist()

	plts_ = {}
	plts_lst = []
	keys_lst = []
	# import pdb; pdb.set_trace()
	# sort the keys
	tsk_lst_ = list(rank_lst.keys())
	print(tsk_lst_)
	tsk_lst_.sort()
	for key_ in tsk_lst_:
		val_ = rank_lst[key_]
		# import pdb; pdb.set_trace()
		if not key_:
			plts_[key_] = plt.bar(ind, rank_lst[key_], width=0.35)
		else:
			plts_[key_] = plt.bar(ind, rank_lst[key_], bottom=prev_hts, width=0.35)
		plts_lst.append(plts_[key_])
		keys_lst.append(f'Task {key_}')
		prev_hts =  (np.array(prev_hts) + np.array(rank_lst[key_])).tolist() if key_ else np.array(rank_lst[key_])
		# import pdb; pdb.set_trace()
	plts_rem = plt.bar(ind, rem_, bottom=rank_lst[len(rank_lst.keys()) -1], width=0.35)
	plts_lst.append(plts_rem)
	keys_lst.append(f'REMAINING')

	plt.title(net_name)
	plt.legend(plts_lst, keys_lst)
	plt.xticks()
	plt.subplots_adjust(left=0.05, bottom=0.36, right=1.0, top=0.96)
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(18.5, 10.5)

	#import pdb; pdb.set_trace()

	plt.xticks(ind, all_rank, rotation=90)
	plt.ylabel('Channels consumed')

	sv_dir = f'graphs/{dir_}'
	if not os.path.exists(sv_dir):
		os.makedirs(sv_dir)

	plt.savefig(os.path.join(sv_dir, 'Channels_Tasks_allocation.png'))
	plt.close()
	return