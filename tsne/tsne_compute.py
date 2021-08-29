import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import cycle
import colorsys
from matplotlib.legend_handler import HandlerTuple

# Hyper-parameters
run_s=['A50','B50','C50','A200','B200','C200','A350','B350','C350','A500','B500','C500','A650','B650','C650','A800','B800','C800'] #['B10','B200','B500','B800']
#run_s=['A800','B800','C800']
#run_s=['B800']
set_s=['train']            # 'train' or 'val'
init='pca'             # 'pca' or ‘random’
perp_s=[20]   # Between 5 and 50 is good (~expected close neighbours)
lr_s=[100]	       # Between 10 and 1000
n_iter=50000           # Should be at least 250. Put 50000
sampler_seed_s=[2] # 2 for train # 5 for val
tsne_seed_s=[0]
override_choice=None
#override_choice=[7245,4218,1392,4901,4057,808,1676,1020,3541,2542] # train best case

#plt.rcParams['axes.facecolor'] = 'none'
colors=cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

if override_choice:
	sampler_seed_s = ['custom']

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

# Counter for the number of plots generated
plotnr=1
totalnr=len(run_s)*len(set_s)*len(perp_s)*len(lr_s)*len(sampler_seed_s)*len(tsne_seed_s)

for run in run_s:
	for set in set_s:	

		# Load of original pointclouds ids, to correctly identify
		pc_ids=np.loadtxt('kitti_tsne_'+set+'_ids.txt',dtype=np.int16)

		# Load embeddings and corresponding labels (point cloud idx) for whole dataset
		# Each label identifies the point cloud 4 embedding originate from
		# So, all_embeds[0:4,:] are all embeddings of the 'all_labels[0]'-th point cloud 
		all_embeds=np.load('data/embeds_'+run+'_'+set+'.npy')
		all_labels=np.load('data/labels_'+run+'_'+set+'.npy')
		all_labels=pc_ids[all_labels]

		# Sample 10 random pointcloud labels
		# Don't include pc labels that have few embeddings generated
		unique_labels,unique_counts = np.unique(all_labels,return_counts=True)
		unique_labels=unique_labels[unique_counts==np.max(unique_counts)]

		for sampler_seed in sampler_seed_s:
			
			if override_choice:
				chosen_labels = np.array(override_choice)
			else:
				np.random.seed(sampler_seed)
				chosen_labels = np.random.choice(unique_labels,10,replace=False)
			chosen_labels = np.sort(chosen_labels)

			# Store all embeddings from the chosen pointclouds
			i=0
			embeds=[]
			labels=[]

			for label in all_labels:
				if np.isin(label,chosen_labels):
					labels.append(label)
					embeds.append(all_embeds[4*i:4*i+4,:])
				i+=1

			labels=np.array(labels)
			embeds=np.array(embeds).reshape(-1,128)

			for tsne_seed in tsne_seed_s:
				for perplexity in perp_s:
					for lr in lr_s:

						# Run TSNE
						print('-------------------- Running t-SNE nr',plotnr,'/',totalnr,'--------------------')
						tsne=TSNE(n_components=2,perplexity=perplexity,learning_rate=lr,n_iter=n_iter,init=init,random_state=tsne_seed,verbose=1)
						embeds2D=tsne.fit_transform(embeds)
						plotnr+=1

						filename='{}_{}'.format(run,set)
						np.save('tsne_embeds/'+filename, embeds2D)
						np.save('tsne_embeds/chosen_labels_'+filename, chosen_labels)
						np.save('tsne_embeds/labels_'+filename, labels)