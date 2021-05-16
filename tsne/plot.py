import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import cycle

# Hyper-parameters
run='B800'
set='train'            # 'train' or 'val'
perp_s=[10,20,30,40]   # Between 5 and 50 is good (~expected close neighbours)
lr_s=[100,200]	       # Between 10 and 1000
n_iter=50000           # Should be at least 250
init='pca'             # 'pca' or ‘random’
sampler_seed_range=1
tsne_seed_range=1

colors=cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# Load of original pointclouds ids, to correctly identify
pc_ids=np.loadtxt(set+'.txt',dtype=np.int16)

# Load embeddings and corresponding labels (point cloud idx) for whole dataset
# Each label identifies the point cloud 4 embedding originate from
# So, all_embeds[0:4,:] are all embeddings of the 'all_labels[0]'-th point cloud 
all_embeds=np.load('data/embeds_'+run+'_'+set+'.npy')
all_labels=np.load('data/labels_'+run+'_'+set+'.npy')

# Sample 10 random pointcloud labels
# Don't include pc labels that have few embeddings generated
unique_labels,unique_counts = np.unique(all_labels,return_counts=True)
unique_labels=unique_labels[unique_counts==np.max(unique_counts)]
j=0

for sampler_seed in range(sampler_seed_range):

	np.random.seed(sampler_seed)
	chosen_labels = np.random.choice(unique_labels,10)

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

	for tsne_seed in range(tsne_seed_range):
		for perplexity in perp_s:
			for lr in lr_s:

				# Run TSNE
				print('-------------------- Running t-SNE nr',j,'--------------------')
				tsne=TSNE(n_components=2,perplexity=perplexity,learning_rate=lr,n_iter=n_iter,init=init,random_state=tsne_seed,verbose=1)
				embeds2D=tsne.fit_transform(embeds)
				j+=1

				# Plot 2D embeddings
				# Each color represents a different pointcloud label
				legend=[]

				for label in chosen_labels:
					
					color=next(colors)
					legend.append(Line2D([0], [0], linestyle='', marker='o', color=color, label=str(pc_ids[label])))
					
					for label_idx in np.where(labels == label)[0]:
						
						#print('label',label,'found at',label_idx)
						data_idx = 4*label_idx
						volx_idx = data_idx + 2
						
						# plot 2 data embeddings + 2 voxel embeddings
						plt.scatter(embeds2D[data_idx:data_idx+2,0],embeds2D[data_idx:data_idx+2,1],c=color,marker='h')
						plt.scatter(embeds2D[volx_idx:volx_idx+2,0],embeds2D[volx_idx:volx_idx+2,1],c=color,marker='X')

				plt.legend(handles=legend,handlelength=0.5,labelspacing=0.1,fontsize=10)
				plt.axis('off')

				filename='{}_{}_{}_{}_{}_{}_{}'.format(run,set,init,perplexity,lr,sampler_seed,tsne_seed)
				plt.savefig('figures/'+filename+'.png', dpi=300, bbox_inches='tight',pad_inches = 0)
				plt.clf()

				#plt.show()
