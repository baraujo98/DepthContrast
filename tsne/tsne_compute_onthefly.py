import numpy as np
from sklearn.manifold import TSNE
import time
import glob
import os

inputpath ="/home/baraujo/DepthContrast/tsne/data/animation/"
outputpath="/home/baraujo/DepthContrast/tsne/tsne_embeds/"


def computetsne(filepath,filename):

	# Load of original pointclouds ids, to correctly identify
	#pc_ids=np.loadtxt('kitti_tsne_'+set+'_ids.txt',dtype=np.int16)

	# Load embeddings and corresponding labels (point cloud idx) for whole dataset
	# Each label identifies the point cloud 4 embedding originate from
	# So, all_embeds[0:4,:] are all embeddings of the 'all_labels[0]'-th point cloud 
	embeds=np.load(filepath)
	labels=np.load(filepath.replace('embeds','labels'))
	#labels=pc_ids[labels]
	#chosen_labels = np.unique(labels)

	# Run TSNE
	tsne=TSNE(n_components=2,perplexity=20,learning_rate=100,n_iter=50000,init='pca',random_state=0,verbose=3)
	embeds2D=tsne.fit_transform(embeds)

	np.save(outputpath+filename, embeds2D)
	np.save(outputpath+filename.replace('embeds','labels'), labels)
	#np.save(outputpath+filepath.replace('embeds','chosen_labels'), chosen_labels)


done=[]

while True:

	filepaths=glob.glob(inputpath+"/labels*")

	for filepath in filepaths:
		if filepath in done:
			continue
		filepath = filepath.replace("labels","embeds")
		filename = os.path.basename(filepath)
		print('Computing ->',len(done)+1,'/',len(filepaths))
		computetsne(filepath,filename)
		done.append(filename)
	
	print('Going to sleep')
	time.sleep(100)
