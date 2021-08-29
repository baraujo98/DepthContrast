import numpy as np
from scipy.spatial.distance import pdist,squareform
import pandas as pd
from collections import defaultdict

run_s=['A50','A200','A500','A800','B50','B200','B500','B800','C50','C200','C500','C800'] #['C200','C500','C800']
set_s=['train','val']
metric_s=['euclidean']
#metric_s=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
k_s=[1,3,5,10,20,40,60]

# Counter for the number of metrics calculated
resulnr=1
totalnr=len(run_s)*len(set_s)*len(metric_s)*len(k_s)

df=defaultdict(list)

def append_df(df,set,run,k,metric,recalls):
    df['set'].append(set)
    df['run'].append(run[0])
    df['epoch'].append(run[1:])
    df['k'].append(k)
    df['metric'].append(metric)
    for i in range(len(k_s)):
        df['recall@'+str(k_s[i])].append(recalls[i])


for set in set_s:

    # Load of original pointclouds ids, to correctly identify
    pc_ids=np.loadtxt('kitti_tsne_'+set+'_ids.txt',dtype=np.int16)

    for run in run_s:

        # Load embeddings and corresponding labels (point cloud idx) for whole dataset
        # Each label identifies the point cloud 4 embedding originate from
        # So, all_embeds[0:4,:] are all embeddings of the 'all_labels[0]'-th point cloud 
        embeds=np.load('data/embeds_'+run+'_'+set+'.npy')
        labels=np.load('data/labels_'+run+'_'+set+'.npy')
        labels=pc_ids[labels]
        labels=np.repeat(labels,4)
        nr_embeds=embeds.shape[0]

        for metric in metric_s:

            dmatrix=squareform(pdist(embeds,metric))

            recalls=[]

            for k in k_s:

                positives=np.zeros(nr_embeds)

                for i in range(nr_embeds):
                    darray=dmatrix[i,:]
                    sorted_idx=np.argsort(darray)
                    
                    for j in range(k+1):
                        if labels[sorted_idx[j]]==labels[i]:
                            positives[i]+=1

                recall=(np.mean(positives)-1)/k
                recalls.append(recall)
                print('{}/{}\t{} {}\trecall@{}({})\t{}'.format(resulnr,totalnr,set,run,k,metric,recall),flush=True)
                resulnr+=1

            append_df(df,set,run,k,metric,recalls)

df = pd.DataFrame(df)
df.to_pickle('output.pkl')

