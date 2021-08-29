import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import load
from yplot import read_tfevents
import yplot as yplot
from collections import defaultdict
import os
    
import glob

files = glob.glob("/home/baraujo/kitti/training/label_2/*")

cars=0
peds=0
cycs=0

for file in files:
    with open (file, "r") as myfile:
        labels=myfile.readlines()
        for label in labels:
            cls=label.split(' ')[0]
            if cls=='Car':
                cars+=1
            if cls=='Pedestrian':
                peds+=1
            if cls=='Cyclist':
                cycs+=1

print('Cars ',cars)
print('Peds ',peds)
print('Cycs ',cycs)

# for run in runs:

#     runpath = os.path.join(path,run,'eval/eval_all_default/default/tensorboard_val')
#     files = next(os.walk(runpath),([],[],[]))[2]

#     def def_value():
#         return np.empty(80)*np.nan
#     df=defaultdict(def_value)

#     df['epoch']=np.arange(80)+1

#     for file in files:
#         tb = read_tfevents(os.path.join(runpath,file))
#         for e in tb:
#             df[e.summary.value[0].tag][e.step-1]=e.summary.value[0].simple_value
            
#     df=pd.DataFrame(df)
#     df.dropna(inplace=True)
#     df.to_pickle('tb_pcd/'+run+'.pkl')
#     #df.to_csv('tb_pcd/'+run+'.csv',index=False)

