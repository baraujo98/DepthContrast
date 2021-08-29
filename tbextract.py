import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import load
from yplot import read_tfevents
import yplot as yplot
from collections import defaultdict
import os

paths = ['/home/baraujo/DepthContrast/third_party/OpenPCDet/output/kitti_models/pointrcnn_iou_finetune',
         '/home/baraujo/DepthContrast/third_party/OpenPCDet/output/kitti_models/pointrcnn_iou'
        ]

for path in paths:
    
    runs = next(os.walk(path),([],[],[]))[1]

    for run in runs:

        runpath = os.path.join(path,run,'eval/eval_all_default/default/tensorboard_val')
        files = next(os.walk(runpath),([],[],[]))[2]

        def def_value():
            return np.empty(80)*np.nan
        df=defaultdict(def_value)

        df['epoch']=np.arange(80)+1

        for file in files:
            tb = read_tfevents(os.path.join(runpath,file))
            for e in tb:
                df[e.summary.value[0].tag][e.step-1]=e.summary.value[0].simple_value
                
        df=pd.DataFrame(df)
        df.dropna(inplace=True)
        df.to_pickle('tb_pcd/'+run+'.pkl')
        #df.to_csv('tb_pcd/'+run+'.csv',index=False)

