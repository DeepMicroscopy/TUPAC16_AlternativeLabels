"""

    Extraction of detections in TUPAC CSV format

    Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg
    
    To reduce complexity of calculation, the F1 score is here derived by querying a KD Tree.
    
    This is possible, since all objects are round and have the same diameter. 

"""
import pickle
import os
import numpy as np
from lib.nms_WSI import nms

def exportCSV(filepath,resultsfile, threshold):
    results = pickle.load(open(resultsfile,'rb'))
    results = nms(results, threshold)
    for k in results:
        dirname = k.split('_')[0]
        os.system(f'mkdir -p {filepath}/{dirname}')
        boxes = np.array(results[k])
        center_x = (boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2).tolist() if boxes.shape[0]>0 else []
        center_y = (boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2).tolist() if boxes.shape[0]>0 else []
        scores = (boxes[:,-1]).tolist() if boxes.shape[0]>0 else []
        f = open(f'{filepath}/{dirname}/01.csv','w')
        for (cx,cy,s) in zip(center_x,center_y,scores):
            if (s>threshold):
                f.write(f'{int(cy)},{int(cx)}\n')
        f.close()        


