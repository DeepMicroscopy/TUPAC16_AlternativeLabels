#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import numpy as np
from SlideRunner.dataAccess.database import Database
from tqdm import tqdm
from pathlib import Path
import openslide
import time
import pickle
import cv2
import torchvision.transforms as transforms
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys

sys.path.append('lib/')
from data_loader import *

from lib.object_detection_helper import *
from model.RetinaNetFocalLoss import RetinaNetFocalLoss
from model.RetinaNet import RetinaNet


# In[2]: 


import sys

if (len(sys.argv)<5):
    print('Syntax: Inference-RetinaNet-var.py ModelSavepath.pth Database.sqlite DatasetName val/test test_slides.p [SlideDir]')
    exit()

fname = 'RetinaNet-ODAEL-export.pth'#RetinaNet-CMCT-ODAEL-export.pth'
#FullModel256-Fold1-HardExampSampling-export.pth'
if len(sys.argv)>1:
    fname = sys.argv[1]

size=512
path = Path('./')

database = Database()
database.open(str(sys.argv[2]))#Slides_Mitosis_final_checked_cleaned.sqlite'))
slidedir = 'WSI' if len(sys.argv)<6 else sys.argv[6]
datasetname= sys.argv[3]




try:
    result_boxes = pickle.load(open("%s-%s-inference_results_boxes.p" % (fname,datasetname), "rb"))
except:
    result_boxes = {}



size = 512
level = 0

files = []


# In[3]:

test_slide_filenames = pickle.load(open(sys.argv[5],'rb'))#[x for x in np.arange(53,108)]


print('Test slides are: ',test_slide_filenames)

#val = '-val' if sys.argv[4] == 'val' else ''
val = '-val'


print('Summary: \n\n')
print('%20s: %20s' % ('Model', fname))
print('%20s: %20s' % ('Database', sys.argv[2]))
print('%20s: %20s' % ('Datasetname', datasetname))
print('%20s: %20s' % ('Validation/test', 'validation' if val=='-val' else 'test'))


# In[4]:


files=list()
train_slides=list()
val_slides=list()
test_slides=list()
slidenames = list()
getslides = """SELECT uid, directory, filename FROM Slides"""
for idx, (currslide, folder, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
    if (((sys.argv[4] == 'val') and (currslide not in test_slide_filenames)) 
        or (not (sys.argv[4] == 'val') and (currslide in test_slide_filenames))):
        slidenames += [currslide]

        

        database.loadIntoMemory(currslide)

        slide_path = path / slidedir / filename

        slide = openslide.open_slide(str(slide_path))

        level = 0#slide.level_count - 1

        files.append(SlideContainer(file=slide_path, level=level, width=size, height=size, y=[[], []], annotations=dict()))
        test_slides.append(idx)
                         
        
print('Running on slides:', slidenames)

state = torch.load(fname, map_location='cpu')     if defaults.device == torch.device('cpu')     else torch.load(fname)
model = state.pop('model').cuda()
mean = state['data']['normalize']['mean']
std = state['data']['normalize']['std']
print('MEAN:',mean)
print('STD:',std)

# In[ ]:





#anchors = create_anchors(sizes=[(16,16)], ratios=[1], scales=[0.3, 0.375,0.45])
anchors = create_anchors(sizes=[(32,32)], ratios=[1], scales=[0.6, 0.7,0.8,0.9])

detect_thresh = 0.3 
nms_thresh = 0.4
result_regression = {}



import threading
#from mu import Queue as mpQueue
from queue import Queue
import queue
import time

jobQueue=Queue()
outputQueue=Queue()

def getPatchesFromQueue(jobQueue, outputQueue):
    x,y=0,0
    try:
        while (True):
            if (outputQueue.qsize()<100):
                status, x,y, slide_container = jobQueue.get(timeout=60)
                if (status==-1):
                    return
                outputQueue.put((x,y,slide_container.get_patch(x, y) / 255.))
            else:
                time.sleep(0.1)
    except queue.Empty:
        print('One worker died.')
        pass # Timeout happened, exit



def getBatchFromQueue(batchsize=8):
    images = np.zeros((batchsize,3, size,size))
    x = np.zeros(batchsize)
    y = np.zeros(batchsize)
    try:
        bs=0
        for k in range(batchsize):
            x[k],y[k],images_temp = outputQueue.get(timeout=5)
            images[k] = images_temp.transpose((2,0,1))
            bs+=1
        return images,x,y
    except queue.Empty:
        return images[0:bs],x[0:bs],y[0:bs]



# In[8]:


def rescale_box(bboxes, size: Tensor):
    bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
    bboxes[:, :2] = (bboxes[:, :2] + 1) * size / 2
    bboxes[:, 2:] = bboxes[:, 2:] * size / 2
    bboxes = bboxes.long()
    return bboxes


# In[9]:


import time
from functools import partial
class timerObj():
    model = 0
    transform = 0
    get = 0
    postproc = 0
    def __str__(self):
        return 'Model: %.2f, Transform: %.2f, Get: %.2f, Postproc: %.2f' % (self.model, self.transform, self.get, self.postproc)

t = timerObj()
batchsize=8

# Set up queued image retrieval
jobs = []
for i in range(1):
    p = threading.Thread(target=getPatchesFromQueue, args=(jobQueue, outputQueue), daemon=True)
    jobs.append(p)
    p.start()

device = torch.device("cuda:0")
model = model.cuda(device)
with torch.no_grad():
    for slide_container in (files):

        print('Getting overview from ',slide_container.file)
        size = state['data']['tfmargs']['size']
        #result_boxes[slide_container.file.name] = []

        if '/'.join(str(slide_container.file).split('/')[-1:]) in result_boxes:
            continue
        result_boxes[str(slide_container.file).split(os.sep)[-1]] = []
        print('Processing WSI ...')

        n_Images=0
        for x in tqdm(range(0, slide_container.slide.level_dimensions[level][0] - 1 * size, int(0.9*size)), position=1):
            for y in range(0, slide_container.slide.level_dimensions[level][1] - 1*  size, int(0.9*size)):
                jobQueue.put((0,x,y, slide_container))
                n_Images+=1
        

        for kImage in tqdm(range(int(np.ceil(n_Images/batchsize))), desc='Processing %s' % slide_container.file):
                

                npBatch,xBatch,yBatch = getBatchFromQueue(batchsize=batchsize)
                imageBatch = torch.from_numpy(npBatch.astype(np.float32, copy=False)).cuda()
                
                patch = imageBatch
                
                for p in range(patch.shape[0]):
                    patch[p] = transforms.Normalize(mean,std)(patch[p])
                
                class_pred_batch, bbox_pred_batch, _ = model(
                    patch[:, :, :, :])

                for b in range(patch.shape[0]):
                    x_real = xBatch[b]
                    y_real = yBatch[b]
                    
                    for clas_pred, bbox_pred in zip(class_pred_batch[b][None,:,:], bbox_pred_batch[b][None,:,:],
                                                                            ):
                        modelOutput = process_output(clas_pred, bbox_pred, anchors, detect_thresh)
                        bbox_pred, scores, preds = [modelOutput[x] for x in ['bbox_pred', 'scores', 'preds']]
                        
                        if bbox_pred is not None:
                            to_keep = nms(bbox_pred, scores, nms_thresh)
                            bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

                            t_sz = torch.Tensor([size, size])[None].float()

                            bbox_pred = rescale_box(bbox_pred, t_sz)

                            for box, pred, score in zip(bbox_pred, preds, scores):
                                y_box, x_box = box[:2]
                                h, w = box[2:4]

                                result_boxes[str(slide_container.file).split(os.sep)[-1]].append(np.array([x_box + x_real, y_box + y_real,
                                                                                         x_box + x_real + w, y_box + y_real + h,
                                                                                         pred, score]))

        import time
        #fwriter.close()
        pickle.dump(result_boxes, open("%s-%s-inference_results_boxes.p" % (fname,datasetname), "wb"))
    jobQueue.put((-1,0,0,None))
        








