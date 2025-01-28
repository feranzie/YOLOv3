"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

#import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES+True

class YOLODataset(Dataset):
    def __init (
            self,
            csv-file,
            img_dir, label_dir,
            anchors,
            image_size=416,
            S[13,26,52],  # grid sizes
            C=20, # classes
            transform=None
    ):
        self.annotations=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.transform=transform
        self.S=S
        self.anchors=torch.tensor(anchors[0]*anchors[1]* anchors[2])
        self.num_anchors=self.anchors.shape[0]
        self.num_anchors_per_scale=self.num_anchors//3
        self.C=C
        self.ignore_iou_treshold=0.5  

    def __len__(self):
        return(self.annotations)

    def __getitem__(self,index):
    Label_path=os.path.join(self.label_dir,self.annotations.iloc[indesx,1])
    bboxes=np.roll(np.loadtxt(fname=Label_path,delimeter="", ndim=2),4,axis=1).tolist() #[class,x,y,w,h]
    img_path=os.path.join(self.img_dir,self.annotations.iloc[indesx,0])
    image=np.array(Image.open(img_path).convert("RGB"))
    if self.transform:
        augmentations=self.transform(image=image, bboxes=bboxes)
        image=augmentations['images']
        bboxes=augmentations["bboxes"]

    #get same no of anchors for each prediction
    targets=[torch.zeroes((self.num_anchors//3,S,S,6, for S in self.S ))]
    #for each target prob that there is an object
    #{p_o,x,y,w,h.class}

    #check which anchor should be responsible for prediction by selecting the one with hghest iou
    for box in bboxes:
        oiou_anchors=iou(torch.tensor(box[2:4], self.anchors))
        anchor_indices= iou_anchors.argsort(decending=True), dim=0

        x,y,width,height,class_label=box
        has_anch0r={False,False,False}

        for anchor_idx in anchor_indices:
            #check whic anchor does it belong to
            scale_idx=anchor_idx//self.num_anchors_per_scale #0,1,2
            anchor_on_scale=anchor_idx% self.num_anchors_per_scale
            S=self.S[scale_idx]
