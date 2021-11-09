import os
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from altusi.utils.logger import *

train_img_path = 'Training_Images'
train_annotation_path = 'Training_Annotation/Annotations-export.csv'
valid_img_path = 'Validation_Images'
valid_annotation_path = 'Validation_Annotation/annotations.csv'

def write_line(img_path, img_shape, bboxes, ids, idx):
    H, W, C = img_shape
    
    A, B, C, D = 4, 5, W, H
    
    labels = np.hstack((ids.reshape(-1, 1), bboxes)).astype('float')
    
    labels[:, (1, 3)] /= float(W)
    labels[:, (2, 4)] /= float(H)
    
    labels = labels.flatten().tolist()
    
    str_idx = [str(idx)]
    str_header = [str(_) for _ in [A, B, C, D]]
    str_labels = [str(_) for _ in labels]
    str_path = [img_path]
    
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'

    return line

lines = open('label.txt', 'r').read().strip().split('\n')

# Training Data

image_shapes = []
image_fnames = []
bboxes_lst = []
IDs_lst = []
annotation_lines = open(train_annotation_path,'r').read().strip().split('\n')
annotation_lines = annotation_lines[1:]
f_box_dict = {}
f_id_dict = {}
f_shape_dict = {}
for i in annotation_lines:
    fname = i.split(',')[0][1:-1]
    if fname in os.listdir(train_img_path):
        image_shape = cv.imread(f'{train_img_path}/{fname}').shape
        
        bbox = np.array(list(map(float, i.split(',')[1:5])))
        
        if fname in f_box_dict.keys():
            f_box_dict[fname].append(bbox)
            f_id_dict[fname].append(0)
            
        else:
            f_box_dict[fname] = [bbox]
            f_id_dict[fname] = [0]
            f_shape_dict[fname] = image_shape
image_fnames = list(f_box_dict.keys())
for k,v in f_box_dict.items():
    bboxes_lst.append(np.array(v))
for k,v in f_id_dict.items():
    IDs_lst.append(np.array(v))
for k,v in f_shape_dict.items():
    image_shapes.append(v)
     
with open('train_data.lst', 'w') as fw:
    for i, image_fname in enumerate(image_fnames):
        line = write_line(image_fname, 
                              image_shapes[i],
                              bboxes_lst[i],
                              IDs_lst[i],
                              i)

        fw.write(line)


# Validation Data

image_shapes = []
image_fnames = []
bboxes_lst = []
IDs_lst = []
annotation_lines = open(valid_annotation_path,'r').read().strip().split('\n')
annotation_lines = annotation_lines[1:]
f_box_dict = {}
f_id_dict = {}
f_shape_dict = {}
for i in annotation_lines:
    fname = i.split(',')[0][1:-1]
    if fname in os.listdir(valid_img_path):
        image_shape = cv.imread(f'{valid_img_path}/{fname}').shape
        
        bbox = np.array(list(map(float, i.split(',')[1:5])))
        
        if fname in f_box_dict.keys():
            f_box_dict[fname].append(bbox)
            f_id_dict[fname].append(0)
            
        else:
            f_box_dict[fname] = [bbox]
            f_id_dict[fname] = [0]
            f_shape_dict[fname] = image_shape
image_fnames = list(f_box_dict.keys())
for k,v in f_box_dict.items():
    bboxes_lst.append(np.array(v))
for k,v in f_id_dict.items():
    IDs_lst.append(np.array(v))
for k,v in f_shape_dict.items():
    image_shapes.append(v)
     
with open('valid_data.lst', 'w') as fw:
    for i, image_fname in enumerate(image_fnames):
        line = write_line(image_fname, 
                              image_shapes[i],
                              bboxes_lst[i],
                              IDs_lst[i],
                              i)

        fw.write(line)


