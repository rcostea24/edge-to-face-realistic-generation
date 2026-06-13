import numpy as np
import os
import cv2 as cv
import torch

def val_mean_std():
    path = '../Datasets/edge_to_image_faces/val/'
    files = sorted(os.listdir(path))

    for file in files:
        img = cv.imread(os.path.join(path, file))
        print(img.shape)

        
