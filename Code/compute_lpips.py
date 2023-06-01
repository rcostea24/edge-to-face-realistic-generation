import torch
import numpy as np
from scipy.linalg import sqrtm

import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import cv2 as cv

from PerceptualSimilarity.lpips import lpips

preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    real_path = '/home/razvan/Facultate/Licenta/Datasets/CelebA/validation/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/pix2pix/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/5050/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/4060/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/2080/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/0100/'
    fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_sres_lpips/0100/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_res_lpips/0100/'

    print(real_path)
    print(fake_path)

    real_files = sorted(os.listdir(real_path))
    fake_files = sorted(os.listdir(fake_path))

    loss_fn = lpips.LPIPS(net='vgg')
    loss_fn.cuda()

    d = 0

    for i, (real_file, fake_file) in enumerate(zip(real_files, fake_files)):
        real = lpips.im2tensor(lpips.load_image(os.path.join(real_path, real_file)))
        fake = lpips.im2tensor(lpips.load_image(os.path.join(fake_path, fake_file)))

        real = real.cuda()
        fake = fake.cuda()

        d += torch.mean(loss_fn(real, fake)).item()
        if (i+1) % 1000 == 0:
            print(f'Done with {i+1}/6000')

    score = d/len(real_files)
    print(f"LPIPS: {score}")
        


   