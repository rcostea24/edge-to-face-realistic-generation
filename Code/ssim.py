from pytorch_msssim import ssim
from skimage.metrics import structural_similarity
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    real_path = '/home/razvan/Facultate/Licenta/Datasets/CelebA/validation/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/pix2pix/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/5050/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/4080/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/2080/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_lpips/0100/'
    fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_sres_lpips/0100/'
    #fake_path = '/home/razvan/Facultate/Licenta/Datasets/u-net/att_res_lpips/0100/'

    print(real_path)
    print(fake_path)

    real_files = sorted(os.listdir(real_path))
    fake_files = sorted(os.listdir(fake_path))

    real_images = []
    fake_images = []

    score = 0

    for i, (real_file, fake_file) in enumerate(zip(real_files, fake_files)):
        real = cv2.imread(os.path.join(real_path, real_file))
        fake = cv2.imread(os.path.join(fake_path, fake_file))

        score += structural_similarity(real, fake, data_range=fake.max() - fake.min(), multichannel=True)

        if (i+1) % 1000 == 0:
            print(f'Done with {i+1}/6000')

    print(f"SSIM: {score/len(real_files)}")

    
    