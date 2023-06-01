import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import torchvision.transforms as T
from pix2pix import Generator 

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

def get_gen(path, type=None):
    checkpoint = torch.load(path) 

    gen = Generator()
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen


if __name__ == '__main__':
    path = '/home/razvan/Facultate/Licenta/Pix2pix/b32/checkpoints/96_g_loss_gen.pth.tar'
    gen = get_gen(path)

    dataset_path = '../Datasets/CelebA/edge_to_image_faces_celeba/val/'
    #dataset_path = '../Datasets/edge2face/val/'
    files = sorted(os.listdir(dataset_path))
    n = 5
    for k in range(6):
        im = Image.new('RGB', (256*3, 256*n))

        i = 0
        for file in files[n*k:n*(k+1)]:
            image = Image.open(dataset_path + file)
            img = np.array(image)
            input_image = transform_only_input(image=img[:, :256, :])['image']

            with torch.no_grad():
                y_gen = gen(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_gen = y_gen * 0.5 + 0.5
            
            transform = T.ToPILImage()
            y_gen = transform(y_gen)

            new_im = Image.new('RGB', (256*3, 256))

            new_im.paste(image, (0,0))
            new_im.paste(y_gen, (256*2,0))

            im.paste(new_im, (0, 256*i))
            i+=1
            print(f'Done with image {i}/{n}')

        im.save(f'../Pix2pix/b32/test_96_{k}.jpg')
    
    

    

        

        
        
