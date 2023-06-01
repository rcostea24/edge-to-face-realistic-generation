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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        return x

# Generator
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2 ,1 , bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.deconv(x)
        return self.dropout(x) if self.use_dropout else x
    
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        self.down4 = Down(features * 8, features * 8)
        self.down5 = Down(features * 8, features * 8)
        self.down6 = Down(features * 8, features * 8)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())

        self.up1 = Up(features * 8, features * 8, dropout=True)
        self.up2 = Up(features * 8 * 2, features * 8, dropout=True)
        self.up3 = Up(features * 8 * 2, features * 8, dropout=True)
        self.up4 = Up(features * 8 * 2, features * 8, dropout=False)
        self.up5 = Up(features * 8 * 2, features * 4, dropout=False)
        self.up6 = Up(features * 4 * 2, features * 2, dropout=False)
        self.up7 = Up(features * 2 * 2, features, dropout=False)
        
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),nn.Tanh(),)

    def forward(self, x):
        #print(x.shape)
        d1 = self.initial_down(x)
        #print(d1.shape)
        d2 = self.down1(d1)
        #print(d2.shape)
        d3 = self.down2(d2)
        #print(d3.shape)
        d4 = self.down3(d3)
        #print(d4.shape)
        d5 = self.down4(d4)
        #print(d5.shape)
        d6 = self.down5(d5)
        #print(d6.shape)
        d7 = self.down6(d6)
        #print(d7.shape)
        bottleneck = self.bottleneck(d7)
        #print(bottleneck.shape)
        up1 = self.up1(bottleneck)
        #print(up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        #print(up2.shape)
        up3 = self.up3(torch.cat([up2, d6], 1))
        #print(up3.shape)
        up4 = self.up4(torch.cat([up3, d5], 1))
        #print(up4.shape)
        up5 = self.up5(torch.cat([up4, d4], 1))
        #print(up5.shape)
        up6 = self.up6(torch.cat([up5, d3], 1))
        #print(up6.shape)
        up7 = self.up7(torch.cat([up6, d2], 1))
        #print(up7.shape)
        return self.final_up(torch.cat([up7, d1], 1))  
    
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        #A.ToGray(always_apply=True, p=1.0),
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

class EdgeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :]
        target_image = image[:, 256:, :]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image 


def get_gen(path):
    checkpoint = torch.load(path) 

    gen = Generator(in_channels=3, features=64)
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
        

if __name__ == '__main__':
    # gen = Generator()
    # x = torch.rand([1, 3, 256, 256])
    # count_parameters(gen)
    # #print(gen)
    # print(gen(x).shape)

    disc = Discriminator()
    x = torch.rand([1, 3, 256, 256])
    y = torch.rand([1, 3, 256, 256])
    count_parameters(disc)
    disc(x, y).shape

