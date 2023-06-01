import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from deep_att import Generator as GeneratorDeepWithAttention
from deep_att_res import Generator as GeneratorResidual
from deep_att_small_res import Generator as GeneratorSmallResidual
from pix2pix import Generator as GeneratorPix2Pix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

transform_input = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

def get_gen(path, type=None):
    checkpoint = torch.load(path) 

    if type == 1:
        gen = GeneratorDeepWithAttention()
    elif type == 2:
        gen = GeneratorResidual()
    elif type == 3:
        gen = GeneratorPix2Pix()
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen

def create_edges(image: np.array) -> np.array:
    mean = np.mean(image)
    l = 0.66 * mean
    u = 1.33 * mean
    edges = cv2.Canny(image, l, u)
    return edges

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

def expo():
    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/0100/checkpoints/35_loss_gen.pth.tar'
    gen_att = get_gen(path, type=1)

    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionResidual_LPIPS/0100/checkpoints/36_loss_gen.pth.tar'
    gen_res_att = get_gen(path, type=2)

    path = '/home/razvan/Facultate/Licenta/Pix2pix/b32/checkpoints/96_g_loss_gen.pth.tar'
    gen_pix2pix = get_gen(path, type=3)


    dataset_path = '../Datasets/CelebA/edge_to_image_faces_celeba/val/'
    #dataset_path = '../Datasets/edge2face/val/'
    files = sorted(os.listdir(dataset_path))
    n = 5

    for k in range(7):
        im = Image.new('RGB', (256*5, 256*n))

        i = 0
        to_pil = T.ToPILImage()
        for file in files[n*k:n*(k+1)]:
            image = Image.open(dataset_path + file)

            #Pix2Pix
            img = np.array(image)
            input_image = transform_only_input(image=img[:, :256, :])['image']

            with torch.no_grad():
                y_pix2pix = gen_pix2pix(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_pix2pix = y_pix2pix * 0.5 + 0.5
            y_pix2pix = to_pil(y_pix2pix)

            # Att & ResAtt
            img = np.array(image)
            input_image = transform_input(Image.fromarray(img[:, :256, :]))

            with torch.no_grad():
                y_att = gen_att(input_image.unsqueeze(0))[0].to('cpu').detach()
                y_res_att = gen_res_att(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_att = y_att * 0.5 + 0.5
            y_att = to_pil(y_att)

            y_res_att = y_res_att * 0.5 + 0.5
            y_res_att = to_pil(y_res_att)

            new_im = Image.new('RGB', (256*5, 256))

            new_im.paste(image, (0,0))
            new_im.paste(y_pix2pix, (256*2,0))
            new_im.paste(y_att, (256*3,0))
            new_im.paste(y_res_att, (256*4,0))

            im.paste(new_im, (0, 256*i))
            i+=1
            print(f'{k} Done with image {i}/{n}')

        im.save(f'../Datasets/Expo/{k}.jpg')
    
def expo_s():
    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/0100/checkpoints/35_loss_gen.pth.tar'
    gen_att = get_gen(path, type=1)

    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionResidual_LPIPS/0100/checkpoints/36_loss_gen.pth.tar'
    gen_res_att = get_gen(path, type=2)

    path = '/home/razvan/Facultate/Licenta/Pix2pix/b32/checkpoints/96_g_loss_gen.pth.tar'
    gen_pix2pix = get_gen(path, type=3)


    dataset_path = '../Datasets/sketches/ar_resized/'
    files = sorted(os.listdir(dataset_path))
    n = 5

    for k in range(7):
        im = Image.new('RGB', (256*5, 256*n))

        i = 0
        to_pil = T.ToPILImage()
        for file in files[n*k:n*(k+1)]:
            image = Image.open(dataset_path + file)
            img = np.array(image)
            edges = create_edges(img)
            edges_pil =Image.fromarray(edges)

            #Pix2Pix
            edges_pix2pix = np.expand_dims(edges, axis=2)
            edges_pix2pix = np.concatenate((np.concatenate((edges_pix2pix, edges_pix2pix), axis=2), edges_pix2pix), axis=2)
            input_image = transform_only_input(image=edges_pix2pix)['image']

            with torch.no_grad():
                y_pix2pix = gen_pix2pix(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_pix2pix = y_pix2pix * 0.5 + 0.5
            y_pix2pix = to_pil(y_pix2pix)

            # Att & ResAtt
            input_image = transform_input(edges_pil)

            with torch.no_grad():
                y_att = gen_att(input_image.unsqueeze(0))[0].to('cpu').detach()
                y_res_att = gen_res_att(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_att = y_att * 0.5 + 0.5
            y_att = to_pil(y_att)

            y_res_att = y_res_att * 0.5 + 0.5
            y_res_att = to_pil(y_res_att)

            new_im = Image.new('RGB', (256*5, 256))

            new_im.paste(image, (0,0))
            new_im.paste(edges_pil, (256,0))
            new_im.paste(y_pix2pix, (256*2,0))
            new_im.paste(y_att, (256*3,0))
            new_im.paste(y_res_att, (256*4,0))

            im.paste(new_im, (0, 256*i))
            i+=1
            print(f'{k} Done with image {i}/{n}')

        im.save(f'../Datasets/Expo/{k}_sketch.jpg')


if __name__ == '__main__':
    #expo()
    expo_s()
    
    

    

        

        
        
