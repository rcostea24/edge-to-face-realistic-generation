import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from deep_att import Generator as GeneratorDeepWithAttention
from pix2pix import Generator as GeneratorPix2Pix, EdgeDataset as Pix2PixEdgeDataset, transform_only_input
from deep_att_res import Generator as GeneratorResidual
from deep_att_small_res import Generator as GeneratorSmallResidual
import cv2

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))


def create_edges(image: np.array) -> np.array:
    mean = np.mean(image)
    l = 0.66 * mean
    u = 1.33 * mean
    edges = cv2.Canny(image, l, u)
    return edges

def get_gen(path, type=None):
    checkpoint = torch.load(path) 

    if type == 1:
        gen = GeneratorDeepWithAttention()
    elif type == 2:
        gen = GeneratorResidual()
    elif type == 3:
        gen = GeneratorSmallResidual()   
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen

def get_pix2pix_gen():
    checkpoint = torch.load('/home/razvan/Facultate/Licenta/Pix2pix/b32/checkpoints/96_g_loss_gen.pth.tar') 

    gen = GeneratorPix2Pix(in_channels=3, features=64)
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen

def resize_sketches():
    def resizeAndPad(img, size, padColor=0):
        h, w = img.shape[:2]
        sh, sw = size
        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC
        # aspect ratio of image
        aspect = w/h
        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
            # set pad color
            if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, 
                np.ndarray)): # color image but only one color provided
                padColor = [padColor]*3
        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, 
            pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,  
            value=padColor)
        return scaled_img

    dataset_path = '../Datasets/sketches/cuhk/'
    files = sorted(os.listdir(dataset_path))

    i = 0
    for file in files:
        image = cv2.imread(dataset_path+file)
        image = center_crop(image, (190, 190))
        image = resizeAndPad(image, (256, 256))
        cv2.imwrite(f'../Datasets/sketches/cuhk_resized/{i}_center_crop.jpg', image)
        print(f"Done with {i}")
        i+=1

def test():
    # TYPE=1
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttention/checkpoints/46_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/5050/checkpoints/34_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/4060/checkpoints/33_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/2080/checkpoints/39_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/0100/checkpoints/35_loss_gen.pth.tar'

    #TYPE=3
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionSResidual_LPIPS/0100/checkpoints/32_loss_gen.pth.tar'

    #TYPE=2
    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionResidual_LPIPS/0100/checkpoints/36_loss_gen.pth.tar'

    gen = get_gen(path, type=2)

    dataset_path = '../Datasets/sketches/ar_resized/'
    files = sorted(os.listdir(dataset_path))

    for k in range(6):
        n = 5
        im = Image.new('RGB', (256*3, 256*n))

        i = 0
        for file in files[:n]:
            image = Image.open(dataset_path+file)
            
            img = np.array(image)
            edges = create_edges(img[:, :256, :])

            transform_input = T.Compose([
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])

            edges_pil = Image.fromarray(edges)

            input_image = transform_input(edges_pil)

            with torch.no_grad():
                y_gen = gen(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_gen = y_gen * 0.5 + 0.5
            
            transform = T.ToPILImage()
            y_gen = transform(y_gen)

            new_im = Image.new('RGB', (256*3, 256))

            new_im.paste(image, (0,0))
            new_im.paste(edges_pil, (256, 0))
            new_im.paste(y_gen, (256*2,0))

            im.paste(new_im, (0, 256*i))
            i+=1
            print(f'Done with image {i}/{n}')

        im.save(f'../U-net/DeepAttentionResidual_LPIPS/0100/test_ar_32_{k}.jpg')

def test_pix2pix():
    gen = get_pix2pix_gen()

    dataset_path = '../Datasets/sketches/ar_resized/'
    files = sorted(os.listdir(dataset_path))
    for k in range(6):
        n = 5#len(files)
        im = Image.new('RGB', (256*3, 256*n))

        i = 0
        for file in files[n*k:n*(k+1)]:
            image = Image.open(dataset_path+file)
            
            img = np.array(image)
            edges = create_edges(img[:, :256, :])
            edges = np.expand_dims(edges, axis=2)
            edges = np.concatenate((np.concatenate((edges, edges), axis=2), edges), axis=2)

            edges_pil = Image.fromarray(edges)

            input_image = transform_only_input(image=edges)["image"]

            with torch.no_grad():
                y_gen = gen(input_image.unsqueeze(0))[0].to('cpu').detach()

            y_gen = y_gen * 0.5 + 0.5
            
            transform = T.ToPILImage()
            y_gen = transform(y_gen)

            new_im = Image.new('RGB', (256*3, 256))

            new_im.paste(image, (0,0))
            new_im.paste(edges_pil, (256, 0))
            new_im.paste(y_gen, (256*2,0))

            im.paste(new_im, (0, 256*i))
            i+=1
            print(f'Done with image {i}/{n}')

        im.save(f'../Pix2pix/b32/test_ar_96_{k}.jpg')


if __name__ == '__main__':
    #resize_sketches()
    test()
    #test_pix2pix()
