import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from deep_att import Generator as GeneratorDeepWithAttention
from deep_att_res import Generator as GeneratorResidual
import cv2 
from skimage.metrics import structural_similarity

def preprocess(image):
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
    
    if image.shape[0] != 256 and image.shape[1] != 256:
        image = resizeAndPad(image)
    mean = np.mean(image)
    l = 0.66 * mean
    u = 1.33 * mean
    edges = cv2.Canny(image, l, u)

    transform_input = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    edges_pil = Image.fromarray(edges)

    input_image = transform_input(edges_pil)

    return input_image

def get_gen(path, type=None):
    checkpoint = torch.load(path) 

    if type == 1:
        gen = GeneratorDeepWithAttention()
    elif type == 2:
        gen = GeneratorResidual()
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen

if __name__ == '__main__':
    image = Image.open('/home/razvan/Facultate/Licenta/Datasets/random/sketch_256.jpg')
    input_image = preprocess(np.array(image))
    
    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionResidual_LPIPS/0100/checkpoints/16_loss_gen.pth.tar'
    gen = get_gen(path, type=2)

    with torch.no_grad():
        y_gen = gen(input_image.unsqueeze(0))[0].to('cpu').detach()

    y_gen = y_gen * 0.5 + 0.5

    transform = T.ToPILImage()
    y_gen = transform(y_gen)
    
    ssim = structural_similarity(np.array(image), np.array(y_gen), channel_axis=2, multichannel=True)
    print(ssim)

    im = Image.new('RGB', (256*3, 256))
    im.paste(image, (0, 0))
    im.paste(transform(input_image), (256, 0))
    im.paste(y_gen, (256*2, 0))
    im.show()
