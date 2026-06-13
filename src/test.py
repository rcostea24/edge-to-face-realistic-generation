import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from deep_att import Generator as GeneratorDeepWithAttention
from deep_att_res import Generator as GeneratorResidual
from deep_att_small_res import Generator as GeneratorSmallResidual

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
        gen = GeneratorSmallResidual()
    gen = nn.DataParallel(gen)
    gen.to('cuda')

    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    return gen


if __name__ == '__main__':
    # TYPE=1
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttention/checkpoints/46_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/5050/checkpoints/34_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/4060/checkpoints/33_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/2080/checkpoints/39_loss_gen.pth.tar'
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionLPIPS/0100/checkpoints/35_loss_gen.pth.tar'

    #TYPE=3
    path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionSResidual_LPIPS/0100/checkpoints/32_loss_gen.pth.tar'

    #TYPE=2
    #path = '/home/razvan/Facultate/Licenta/U-net/DeepAttentionResidual_LPIPS/0100/checkpoints/36_loss_gen.pth.tar'
    gen = get_gen(path, type=3)

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
            input_image = transform_input(Image.fromarray(img[:, :256, :]))

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
            print(f'{k} Done with image {i}/{n}')

        #im.save('../U-net/DeepAttentionLPIPS_FFHQ/0100/test_15.jpg')
        im.save(f'../U-net/DeepAttentionSResidual_LPIPS/0100/test_32_{k}.jpg')
    
    

    

        

        
        
