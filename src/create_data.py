import numpy as np
import cv2 as cv
import os

def create_edge_map(in_path, out_path):
    files = sorted(os.listdir(in_path))
    n = 0
    num_images = len(files)

    for file in files:
        img = cv.imread(os.path.join(in_path, file), 0)
        img_blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)

        mean = np.mean(img)
        l = 0.66 * mean
        u = 1.33 * mean
        edges = cv.Canny(img, l, u)

        n+=1
        print(f'Saving image {n}/{num_images}')
        cv.imwrite(os.path.join(out_path, file), edges)

def create_name(i):
    name = ''
    for _ in range(5-len(str(i))):
        name += '0'
    name += str(i)
    return name

def concat_data(img_path, edge_path, out_path):
    files = sorted(os.listdir(img_path))
    n = 0
    num_images = len(files)
    n_train = 0
    n_val = 0

    for file in files:
        output = cv.imread(os.path.join(img_path, file))
        input = cv.imread(os.path.join(edge_path, file))

        image = cv.hconcat([input, output])

        n+=1

        if n % 5 == 0:
            cv.imwrite(os.path.join(out_path, f'val/{create_name(n_val)}.jpg'), image)
            n_val += 1
        else:
            cv.imwrite(os.path.join(out_path, f'train/{create_name(n_train)}.jpg'), image)
            n_train += 1
        
        print(f'Saved image {n}/{num_images}')

def downscale(in_path, out_path):
    files = sorted(os.listdir(in_path))
    n = 0
    num_images = len(files)

    for file in files:
        image = cv.imread(os.path.join(in_path, file))
        d_image = cv.resize(image, dsize=None, fx=0.25, fy=0.25)

        n += 1
        cv.imwrite(os.path.join(out_path, f'{create_name(n)}.jpg'), d_image)
        print(f'Saved image {n}/{num_images}')
    
        
if __name__ == '__main__':
    #downscale(in_path='../Datasets/CelebA/celeba_hq_256/', out_path='../Datasets/CelebA/celeba_64/')
    #create_edge_map(in_path='../Datasets/CelebA/celeba_hq_256/', out_path='../Datasets/CelebA/celeba_edge_b_256/')
    concat_data(img_path='../Datasets/CelebA/celeba_hq_256/', edge_path='../Datasets/CelebA/celeba_edge_b_256/', out_path='../Datasets/CelebA/edge_to_face_b/')

