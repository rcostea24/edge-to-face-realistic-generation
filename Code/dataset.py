from torchvision.utils import save_image
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T

transform_input = T.Compose([
    T.Grayscale(num_output_channels=1),
    #T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

transform_target = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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
        
        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)

        input_image = transform_input(input_image)
        target_image = transform_target(target_image)

        return input_image, target_image