import torch 
import torchvision

import os
import natsort
from PIL import Image


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("L")
        tensor_image = self.transform(image)
        return tensor_image
    
    
def loader(root = './data', batch_size = 64) :
    tsfm = torchvision.transforms.ToTensor()
    _dataset = CustomDataSet(root, transform = tsfm)

    _loader = torch.utils.data.DataLoader(_dataset , 
                                          batch_size=batch_size, 
                                          shuffle=True)
    
    return _loader