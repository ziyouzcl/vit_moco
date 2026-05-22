import os

import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class ImageListDataset(VisionDataset):
    def __init__(self, data_root, listfile, transform, gray=False, nolabel=False, multiclass=False):
        self.image_list = []
        self.label_list = []
        self.nolabel = nolabel
        with open(listfile) as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split()
                image_path = os.path.join(data_root, items[0])
                if not nolabel:
                    if not multiclass:
                        label = int(items[1])
                    elif multiclass:
                        label = list(map(float, items[1:]))
                    else:
                        raise ValueError("Line format is not right")
                self.image_list.append(image_path)
                if not nolabel:
                    self.label_list.append(label)

        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = self.transform(image)

        if not self.nolabel:
            label = self.label_list[index]
            return image, torch.tensor(label)
        else:
            return image
