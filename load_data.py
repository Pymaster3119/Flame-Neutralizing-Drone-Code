import torch
import os
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, trainingInUse, testingInUse, validationInUse, mikeyInUse, vedantInUse, adityaInUse, transform):
        self.transform = transform
        self.dataset_dir = "CompositeDataset/"
        self.images = []

        peopleinuse = []
        if mikeyInUse:
            peopleinuse.append("mikey/")
        if vedantInUse:
            peopleinuse.append("vedant/")
        if adityaInUse:
            peopleinuse.append("aditya/")

        splitsinuse = []
        if trainingInUse:
            splitsinuse.append("train/")
        if testingInUse:
            splitsinuse.append("test/")
        if validationInUse:
            splitsinuse.append("val/")

        for split in splitsinuse:
            for person in peopleinuse:
                path = self.dataset_dir + split + person

                for i in range(0, 2):
                    for j in os.listdir(os.path.join(path, str(i))):
                        imgname = os.path.join(path, str(i), j)
                        self.images.append((imgname, i))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        imgname = self.images[idx]
        image = Image.open(imgname[0]).convert('RGB')
        label = imgname[1]
        if self.transform != None:
            image = self.transform(image)

        return image, torch.tensor(label)