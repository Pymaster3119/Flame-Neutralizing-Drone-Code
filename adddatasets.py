import torch
from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader, random_split
import tqdm
import os

batch_size = 1
DATASET_PATH = "CompositeDataset/"

#region Mikey
dataset = datasets.ImageFolder(root="FireDataset_1_InAndOutdoor")
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
validation_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - validation_size #do this so that we get whole number splits

print(f"Training images:   {train_size}")
print(f"Validation images: {validation_size}")
print(f"Testing images:    {test_size}")

train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size, validation_size, test_size])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#train set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(train_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"train/"+"mikey/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"train/"+"mikey/"+"1/"+str(firenumber)+".png")

#test set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(test_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"test/"+"mikey/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"test/"+"mikey/"+"1/"+str(firenumber)+".png")

#val set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(validation_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"val/"+"mikey/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"val/"+"mikey/"+"1/"+str(firenumber)+".png")


#region Vedant
dataset = datasets.ImageFolder(root="VedantDataset")
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
validation_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - validation_size #do this so that we get whole number splits

print(f"Training images:   {train_size}")
print(f"Validation images: {validation_size}")
print(f"Testing images:    {test_size}")

train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size, validation_size, test_size])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#train set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(train_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"train/"+"vedant/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        print("here")
        image.save(DATASET_PATH+"train/"+"vedant/"+"1/"+str(firenumber)+".png")

#test set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(test_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"test/"+"vedant/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"test/"+"vedant/"+"1/"+str(firenumber)+".png")

#val set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(validation_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"val/"+"vedant/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"val/"+"vedant/"+"1/"+str(firenumber)+".png")


#region aditya
class FASDDDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset_dir = "FASDD/images"
        self.images = []

        for i in ["train", "test", "val"]:
            for idx, j in enumerate(tqdm.tqdm(os.listdir(os.path.join(dataset_dir, i)))):
                imgname = os.path.join(dataset_dir, i, j)
                self.images.append(imgname)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        imgname = self.images[idx]
        image = Image.open(imgname).convert('RGB')
        label = 0
        if "bothfireandsmoke" in imgname.lower() or "fire_" in imgname.lower():
            label = 1

        return image, torch.tensor(label)
    
dataset = FASDDDataset()
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
validation_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - validation_size #do this so that we get whole number splits

print(f"Training images:   {train_size}")
print(f"Validation images: {validation_size}")
print(f"Testing images:    {test_size}")

train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size, validation_size, test_size])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#train set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(train_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"train/"+"aditya/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        print("here")
        image.save(DATASET_PATH+"train/"+"aditya/"+"1/"+str(firenumber)+".png")

#test set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(test_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"test/"+"aditya/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"test/"+"aditya/"+"1/"+str(firenumber)+".png")

#val set
nonenumber = 0
firenumber = 0
for i in tqdm.tqdm(validation_dataset):
    image = i[0]
    label = i[1]
    if label == 0:
        nonenumber += 1
        image.save(DATASET_PATH+"val/"+"aditya/"+"0/"+str(nonenumber)+".png")
    if label == 1:
        firenumber += 1
        image.save(DATASET_PATH+"val/"+"aditya/"+"1/"+str(firenumber)+".png")