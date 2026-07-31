import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import tqdm
import os
import numpy as np
from sahi.slicing import get_slice_bboxes
import torch.nn.functional as F

#3.1: Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=0),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
        )

        self.postskip2maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(127008, 128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32, 1) #put num_classes instead of 1 if using multi-class
        )

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        z = y 
        z = self.postskip2maxpool(z)
        z = self.classifier(z)
        return z

model = SimpleCNN()
model.load_state_dict(torch.load("highres.pth"))
model = model.to("mps")

class FASDDDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        dataset_dir = "FireDataset_1_InAndOutdoor"
        self.images = []
        self.widths = []
        self.heights = []

        for i in ["0", "1"]:
            for idx, j in enumerate(tqdm.tqdm(os.listdir(os.path.join(dataset_dir, i)))):
                imgname = os.path.join(dataset_dir, i, j)
                self.images.append(imgname)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        imgname = self.images[idx]
        image = Image.open(imgname).convert('RGB')
        label = 0
        if "/1/" in imgname.lower():
            label = 1

        self.widths.append(image.width)
        self.heights.append(image.height)
        image = self.transform(image)

        ## R < G < B
        mask = np.copy(image)
        binmask = (image[0,:,:] < image[1,:,:]) | (image[1,:,:] < image[2,:,:])
        binmask = binmask.tile(3,1,1)
        mask[binmask] = 0

        #RGB -> HSV
        hsv = np.copy(mask)
        thetanumerator = 0.5 * ((mask[0,:,:] - mask[1,:,:]) + (mask[0,:,:] - mask[2,:,:]))
        thetadenominator = np.sqrt((mask[0,:,:] - mask[1,:,:])**2 + (mask[0,:,:] - mask[1,:,:])*(mask[0,:,:] - mask[2,:,:]) + 1e-10) + 1e-10
        theta = np.acos(np.clip(thetanumerator/thetadenominator, -1, 1))
        hue = np.where(mask[2]>mask[0], theta, 360-theta)
        saturation = 1 - (3/(mask[0,:,:] + mask[1,:,:] + mask[2,:,:] + 1e-10)) * np.min(mask,axis=0)
        intensity = 1/3 * (mask[0,:,:] + mask[1,:,:] + mask[2,:,:])

        hsv[0] = hue / 360
        hsv[1] = saturation
        hsv[2] = intensity

        puttogether = np.concatenate((image, hsv), axis=0)
        return puttogether, torch.tensor(label)

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = FASDDDataset(transform)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)
images, labels = next(iter(test_loader))
images = images.to("mps")

print(f"Number of images: {len(test_dataset)}")


#6.1: Predictions on test set
# Store predictions and true labels
all_predictions = []
all_labels = []

# Disable gradient calculations

best_acc = 0
best_thres = 0
for k in range(1, 1, 1):
    missed_imgs = []
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images = images.to("mps")
            labels = labels.to("mps")

            target_shape = tuple(max(c, m) for c, m in zip(images.shape, (1,6,256,256)))
            images= F.interpolate(images, size=target_shape[2:], mode='bilinear', align_corners=False)
            percentage_on = torch.count_nonzero(images[0,3,:,:])/images[0,3,:,:].numel()
            if percentage_on < k/100:
                # Get model outputs
                #1. GET BOUNDING BOXES
                bboxes = get_slice_bboxes(images.shape[2], images.shape[3], 256, 256)
                predicted = 0
                for i in bboxes:
                    left = images.shape[2] - 256 if i[2] > images.shape[2] - 1 else i[0]
                    right = min(i[2], images.shape[2])
                    up = images.shape[3] - 256 if i[3] > images.shape[3] - 1 else i[1]
                    down = min(i[3], images.shape[3])
                    if down-up != 256:
                        print(down-up)
                        print(i)
                        print(images.shape)
                        print(down)
                        print(up)
                    curr_pred = torch.sigmoid(model(images[:,:,left:right,up:down]))
                    if curr_pred > 0.5:
                        predicted = 1
                        break
            else:
                target_shape = tuple(min(c, m) for c, m in zip(images.shape, (1,6,256,256)))
                images= F.interpolate(images, size=target_shape[2:], mode='bilinear', align_corners=False)
                predicted = int(torch.sigmoid(model(images)) > 0.5)
            # Store results
            predictedcurr = predicted
            labelscurr = labels.cpu().numpy()
            all_predictions.append(predictedcurr)
            all_labels.extend(labelscurr)


    #6.2: Test Accuracy
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report
    print('-'*60)
    print(k/1000)
    test_accuracy = accuracy_score(all_labels,all_predictions)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        best_thres = k

    #6.3: Classification Report
    #gives precision, recall, and F1
    print(classification_report(all_labels,all_predictions))


missed_imgs = []
with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader):
        images = images.to("mps")
        labels = labels.to("mps")

        target_shape = tuple(max(c, m) for c, m in zip(images.shape, (1,6,256,256)))
        images= F.interpolate(images, size=target_shape[2:], mode='bilinear', align_corners=False)

        target_shape = tuple(min(c, m) for c, m in zip(images.shape, (1,6,256,256)))
        images= F.interpolate(images, size=target_shape[2:], mode='bilinear', align_corners=False)
        predicted = int(torch.sigmoid(model(images)) > 0.5)
        
        percentage_on = torch.count_nonzero(images[0,3,:,:])/images[0,3,:,:].numel()
        if predicted == 0 :
            # Get model outputs
            #1. GET BOUNDING BOXES
            bboxes = get_slice_bboxes(images.shape[2], images.shape[3], 256, 256)
            predicted = 0
            for i in bboxes:
                left = images.shape[2] - 256 if i[2] > images.shape[2] - 1 else i[0]
                right = min(i[2], images.shape[2])
                up = images.shape[3] - 256 if i[3] > images.shape[3] - 1 else i[1]
                down = min(i[3], images.shape[3])
                if down-up != 256:
                    print(down-up)
                    print(i)
                    print(images.shape)
                    print(down)
                    print(up)
                curr_pred = torch.sigmoid(model(images[:,:,left:right,up:down]))
                if curr_pred > 0.5:
                    predicted = 1
                    break
        # Store results
        predictedcurr = predicted
        labelscurr = labels.cpu().numpy()
        all_predictions.append(predictedcurr)
        all_labels.extend(labelscurr)


#6.2: Test Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
test_accuracy = accuracy_score(all_labels,all_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#6.3: Classification Report
#gives precision, recall, and F1
print(classification_report(all_labels,all_predictions))

#6.4: Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# Display the counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i, j]),ha="center",va="center",color="black",fontsize=12)

plt.colorbar()
plt.tight_layout()
plt.show()