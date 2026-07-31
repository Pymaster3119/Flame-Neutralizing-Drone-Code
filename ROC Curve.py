import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from sklearn.metrics import roc_curve, auc
import numpy as np
import os


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
model = model.to("cpu")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import tqdm

img_width = 256
img_height = 256

class FASDDDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        dataset_dir = "FireDataset_1_InAndOutdoor"
        self.images = []

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

transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
test_dataset = FASDDDataset(transform)
test_loader = DataLoader(test_dataset,batch_size=128,shuffle=True)
images, labels = next(iter(test_loader))
images = images.to("cpu")

print(f"Number of images: {len(test_dataset)}")
# Make predictions
with torch.no_grad():
    outputs = model(images)
    # Convert logits to probabilities
    probabilities = torch.sigmoid(outputs) #replace with "torch.softmax(outputs,dim=1)" if multi-class
    # Get predicted class and confidence
    predictions = (probabilities >= 0.5).float()
    #replace above line with "confidence, predictions = torch.max(probabilities,dim=1)" if multi-class

#Format properly (only for binary)
probabilities = probabilities.squeeze()
predictions = predictions.squeeze()

# Move data back to CPU for plotting
images = images.cpu()
predictions = predictions.cpu()
confidence = probabilities.cpu()
labels = labels.cpu()


#6.1: Predictions on test set
# Store predictions and true labels
all_predictions = []
all_labels = []
all_probas = []

# Disable gradient calculations
with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader):
        images = images.to("cpu")
        labels = labels.to("cpu")

        # Get model outputs
        outputs = model(images)

        # Convert outputs into class predictions

        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities >= 0.5).float()
        #"_, predictions = torch.max(outputs,1)" replaces above two lines if multi-class

        # Store results
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probas.extend(probabilities.cpu().numpy())

#6.2: Test Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

test_accuracy = accuracy_score(all_labels,all_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print(len(all_probas))
print(len(test_loader))
fpr, tpr, thresholds = roc_curve(all_labels, all_probas, drop_intermediate=False)
print(thresholds)

roc_auc = auc(fpr, tpr)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal guessing line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

#calculate best threshold
best_threshold = -10.0
best_acc = -10.0
for j in tqdm.trange(0,10000, 1):
    i = j / 10000
    outputs = []
    for k in all_probas:
        outputs.append(1 if k>i else 0)
    test_accuracy = accuracy_score(all_labels,outputs)
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        best_threshold = i

print(best_acc)
print(best_threshold)

#6.1: Predictions on test set
# Store predictions and true labels
all_predictions = []
all_labels = []

# Disable gradient calculations
missed_imgs = []
with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader):
        images = images.to("cpu")
        labels = labels.to("cpu")

        # Get model outputs
        outputs = model(images)

        # Convert outputs into class predictions

        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities >= 0.5).float()
        #"_, predictions = torch.max(outputs,1)" replaces above two lines if multi-class

        # Store results
        predictedcurr = predicted.cpu().numpy()
        labelscurr = labels.cpu().numpy()
        all_predictions.extend(predictedcurr)
        all_labels.extend(labelscurr)

        for i in range(len(predictedcurr)):
            pred = predictedcurr[i]
            label = labelscurr[i]
            print(int(pred[0]))
            if int(pred[0]) != int(label):
                missed_imgs.append((images.cpu().numpy()[i], int(pred[0]), int(label)))


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


#Display misses
print(len(missed_imgs))
for i in range(0, len(missed_imgs) - 8, 8):
    images = missed_imgs[i:i+8]
    plt.figure(figsize=(12, 8))


    for j in range(8):

        plt.subplot(2, 4, j + 1)

        # Change image from (channels, height, width)
        # to (height, width, channels)
        image = np.transpose(images[j][0][0:3,:,:], (1, 2, 0)) # We do this because PyTorch stores images as channel x height x width

        plt.imshow(image)
        predicted_class = int(images[j][1])
        actual_class = int(images[j][2])

        # If prediction is class 0, flip confidence
        # because sigmoid gives probability of class 1

        plt.title(
            f"Pred: {predicted_class}\n"
            f"True: {actual_class}\n")

        plt.axis("off")

    plt.tight_layout()
    plt.show()