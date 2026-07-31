import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from sklearn.metrics import roc_curve, auc
import numpy as np


#3.1: Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,padding=0),
            nn.ReLU()
        )

        self.postskip2maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(262144, 128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32, 1) #put num_classes instead of 1 if using multi-class
        )

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        x = self.skip2(x)
        z = x + y
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

transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
test_dataset = datasets.ImageFolder(root="FireDataset_1_InAndOutdoor",transform=transform)
test_loader = DataLoader(test_dataset,batch_size=128,shuffle=True)
images, labels = next(iter(test_loader))
images = images.to("cpu")

print(f"Number of images: {len(test_dataset)}")
print(f"Number of classes: {len(test_dataset.classes)}")
print(f"Number of images in class 0: {sum([1 for _, label in test_dataset.samples if label == 0])}")
print(f"Number of images in class 1: {sum([1 for _, label in test_dataset.samples if label == 1])}")

print("\nClasses:")

for index, class_name in enumerate(test_dataset.classes):
    print(f"{index}: {class_name}") #CNNs see these classes as integers, which are then mapped back to classes

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
        predicted = (probabilities >= best_threshold).float()
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

#6.3: Classification Report
#gives precision, recall, and F1
print(classification_report(all_labels,all_predictions))


#Do test augmentation
#only on images where fire is detected
for n_samples in range(1,5):
    all_predictions = []
    all_labels = []
    all_probas = []

    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    val_dataset = datasets.ImageFolder(root="FireDataset_1_InAndOutdoor",transform=transform)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=True)
    resample_transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.ScaleJitter(target_size=(img_width, img_height), scale_range=(0.7, 1.3)), v2.Resize((img_width, img_height)),v2.ToTensor()]) #converts to channel x height x width

    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader):
            images = images.to("cpu")
            labels = labels.to("cpu")

            # Get model outputs
            outputs = model(images)

            # Convert outputs into class predictions
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities >= best_threshold).float()
            if (predicted[0].item() == 1):
                #Time to resample!
                numfires = 0
                for i in range(n_samples):
                    images = resample_transform(images)
                    outputs = model(images)
                    # Convert outputs into class predictions
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities >= best_threshold).float()[0].item()
                    if predicted == 1:
                        numfires+=1
                
                predicted = 1 if numfires>=0.5*n_samples else 0
                predicted = torch.tensor([[predicted]], dtype=torch.float32)


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

    #6.3: Classification Report
    #gives precision, recall, and F1
    print(classification_report(all_labels,all_predictions))