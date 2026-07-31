import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import load_data
import architecture
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

#3.1: Define CNN

model = architecture.SimpleCNN()
model.load_state_dict(torch.load("finetuned.pth"))
model = model.to("mps")

transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
mikeytrain = load_data.Dataset(True,False,False,True,False,False,transform)
mikeytest = load_data.Dataset(False,True,False,True,False,False,transform)
mikeyval = load_data.Dataset(False,False,True,True,False,False,transform)
mikeyall = load_data.Dataset(True, True, True, True, False, False, transform)

vedanttrain = load_data.Dataset(True,False,False,True,True,False,transform)
vedanttest = load_data.Dataset(False,True,False,True,True,False,transform)
vedantval = load_data.Dataset(False,False,True,True,True,False,transform)
vedantall = load_data.Dataset(True, True, True, True, True, False, transform)

adityatrain = load_data.Dataset(True,False,False,False,False,True,transform)
adityatest = load_data.Dataset(False,True,False,False,False,True,transform)
adityaval = load_data.Dataset(False,False,True,False,False,True,transform)
adityaall = load_data.Dataset(True, True, True, False, False, True, transform)

mikeytrainloader = DataLoader(mikeytrain, batch_size = 128, shuffle=False)
mikeytestloader = DataLoader(mikeytest, batch_size = 128, shuffle=False)
mikeyvalloader = DataLoader(mikeyval, batch_size = 128, shuffle=False)
mikeyallloader = DataLoader(mikeyall, batch_size = 128, shuffle=False)

vedanttrainloader = DataLoader(vedanttrain, batch_size = 128, shuffle=False)
vedanttestloader = DataLoader(vedanttest, batch_size = 128, shuffle=False)
vedantvalloader = DataLoader(vedantval, batch_size = 128, shuffle=False)
vedantallloader = DataLoader(vedantall, batch_size = 128, shuffle=False)

adityatrainloader = DataLoader(adityatrain, batch_size = 128, shuffle=False)
adityatestloader = DataLoader(adityatest, batch_size = 128, shuffle=False)
adityavalloader = DataLoader(adityaval, batch_size = 128, shuffle=False)
adityaallloader = DataLoader(adityaall, batch_size = 128, shuffle=False)

def display_outputs(loader):
    images, labels = next(iter(loader))
    images = images.to("mps")

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
    plt.figure(figsize=(12, 8))


    for i in range(8):

        plt.subplot(2, 4, i + 1)

        # Change image from (channels, height, width)
        # to (height, width, channels)
        image = images[i][0:3,:,:].permute(1, 2, 0) # We do this because PyTorch stores images as channel x height x width

        plt.imshow(image)

        predicted_class = int(predictions[i].item())
        actual_class = int(labels[i].item())

        confidence = probabilities[i].item()
        # If prediction is class 0, flip confidence
        # because sigmoid gives probability of class 1
        if predictions[i] == 0:
            confidence = 1 - confidence
        confidence_percent = confidence * 100

        plt.title(
            f"Pred: {predicted_class}\n"
            f"True: {actual_class}\n"
            f"Confidence: {confidence_percent:.1f}%")

        plt.axis("off")

    plt.tight_layout()
    plt.show()

def get_predictions(loader, threshold = 0.5):
    all_predictions = []
    all_labels = []
    all_probas = []

    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader):
            images = images.to("mps")
            labels = labels.to("mps")

            # Get model outputs
            outputs = model(images)

            # Convert outputs into class predictions

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities >= threshold).float()
            #"_, predictions = torch.max(outputs,1)" replaces above two lines if multi-class

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probabilities.cpu().numpy())

    return all_predictions, all_labels, all_probas


def get_metrics(all_predictions, all_labels):
    test_accuracy = accuracy_score(all_labels,all_predictions)

    #6.3: Classification Report
    #gives precision, recall, and F1
    class_report = classification_report(all_labels,all_predictions)
    return test_accuracy, class_report

def displays(all_predictions,all_labels, all_probas):
    #6.4: Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Use descriptive class labels on axes
    plt.xticks([0, 1], ["No Fire", "Fire"])
    plt.yticks([0, 1], ["No Fire", "Fire"])


    # Display the counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i, j]),ha="center",va="center",color="black",fontsize=12)

    plt.colorbar()
    plt.tight_layout()
    plt.show()

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

def calc_best_thresholds(all_predictions, all_labels, all_probas):
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

    return best_threshold


#display_outputs(mikeyallloader)

all_loaders = {
    # "mikey train": mikeytrainloader,
    # "mikey test": mikeytestloader,
    # "mikey val": mikeyvalloader,
    # "mikey all": mikeyallloader,

    #"vedant train": vedanttrainloader,
    "vedant test": vedanttestloader,
    "vedant val": vedantvalloader,
    #"vedant all": vedantallloader,

    # "aditya train": adityatrainloader,
    # "aditya test": adityatestloader,
    # "aditya val": adityavalloader,
    # "aditya all": adityaallloader,
}

def mainloop():
    output = ""
    sendout = ""
    thresholdg = 0.1185
    for i in all_loaders.keys():
        preds, labs, probas = get_predictions(all_loaders[i], threshold=0.5)
        metrics = get_metrics(preds, labs)
        output += "-" * 60 + "\n" + i + "\n"
        output += f"Accuracy: {metrics[0] * 100}% \n\n"
        output += metrics[1]
        output += "\n"
        if (i == list(all_loaders.keys())[1]):
            sendout = metrics[0]

        # threshold = thresholdg#calc_best_thresholds(preds, labs, probas)
        # #preds, labs, probas = get_predictions(all_loaders[i], threshold)
        # print(type(probas))
        # print(probas[0][0])
        # preds = [probas[i][0] > thresholdg for i in range(len(probas))]
        # metrics = get_metrics(preds, labs)
        # output += f"Thresholded: {threshold} \n"
        # output += f"Accuracy: {metrics[0] * 100}% \n\n"
        # output += metrics[1]
        # output += "\n"


        displays(preds, labs, probas)

    # with open("datasetoutputs.txt", "w", encoding="utf-8") as file:
    #     file.write(output)

    return sendout

if __name__ == "__main__":
    print(mainloop())
