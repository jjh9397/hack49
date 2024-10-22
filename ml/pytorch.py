from pathlib import Path
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.preprocessing import robust_scale, minmax_scale, scale
import wandb
import gc
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import signal
import mne
import os
from dotenv import load_dotenv


class EEGDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # subj = row["subj"]
        epochs = row["epoch"]   # (n_channels, n_times)
        labels = row["label"]

        # add 3 channels for use with 2d CNN
        epochs = epochs[np.newaxis, :, :]
        epochs = np.concatenate([epochs,epochs,epochs], axis=0)
        epochs = np.transpose(epochs, (1,2,0))

        if self.transform is not None:
            epochs = self.transform(image=epochs)["image"]

        epochs = np.transpose(epochs, (2,0,1))

        return torch.tensor(epochs).float(), torch.tensor(labels).float()

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.net.classifier[1] = nn.Linear(1280, 1)
    
    def forward(self, eeg):
        return self.net(eeg)

def buildDatasets():
    df = pd.read_pickle("ml/dataset.pkl")
    labels = df["label"]
    subjs = df['subj']
    print(subjs)

    train_idx, val_idx = train_test_split(df.index, test_size=0.2, shuffle=True, stratify=labels)

    transforms = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5)
    ])
    dataset = EEGDataset(df, transforms)
    trainset = Subset(dataset,  train_idx)
    valset = Subset(dataset, val_idx)

    return trainset, valset

def buildDataloaders(trainset, valset, batch_size):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    
    return trainloader, valloader

def make(config):
    trainset, valset = buildDatasets()
    trainloader, valloader = buildDataloaders(trainset, valset, config.bsize)

    # net = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    # net = net.to(device)
    # net.classifier[1] = nn.Linear(1280, 1).to(device)
    net = EEGNet().to(device)

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, fused=True)

    wandb.watch(net, crit, log="all", log_freq=16)
    return trainloader, valloader, net, crit, opt

def train(config, trainloader, valloader, net, crit, opt):
    length = len(trainloader)
    vlength = len(valloader)
    bsize = config.bsize
    examples = 0
    
    scheduler = OneCycleLR(
        opt,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    
    for epoch in range(config.epochs):
        net.train()
        
        running_loss = 0.0
        for epochs, labels in trainloader:
            epochs = epochs.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = net(epochs).squeeze(1)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()
            scheduler.step()
            running_loss += loss.item()
            examples += bsize

        avg_loss = running_loss / length
        wandb.log({"train": {"epoch": epoch, "avg_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}}, step=examples)

    # After all epochs are done, evaluate final validation metrics
    net.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    false_positives = 0  # Initialize false positives counter
    true_positives = 0   # Initialize true positives counter
    true_negatives = 0   # Initialize true negatives counter
    false_negatives = 0  # Initialize false negatives counter

    with torch.inference_mode():
        for epochs, labels in valloader:
            epochs = epochs.to(device)
            labels = labels.to(device)

            outputs = net(epochs).squeeze(1)
            
            # Compute loss
            loss = crit(outputs, labels)
            running_loss += loss.item()
            
            # Convert the model output to predictions (binary classification)
            preds = torch.round(torch.sigmoid(outputs))
            
            # Calculate correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Calculate false positives (predicted = 1, actual = 0)
            false_positives += ((preds == 1) & (labels == 0)).sum().item()

            # Calculate true negatives (predicted = 0, actual = 0)
            true_negatives += ((preds == 0) & (labels == 0)).sum().item()

            # Calculate true positives (predicted = 1, actual = 1)
            true_positives += ((preds == 1) & (labels == 1)).sum().item()

            # Calculate false negatives (predicted = 0, actual = 1)
            false_negatives += ((preds == 0) & (labels == 1)).sum().item()

            print(f'[{epoch}, val] loss: {avg_loss:.3f}')

    # Final accuracy calculation
    accuracy = correct / total

    # Final false positives and true negatives calculation
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    # Sensitivity (true positive rate)
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    avg_loss = running_loss / vlength
    
    print(f'Final validation results:')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'False Positives: {false_positives}')
    print(f'Specificity (True Negative Rate): {specificity:.3f}')
    print(f'Sensitivity (True Positive Rate): {sensitivity:.3f}')

    wandb.log({
        "val": {
            "final_accuracy": accuracy,
            "final_false_positives": false_positives,
            "final_specificity": specificity,
            "final_sensitivity": sensitivity
        }
    })

    return net

if __name__ == "__main__":
    load_dotenv()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = 0
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = {
        "bsize": 32,
        "epochs": 16,
        "lr": 1e-3,
        "max_lr": 1e-3,
    }

    key = os.getenv('WANDB_KEY')
    wandb.login(key=key)

    with wandb.init(project="hack49", config=config, mode="online"):
        config = wandb.config
        params = make(config)

        net = train(config, *params)

        wandb.finish()

        torch.save(net.state_dict(), "model.pt")
        # model = EEGNet()
        # model.load_state_dict(torch.load("model.pt", weights_only=True))
        # model.eval()
        # predictions = F.sigmoid(model(epochs))

###

# eeg_net = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
# eeg_net = eeg_net.to(device)
# eeg_net.classifier[1] = nn.Linear(1280, 1).to(device)

# df = pd.read_pickle("dataset.pkl")
# dataset = EEGDataset(df)
# epoch, label = dataset[0]
# print(f"Epoch shape: {epoch.shape}")
# print(f"Epoch: {epoch}")

# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# correct = 0
# total = 0

# for epochs, labels in dataloader:
#     epochs = epochs.to(device)
#     labels = labels.to(device)

#     out = F.sigmoid(eeg_net(epochs))
#     labels = ["Epilepsy" if label == 1 else "Healthy" for label in labels]
#     print(f"Label: {labels}")
#     out = ["Epilepsy" if (pred > 0.5) else "Healthy" for pred in out]
#     print(f"Output: {out}")
#     print()
#     print(f"Comparison: {list(zip(label, out))}")
    
#     correct += sum([1 for true, pred in zip(label, out) if true == pred])
#     total += len(labels)

#     break

# accuracy = correct / total * 100
# print(f"Accuracy: {accuracy:.2f}%")