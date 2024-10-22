
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from ml.utils import userFileToPd
import matplotlib.pyplot as plt
import mne


class EEGDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df
        print("Row: ", row)
        epochs = row["epoch"]   # (n_channels, n_times)

        # add 3 channels for use with 2d CNN
        epochs = epochs[np.newaxis, :, :]
        epochs = np.concatenate([epochs,epochs,epochs], axis=0)
        epochs = np.transpose(epochs, (1,2,0))

        if self.transform is not None:
            epochs = self.transform(image=epochs)["image"]

        epochs = np.transpose(epochs, (2,0,1))

        return torch.tensor(epochs).float()
    
class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.net.classifier[1] = nn.Linear(1280, 1)
    
    def forward(self, eeg):
        return self.net(eeg)
    
def buildData():
    df = pd.read_pickle('ml/testDataset.pkl')
    print(df)
    df = df.iloc[0]
    test_data = EEGDataset(df)

    return test_data


def visualize_eeg(arr: np.array):
    # (n_channels, n_times)
    print('arrshape: ', arr.shape)
    n_channels = 7
    ch_names = ['Fp1', 'Cp5', 'Cp6', 'C3', 'C4', 'O1', 'O2']
    ch_types = ["eeg"] * n_channels
    sf = 256

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sf)

    raw = mne.io.RawArray(arr, info)
    fig = raw.plot(show=False)
    fig.savefig("userfiles/eeg.png")
    plt.close(fig)

def evaluateFile(user_filepath):
    print(f"I got filepath: {user_filepath}\n")
    df = userFileToPd(user_filepath)
    print(df)
    df = df.iloc[0]
    data = EEGDataset(df)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    eeg_net = EEGNet()
    eeg_net.load_state_dict(torch.load('ml/model.pt'))
    eeg_net.eval()

    eeg_net = eeg_net.to(device)

    data = EEGDataset(df)

    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    for epochs in data_loader:
        # visualize_eeg(epochs.squeeze(0)[0, :, :])
        epochs = epochs.to(device)
        print(f"Epoch shape: {epochs.shape}")

        pred = F.sigmoid(eeg_net(epochs)).squeeze(1)

        pred_str = ["Epilepsy" if (p > 0.5) else "Healthy" for p in pred]
        val = [float(p) for p in pred]
        print(f"Output: {pred}")

        return pred_str, round(val[0], 2)

    raise Exception("No data in data_loader")

if __name__ == "__main__":
    # buildData()
    # exit()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    eeg_net = EEGNet()
    eeg_net.load_state_dict(torch.load('ml/model.pt'))
    eeg_net.eval()

    eeg_net = eeg_net.to(device)

    # df = pd.read_pickle("dataset.pkl")
    data = buildData()

    # epoch = data[0]
    # print(f"Epoch shape: {epoch.shape}")
    # print(f"Epoch: {epoch}")

    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    for epochs in data_loader:
        epochs = epochs.to(device)
        print(f"Epoch shape: {epochs.shape}")

        out = F.sigmoid(eeg_net(epochs)).squeeze(1)
        # labels = ["Epilepsy" if label == 1 else "Healthy" for label in labels]
        # print(f"Label: {labels}")
        out = ["Epilepsy" if (pred > 0.5) else "Healthy" for pred in out]
        print(f"Output: {out}")
        # print()
        
        # correct += sum([1 for true, pred in zip(label, out) if true == pred])
        # total += len(labels)

        break

    # accuracy = correct / total * 100
    # print(f"Accuracy: {accuracy:.2f}%")
