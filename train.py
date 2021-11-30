import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import metrics
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
# from google.colab import drive
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
args = parser.parse_args()


model_path = args.datapath + "/brand_new_model.pt"

# train_path = '/content/drive/My Drive/Deep_Learning/Midterm/train/'
train_path = args.datapath + '/train'
# raw_label_file = '/content/drive/My Drive/Deep_Learning/Midterm/train.csv'
raw_label_file = args.datapath + '/train.csv'

img_labels = pd.read_csv(raw_label_file)
img_dir = os.listdir(train_path)

img_with_labels = pd.DataFrame(columns=['id', 'label'])
img_without_labels = pd.DataFrame(columns=['id', 'label'])
# add the row
img_labels.set_index('id', inplace=True)
for i in range(len(img_dir)):
    j = img_labels.loc[img_dir[i][:-11]][0]
    new = pd.DataFrame({
            'id': [img_dir[i]],
            'label': [j]
        })
    if not np.isnan(j):
        img_with_labels = img_with_labels.append(new, ignore_index=True)
    else:
        img_without_labels = img_without_labels.append(new, ignore_index=True)
img_with_labels.to_csv(args.datapath +'/img_with_labels.csv', index=False)
img_without_labels.to_csv(args.datapath + '/img_without_labels.csv', index=False)
img_labels.reset_index(inplace=True)

#seed
seed = 1000010

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(seed)

from torchvision.io import read_image, ImageReadMode

class BoneDataset(Dataset):
    def __init__(self, annotations_file, directory, transform=None, target_transform=None):
        self.dir = directory
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = os.listdir(directory)
        self.transform = transform
        self.target_transform = target_transform
        # self.item = self.classification()
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(os.path.join(self.dir, img_path), mode=ImageReadMode.RGB)
        image = image.type(torch.FloatTensor)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

label_file = args.datapath + '/img_with_labels.csv'
# train_path = '/content/drive/My Drive/Deep_Learning/Midterm/train/'


# merge csv

# pseudo_train_file = '/content/drive/My Drive/Deep_Learning/Midterm/pseudo_label.csv'

'''
pseudo_label = pd.read_csv(pseudo_train_file)
img_with_labels = pd.read_csv(label_file)
img_with_labels = img_with_labels.append(pseudo_label, ignore_index=True)

img_with_labels.to_csv('/content/drive/My Drive/Deep_Learning/Midterm/img_with_all_labels.csv', index=False)
img_with__all_labels_path = '/content/drive/My Drive/Deep_Learning/Midterm/img_with_all_labels.csv'
'''
#model
batch_size = 8
learning_rate = 1e-4
num_show = 3
transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  # transforms.PILToTensor(),
  transforms.Resize([512,512]),
  # transforms.ToPILImage(mode='RGB')
])


# Split All_train_dataset to train & validation

train_dataset = BoneDataset(label_file, train_path, transform=transform)

### Turn on must edit train_dataset to All_train_dataset ###
'''
train_dataset_size = int(len(All_train_dataset)*0.8)
validation_dataset_size = len(All_train_dataset)-train_dataset_size
print(validation_dataset_size)
train_dataset, validation_dataset = random_split(All_train_dataset, [train_dataset_size, validation_dataset_size])
'''
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle = True
)
'''
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle = False
)
'''

model = models.efficientnet_b0(pretrained=True)
model.conv1 = nn.Conv2d(
    1,
    64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False
)
model = nn.Sequential(model, nn.Linear(1000,1), nn.Sigmoid())
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss()

def AUC(y, y_pred):
# y, y_pred must be numpy array
  fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
  result = metrics.auc(fpr, tpr)
  print(result)
  return result

num_epochs = 8

#train the model
# best_valid_loss = np.inf

for idx_epoch in range(num_epochs):
    print(f"Epoch{idx_epoch}")

    #training phase
    model.train()

    train_losses = []
    for image_batch, label_batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        output_batch = model(image_batch)
        output_batch = torch.flatten(output_batch)
        loss = loss_fn(output_batch, label_batch)
        loss.backward()
        optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        train_losses.append(loss)

    train_loss = np.mean(train_losses)
    print(f"Training loss: {train_loss}")
    
    #validating phase
    # model.eval()

    # valid_losses = []
    # valid_accuracies = []
    # valid_label = []
    # valid_output = []
    # for image_batch, label_batch in tqdm(validation_dataloader):
    #     image_batch = image_batch.cuda()
    #     label_batch = label_batch.cuda()
    #     output_batch = model(image_batch)
    #     output_batch = torch.flatten(output_batch)
    #     # print(output_batch.shape)
    #     loss = loss_fn(output_batch, label_batch)
        
    #     loss = loss.detach().cpu().numpy()  
    #     valid_losses.append(loss)

    #     output_batch = output_batch.detach().cpu().numpy()
    #     label_batch = label_batch.detach().cpu().numpy()
    #     label_batch = np.bool(label_batch)
    #     # print(f"{output_batch}, {label_batch}")
    #     valid_label.append(label_batch)
    #     valid_output.append(output_batch)

    # ##### AUC #####
    # print(output_batch.shape)
    # auc = AUC(valid_label, valid_output)
    # valid_accuracies.append(auc)

    # valid_loss = np.mean(valid_losses)
    # valid_accuracy = np.mean(valid_accuracies)
    # print(f"Testing loss: {valid_loss}")

    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(model.state_dict(), model_path)
    print("The model is saved")


label_file = args.datapath + '/img_with_labels.csv'


# merge csv

pseudo_train_file = args.datapath + '/pseudo_label.csv'

pseudo_label = pd.read_csv(pseudo_train_file)
img_with_labels = pd.read_csv(label_file)
img_with_labels = img_with_labels.append(pseudo_label, ignore_index=True)

img_with_labels.to_csv(args.datapath + '/img_with_all_labels.csv', index=False)
img_with__all_labels_path = args.datapath + '/img_with_all_labels.csv'

#model
batch_size = 8
learning_rate = 1e-4
num_show = 3
transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  # transforms.PILToTensor(),
  transforms.Resize([512,512]),
  # transforms.ToPILImage(mode='RGB')
])


# Split All_train_dataset to train & validation

train_dataset = BoneDataset(label_file, train_path, transform=transform)

### Turn on must edit train_dataset to All_train_dataset ###
'''
train_dataset_size = int(len(All_train_dataset)*0.8)
validation_dataset_size = len(All_train_dataset)-train_dataset_size
print(validation_dataset_size)
train_dataset, validation_dataset = random_split(All_train_dataset, [train_dataset_size, validation_dataset_size])
'''
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle = True
)
'''
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle = False
)
'''

num_epochs = 8

#train the model
# best_valid_loss = np.inf

for idx_epoch in range(num_epochs):
    print(f"Epoch{idx_epoch}")

    #training phase
    model.train()
    
    train_losses = []
    for image_batch, label_batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        output_batch = model(image_batch)
        output_batch = torch.flatten(output_batch)
        loss = loss_fn(output_batch, label_batch)
        loss.backward()
        optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        train_losses.append(loss)

    train_loss = np.mean(train_losses)
    print(f"Training loss: {train_loss}")
    
    #validating phase
    # model.eval()

    # valid_losses = []
    # valid_accuracies = []
    # valid_label = []
    # valid_output = []
    # for image_batch, label_batch in tqdm(validation_dataloader):
    #     image_batch = image_batch.cuda()
    #     label_batch = label_batch.cuda()
    #     output_batch = model(image_batch)
    #     output_batch = torch.flatten(output_batch)
    #     # print(output_batch.shape)
    #     loss = loss_fn(output_batch, label_batch)
        
    #     loss = loss.detach().cpu().numpy()  
    #     valid_losses.append(loss)

    #     output_batch = output_batch.detach().cpu().numpy()
    #     label_batch = label_batch.detach().cpu().numpy()
    #     label_batch = np.bool(label_batch)
    #     # print(f"{output_batch}, {label_batch}")
    #     valid_label.append(label_batch)
    #     valid_output.append(output_batch)

    # ##### AUC #####
    # print(output_batch.shape)
    # auc = AUC(valid_label, valid_output)
    # valid_accuracies.append(auc)

    # valid_loss = np.mean(valid_losses)
    # valid_accuracy = np.mean(valid_accuracies)
    # print(f"Testing loss: {valid_loss}")

    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(model.state_dict(), model_path)
    print("The model is saved")