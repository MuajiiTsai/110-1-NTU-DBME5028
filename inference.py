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
from torchvision.io import read_image, ImageReadMode

parser = argparse.ArgumentParser()
parser.add_argument("-data", "--datapath")
parser.add_argument("-output", "--output_csv_path")
args = parser.parse_args()

model_path = os.path.join(args.datapath, "model_with_nonlabel_training (1).pt")

transform = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  # transforms.PILToTensor(),
  transforms.Resize([512,512]),
  # transforms.ToPILImage(mode='RGB')
])

load_model = torch.load(model_path)
# print(type(load_model))
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
model.load_state_dict(load_model)
model = model.cuda()

output_filename = 'output_with_seed'
test_path = os.path.join(args.datapath, "test")
output_csv = os.path.join(args.datapath, f'{output_filename}.csv')
model.eval()
# model.cpu()
test_data = os.listdir(test_path)
datalist = pd.DataFrame(test_data, columns=['patient_id'])
datalist.set_index('patient_id')
test_output = pd.DataFrame(columns= ['id', 'label'])

for image_list in tqdm(test_data):
  if(image_list[-3:] != 'png'):
    continue
  image = read_image(os.path.join(test_path, image_list), mode=ImageReadMode.RGB)
  image = transform(image)
  image = image.reshape(1, 3, 512, 512)
  image = image.type(torch.FloatTensor)
  image = image.cuda()
  label_pred = model(image)
  label_pred = label_pred.flatten()
  label_pred = label_pred.cpu().detach().numpy()
  new = pd.DataFrame({
          'id': image_list,
          'label': label_pred
      })
  
  test_output = test_output.append(new, ignore_index=True)
  test_output.to_csv(output_csv, index=False)

test_output = pd.read_csv(os.path.join(args.datapath, "output_with_seed.csv"))


# merge label
test_output_merge = pd.DataFrame(columns= ['id', 'label'])
for i in range(1, len(test_output)-1):
  # Comparison
  patient = test_output.loc[i, 'id'][:-11]
  if (test_output_merge['id'] == patient).any() == True:
    test_output_merge.set_index('id', inplace=True)
    if test_output.loc[i, 'label'] > test_output_merge.loc[patient, 'label']:
      test_output_merge.loc[patient, 'label'] = test_output.loc[i, 'label']
    test_output_merge.reset_index(inplace=True)
  else:
    temp = pd.DataFrame({
        'id': test_output.loc[i, 'id'][:-11],
        'label': [test_output.loc[i, 'label']]
    })
    test_output_merge = test_output_merge.append(temp, ignore_index=True)

test_output_merge.to_csv(args.output_csv_path, index=False)