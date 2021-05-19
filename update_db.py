import requests
import pandas as pd

from facenet_pytorch import MTCNN, InceptionResnetV1

import cv2
import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

# Load data

# fetch data from database through API deployed
data = requests.get('http://127.0.0.1:5000/missingperson')
df = pd.json_normalize(data.json())  # convert into dataframe
print(df)

# Initializing for face detection
mtcnn = MTCNN(image_size=240, margin=0, device='cuda', keep_all=False)
resnetv1 = InceptionResnetV1(pretrained='vggface2').eval()

for index, row in df.iterrows():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'Images', str(row['id']) + '.jpg')
    img = cv2.imread(path, 1)
    face = mtcnn(img)
    emb = resnetv1(face.unsqueeze(0))
# for img, idx in loader:
#     face = mtcnn(img)  # noticed mtcnn doesnt recognise for images more than 1MB
#     emb = resnetv1(
#         face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
#     embedding_list.append(emb.detach())  # result in embedding matrix is stored in a list
#     if idx == info.iloc[idx, 0]:
#         name_list.append(info.iloc[idx, 2])  # names are stored in a list
#         status_list.append(info.iloc[idx, 3]) # status are stored in a list
#         since_list.append(info.iloc[idx, 4]) # date of the status update are stored in a list
#
# data = [name_list, embedding_list, status_list, since_list]
# torch.save(data, 'known_faces.pt')  # saving the trained model and the database in .pt file
#
# print('Successfully trained the data ')
