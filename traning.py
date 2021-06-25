from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

# Initializing for face detection
mtcnn = MTCNN(image_size=240, margin=0, device='cuda', keep_all=False)

resnetv1 = InceptionResnetV1(pretrained='vggface2').eval()

# Load data
dataset = datasets.ImageFolder('Images_FaceNet')
info = pd.read_csv('database.csv', header=None)


def collate_wrapper(x):
    return x[0]


loader = DataLoader(dataset, collate_fn=collate_wrapper)

name_list = []  # list of known names
embedding_list = []  # list of embedding
status_list = []  # list of current Status of the person
since_list = []  # list of date of the status reported of person

for img, idx in loader:
    face = mtcnn(img)  # noticed mtcnn doesnt recognise for images more than 1MB
    emb = resnetv1(
        face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
    embedding_list.append(emb.detach())  # result in embedding matrix is stored in a list
    if idx == info.iloc[idx, 0]:
        name_list.append(info.iloc[idx, 2])  # names are stored in a list
        status_list.append(info.iloc[idx, 3]) # status are stored in a list
        since_list.append(info.iloc[idx, 4]) # date of the status update are stored in a list

data = [name_list, embedding_list, status_list, since_list]
torch.save(data, 'known_faces.pt')  # saving the trained model and the database in .pt file


print('Successfully trained the data ')
