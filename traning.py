from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

# Initializing for face detection
mtcnn = MTCNN(image_size=240, margin=0, device='cuda')
resnetv1 = InceptionResnetV1(pretrained='vggface2').eval()

# Load data
dataset = datasets.ImageFolder('./Images_FaceNet')
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_wrapper(batch):
    return batch[0]


loader = DataLoader(dataset, collate_fn=collate_wrapper, pin_memory=True)

name_list = [] # list of known names
embedding_list = [] # list of embedding

for img, idx in loader:
    face = mtcnn(img)
    emb = resnetv1(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
    embedding_list.append(emb.detach())  # resulten embedding matrix is stored in a list
    name_list.append(idx_to_class[idx])  # names are stored in a list

data = [embedding_list, name_list]
torch.save(data, 'known_faces.pt') # saving data.pt file
