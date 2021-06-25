import os

import cv2
import pandas as pd
import requests
from facenet_pytorch import MTCNN, InceptionResnetV1
import json

# Load data

# fetch data from database through API deployed
data = requests.get('http://127.0.0.1:5000/missingperson')
df = pd.json_normalize(data.json())  # convert into dataframe
print(df)

# Initializing for face detection
mtcnn = MTCNN(image_size=240, margin=0, device='cuda', keep_all=False)
resnetv1 = InceptionResnetV1(pretrained='vggface2').eval()

for index, row in df.iterrows():
    data = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'Images', str(row['id']) + '.jpg')
    img = cv2.imread(path, 1)
    face = mtcnn(img)
    emb = resnetv1(face.unsqueeze(0))
    arr = emb.detach().cpu().numpy()
    arr = arr.tolist()
    arr = json.dumps(arr)
    data['embedding'] = arr
    data['id'] = row['id']
    req = requests.put('http://127.0.0.1:5000/missingperson', json=data)
    print(req.status_code)

