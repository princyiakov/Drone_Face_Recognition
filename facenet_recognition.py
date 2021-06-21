import cv2

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image
import pandas as pd
import requests
import numpy as np
import json


def load_known_faces(location):
    data = torch.load(location)
    img_names, encoded_images, status, since = data[0], data[1], data[2], data[3]

    return img_names, encoded_images, status, since


def load_encode_loc(image, kwn_names, kwn_encoding, kwn_status_list, kwn_since_list):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mtcnn = MTCNN(keep_all=True, device='cuda:0')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img = Image.fromarray(img)
    box_list = []
    name_list = []
    info_list = []

    img_cropped_list, probs = mtcnn(img, return_prob=True)
    boxes, _ = mtcnn.detect(img)

    if img_cropped_list is not None:

        for i, (img, prob) in enumerate(zip(img_cropped_list, probs)):
            encode_img = resnet(img_cropped_list[i].unsqueeze(0)).detach()

            dist_list = []

            for encodng in kwn_encoding:
                dist = torch.dist(encode_img, encodng).item()
                dist_list.append(dist)

            min_dist = min(dist_list)
            print('min dis ', dist_list)

            if min_dist < 1.00:
                min_ind = dist_list.index(min_dist)
                name = kwn_names[min_ind]
                box = boxes[i].tolist()
                box = [int(x) for x in box]
                box_list.append(box)
                name_list.append(name)
                info_list.append(name)
                # info_list.append(name)
                info_list.append(kwn_status_list[min_ind])
                # info_list.append(kwn_since_list[min_ind])
    print ('Box list and name list ',box_list, name_list)
    return box_list, name_list, info_list
def cal_dist(curr_emb):
    dist = torch.dist(encode_img, curr_emb).item()
    return dist

def load_encode_loc_api(image):
    global encode_img
    # Fetch known faces data from database through API deployed
    data = requests.get('http://127.0.0.1:5000/missingperson')
    df = pd.json_normalize(data.json())  # convert into dataframe
    df['embedding'] = df['embedding'].map(lambda x: json.loads(x))
    df['embedding'] = df['embedding'].map(lambda x: torch.Tensor(x))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mtcnn = MTCNN(keep_all=True, device='cuda:0')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img = Image.fromarray(img)

    box_list = []
    name_list = []
    info_list = []

    img_cropped_list, probs = mtcnn(img, return_prob=True)
    boxes, _ = mtcnn.detect(img)

    if img_cropped_list is not None:

        for i, (img, prob) in enumerate(zip(img_cropped_list, probs)):
            encode_img = resnet(img_cropped_list[i].unsqueeze(0)).detach()
            df['image'] = encode_img
            df['dist'] = np.nan
            dist_list = []

            df['dist'] = df['embedding'].map(cal_dist)
            min_val = df['dist'].min()
            indx = df['dist'].idxmin()

            if min_val < 1.20:
                name = df['first_name'][indx]
                box = boxes[i].tolist()
                box = [int(x) for x in box]
                box_list.append(box)
                name_list.append(name)
                info_list.append(name)
                info_list.append(df['last_seen'][indx])
    print ('Box list and name list ',box_list, name_list)
    return box_list, name_list, info_list
