from unittest import TestCase
import requests
import pandas as pd
from facenet_recognition import load_encode_loc_api, load_encode_loc, load_known_faces
from utils import singleimg_embedding
import cv2


class TestFaceNetRecognition(TestCase):
    def test_detection_load_encode_loc(self):
        image = cv2.imread('test_princy.jpg')
        box_list, name_list, info_list = load_encode_loc_api(image)
        self.assertTrue(len(box_list) > 0)

    def test_no_detection_load_encode_loc(self):
        img = cv2.imread('test_princy.jpg')
        kwn_names, kwn_encoding, status_list, since_list = load_known_faces('../known_faces.pt')
        box_list, name_list = load_encode_loc(img, kwn_names, kwn_encoding)
        print(len(box_list))
        self.assertTrue(len(box_list) > 0)

    def test_singleimg_embedding(self):
        img = cv2.imread('single.jpg')
        emb = singleimg_embedding(img)

        self.assertTrue(len(emb) > 0)
