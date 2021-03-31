from unittest import TestCase
from Facenet_Recognition import load_encode_loc, cv2, load_known_faces


class TestFaceNetRecognition(TestCase):
    def test_no_detection_load_encode_loc(self):
        img = cv2.imread('test_image.jpg')
        kwn_names, kwn_encoding, status_list, since_list = load_known_faces('../known_faces.pt')
        box_list, name_list = load_encode_loc(img, kwn_names, kwn_encoding)
        self.assertTrue(len(box_list) == 0)

    def test_detection_load_encode_loc(self):
        img = cv2.imread('test_princy.jpg')
        kwn_names, kwn_encoding, status_list, since_list = load_known_faces('../known_faces.pt')
        box_list, name_list = load_encode_loc(img, kwn_names, kwn_encoding)
        print(len(box_list))
        self.assertTrue(len(box_list) > 0)
