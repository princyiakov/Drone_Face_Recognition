from djitellopy import Tello
import numpy as np
from facenet_recognition import *
import dlib
import pygame


class MyDrone(Tello):
    def __init__(self, location):
        super().__init__()
        self.loc = location
        pygame.init()
        self.win = pygame.display.set_mode((400, 400))

        # Initialize Tello Drone
        self.connect()
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 0
        self.streamoff()
        self.streamon()
        print(self.get_battery())
        print('Cuda Devices', dlib.cuda.get_num_devices())

        # Load known faces for recognition
        self.kwn_names, self.kwn_encoding, self.status_list, self.since_list = load_known_faces(
            self.loc)

    def get_fame(self, w, h):
        frame = self.get_frame_read()
        frame = frame.frame
        frame = cv2.resize(frame, (w, h))

        return frame

    def detect_face(self, img):

        # Fetch face location from the frame with 128 encoding of face landmarks
        curr_face_loc, name_list, info_list = load_encode_loc(img, self.kwn_names,
                                                              self.kwn_encoding,
                                                              self.status_list, self.since_list)
        print('Current value is ', curr_face_loc, name_list)
        face_list = []
        face_area = []
        print('face loc', curr_face_loc)
        if len(curr_face_loc):

            for (top, right, bottom, left), name in zip(curr_face_loc, name_list):
                print(top, right, bottom, left)
                cv2.rectangle(img, (top, right), (bottom, left), (0, 255, 2), 2)

                w = right - left
                h = bottom - top
                cx = left + w // 2
                cy = top + h // 2
                area = w * h

                for idx, info in enumerate(info_list):
                    cv2.putText(img, info, (bottom, int(left * idx * 0.2)),
                                cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 255), 1)

                face_list.append([cx, cy])
                face_area.append(area)

                i = face_area.index(max(face_area))

            return img, [face_list[i], face_area[i]]

        else:
            return img, [[0, 0], 0]

    def get_key(self, key_val):
        res = False
        for event in pygame.event.get():
            pass
        key_ip = pygame.key.get_pressed()
        my_key_ip = getattr(pygame, 'K_{}'.format(key_val))
        if key_ip[my_key_ip]:
            res = True

        pygame.display.update()

        return res

    def get_keyboard_control(self):
        speed = 30

        # Set the speed based on the keyboard input
        if self.get_key("LEFT"):
            self.left_right_velocity = - speed
        elif self.get_key("RIGHT"):
            self.left_right_velocity = speed

        elif self.get_key("UP"):
            self.for_back_velocity = speed
        elif self.get_key("DOWN"):
            self.for_back_velocity = - speed

        elif self.get_key("w"):
            self.yaw_velocity = speed
        elif self.get_key("r"):
            self.yaw_velocity = - speed

        elif self.get_key("a"):
            self.up_down_velocity = speed
        elif self.get_key("z"):
            self.up_down_velocity = - speed

        if self.get_key("q"):
            self.land()

        if self.get_key("t"):
            self.takeoff()

        if self.send_rc_control:
            self.send_rc_control(self.left_right_velocity,
                                 self.for_back_velocity,
                                 self.up_down_velocity,
                                 self.yaw_velocity)

        # Reset the values
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
