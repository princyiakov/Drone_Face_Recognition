from drone_deep_learning_facenet import MyDrone, cv2
import calendar
import time
import os
import argparse


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    parser = argparse.ArgumentParser(description='Face Recognition Script')
    parser.add_argument('--model_loc', type=str,
                        default=os.path.join(current_dir, 'known_faces.pt'), required=False,
                        help='location of the trained model')
    args = parser.parse_args()
    drone = MyDrone(args.model_loc)

    p_error = 0
    p_up_dwn_error = 0
    p_for_back_error = 0
    pid = [0.5, 0, 0.5]
    # w, h = 360, 240
    w, h = 480, 360

    start_flight = 1  # Set 1 to not take off

    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    filenm = str(ts) + '.avi'
    out = cv2.VideoWriter(filenm, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))

    while True:
        if start_flight == 0:
            drone.takeoff()
            start_flight = 1
        img = drone.get_fame(w, h)
        img, info = drone.detect_face(img)
        print(info)
        p_error, p_up_dwn_error, p_for_back_error = drone.track_face(info, w, h, pid, p_error,
                                                                     p_up_dwn_error,
                                                                     p_for_back_error)
        out.write(img)
        cv2.imshow('Drone Output', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.land()
            break
    out.release()
    drone.land()


if __name__ == '__main__':
    main()
