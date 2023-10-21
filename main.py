import cv2
import numpy as np
from multiprocessing import Process


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


def start_cap():
    cap = cv2.VideoCapture(0)

    return cap


def detect_light_and_pose(cap):
    ret, _ = cap.read()
    if not ret:
        print("No First Frame.")
        return
    
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    # net = cv2.dnn.readNetFromCaffe("pose_deploy.prototxt", "pose_iter_264000.caffemodel")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        threshold = 200 
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            moments = cv2.moments(max_contour)
            cx = int(moments["m10"] / moments["m00"]) if moments["m00"] else 0
            cy = int(moments["m01"] / moments["m00"]) if moments["m00"] else 0

            print(f"{cx} + {cy}")
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)


        # Getting the pose detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()

        output = output[:, :19, :, :]

        height = frame.shape[0]
        width = frame.shape[1]

        points = []
        for i in range(19):
            body_part = output[0, i, :, :]

            print(cv2.minMaxLoc(body_part))
            (_, val, _, point) = cv2.minMaxLoc(body_part)
            x = (width * point[0]) / output.shape[3]
            y = (height * point[1]) / output.shape[2]

            points.append((int(x), int(y)) if val > 0.15 else None)
            

        for pair in POSE_PAIRS:
            if points[BODY_PARTS[pair[0]]] and points[BODY_PARTS[pair[1]]]:
                cv2.line(frame, points[BODY_PARTS[pair[0]]], points[BODY_PARTS[pair[1]]], (255, 0, 0), 10)
                cv2.circle(frame, points[BODY_PARTS[pair[0]]], 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, points[BODY_PARTS[pair[1]]], 10, (0, 0, 255), cv2.FILLED)

        cv2.imshow("yay", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    cap = start_cap()
    detect_light_and_pose(cap)

    t1 = Process(target=detect_light_and_pose)

    t1.start()

    t1.join()

