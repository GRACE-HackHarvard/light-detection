import cv2
import numpy as np

def detect_light_changes():
    cap = cv2.VideoCapture(0)

    ret, _ = cap.read()
    if not ret:
        print("No First Frame.")
        return
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        threshold = 150 
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            moments = cv2.moments(max_contour)
            cx = int(moments["m10"] / moments["m00"]) if moments["m00"] else 0
            cy = int(moments["m01"] / moments["m00"]) if moments["m00"] else 0

            print(f"{cx} + {cy}")
            cv2.circle(gray, (cx, cy), 10, (0, 255, 0), -1)

        cv2.imshow("yay", gray)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_light_changes()
