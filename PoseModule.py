import cv2
import mediapipe as mp
import time
import math


class poseDetector():

    def __init__(self, mode=False, complexity=1, upBody=False, segmentation=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.upBody = upBody
        self.segmentation = segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.upBody, self.segmentation, self.smooth,
                                     self.detectionCon, self.trackCon)

    def find_pose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        # id=Körperteile: siehe https://google.github.io/mediapipe/solutions/pose.html
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw: cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist

    def angle_between_two_points(self, img, point_a=0, point_b=1, draw=True):
        if self.results.pose_landmarks:
            lma = self.results.pose_landmarks.landmark[point_a]
            lmb = self.results.pose_landmarks.landmark[point_b]

            #Schnittwinkel zwischen den beiden Vektoren berechnen
            counter = (lma.x * lmb.x + lma.y * lmb.y + lma.z * lmb.z)
            denominator = math.sqrt(lma.x**2 + lma.y**2 + lma.z**2)*math.sqrt(lmb.x ** 2 + lmb.y ** 2 + lmb.z ** 2)
            angle_rad = math.acos(counter/denominator) #Winkel in Bogenmass
            angle_deg = int(angle_rad * 180/math.pi) #Winkel in Grad

            #Berechnung der Punkte in Pixel anhand der Abmessungen des Fensters
            h, w, c = img.shape
            ax, ay = int(lma.x * w), int(lma.y * h)
            bx, by = int(lmb.x * w), int(lmb.y * h)
            if draw:
                cv2.circle(img, (ax, ay), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (bx, by), 5, (255, 0, 0), cv2.FILLED)
            return angle_deg

    def angle_between_three_points(self, img, point_a=0, point_b=1, point_c=2, draw=True):
    #point_a ist der Schnittpunkt der beiden Punkte point_b und point_c

        if self.results.pose_landmarks:
            lma = self.results.pose_landmarks.landmark[point_a]
            lmb = self.results.pose_landmarks.landmark[point_b]
            lmc = self.results.pose_landmarks.landmark[point_c]

            bx, by, bz = lmb.x - lma.x, lmb.y-lma.y, lmb.z-lma.z
            cx, cy, cz = lmc.x - lma.x, lmc.y-lma.y, lmc.z-lma.z
            counter = bx * cx + by * cy + bz * cz
            denominator = math.sqrt(bx ** 2 + by ** 2 + bz ** 2) * math.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
            angle_rad = math.acos(counter/denominator) #Winkel in Bogenmass
            angle_deg = int(angle_rad * 180/math.pi)


            #Berechnung der Punkte in Pixel anhand der Abmessungen des Fensters
            h, w, c = img.shape
            ax, ay = int(lma.x * w), int(lma.y * h)
            bx, by = int(lmb.x * w), int(lmb.y * h)
            cx, cy = int(lmc.x * w), int(lmc.y * h)
            if draw:
                cv2.circle(img, (ax, ay), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (bx, by), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return angle_deg


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        #lmlist = detector.find_position(img)
        #angle = detector.angle_between_two_points(img, 13, 15)
        angle = detector.angle_between_three_points(img, 13, 11, 15)
        #print(lmlist)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS: ' + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, 'Winkel: ' + str(angle), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        # Press Escape to Exit
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
