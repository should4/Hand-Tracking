import cv2
import mediapipe as mp
import time

__name__ = "__main__"


class handDetector():
    def __init__(self, mode=False, maxHands=2, Complexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.Complexity = Complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        # 选择 hands 模型
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.Complexity,
                                        self.detectionCon,
                                        self.trackingCon)  # min_detection_confidence=0.98  / min_tracking_confidence=0.1 / model_complexity=0
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # 如果图片中有手掌 则返回手掌上的标志点 否则 返回 none
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
        # 将每个手掌上的21个标志点打印出来
        # for index, lm in enumerate(handLms.landmark):
        #     xPos = int(lm.x * imgWidth)
        #     yPos = int(lm.y * imgHeight)
    def findPosition(self, img, handIndex=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handIndex]
            for index, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([index, cx, cy])
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)

        lmList = detector.findPosition(img)
        
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 100, 150))

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(__name__)
