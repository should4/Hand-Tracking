import cv2
import mediapipe as mp
import time
pTime = 0
cTime = 0


cap = cv2.VideoCapture(0)
# 选择 hands 模型
mpHands = mp.solutions.hands
hands = mpHands.Hands() # min_detection_confidence=0.98  / min_tracking_confidence=0.1 / model_complexity=0
mpDraw = mp.solutions.drawing_utils
# 设定手掌点的特征
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255),thickness=5)
# 设定手掌连线特征
handConStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness=5)
while True:
    ret,img = cap.read()

    if ret:
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # 如果图片中有手掌 则返回手掌上的标志点 否则 返回 none
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                # 将每个手掌上的21个标志点打印出来
                for index, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    # cv2.putText(img,str(index), (xPos-25, yPos+5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
                    if index == 4:
                        cv2.circle(img, (xPos, yPos), 10, (100,56,105), cv2.FILLED)
                    print(index,xPos,yPos)

        # 显示图片的刷新率
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}" , (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,100,150))

        cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break