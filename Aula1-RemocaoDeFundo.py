import numpy as np
import cv2


VIDEO = 'D:/igor-/Documents/Programmazione/Alura/Data Science/69. Visão Computacional detecção de movimento com OpenCV/Rua.mp4'

cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = [] 

for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)
    
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# cv2.imshow('Median frame', medianFrame)
# cv2.waitKey(0)
cv2.imwrite('output/img/Median frame.jpg', medianFrame)

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('Acabou os frames')
        break
    
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', frameGray)
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
    
cap.release()