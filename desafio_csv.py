import cv2
import sys
import csv
import numpy as np

fp = open('Results.csv', mode='w')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

VIDEO = 'D:/igor-/Documents/Programmazione/Alura/Data Science/69. Visão Computacional detecção de movimento com OpenCV/Ponte.mp4'

tipos = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
tipo = tipos[4]

## knn = 3.2136415
## gmg = 5.4116322
## cnt = 2.2595218
## mog = 4.9706952
## mog2 = 3.2264092

def Subtractor(algorithm_type):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print('Erro - Insira uma nova informação')
    sys.exit(1)
    
cap = cv2.VideoCapture(VIDEO)
background_subtractor = []
for i, a in enumerate(tipos):
    print(i, a)
    background_subtractor.append(Subtractor(a))

# e1 = cv2.getTickCount()

def main():
    # frame_number = -1
    while(cap.isOpened):
        ok, frame = cap.read()
        
        if not ok:
            print('Frames acabaram')
            break
        
        # frame_number = frame_number + 1
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
        
        knn = background_subtractor[0].apply(frame)
        gmg = background_subtractor[1].apply(frame)
        cnt = background_subtractor[2].apply(frame)
        mog = background_subtractor[3].apply(frame)
        mog2 = background_subtractor[4].apply(frame)
        
        gmgCount = np.count_nonzero(gmg)
        mogCount = np.count_nonzero(mog)
        mog2Count = np.count_nonzero(mog2)
        knnCount = np.count_nonzero(knn)
        cntCount = np.count_nonzero(cnt)

        writer.writerow({'Frame': 'GMG', 'Pixel Count': gmgCount})
        writer.writerow({'Frame': 'MOG', 'Pixel Count': mogCount})
        writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog2Count})
        writer.writerow({'Frame': 'KNN', 'Pixel Count': knnCount})
        writer.writerow({'Frame': 'CNT', 'Pixel Count': cntCount})
        
        cv2.imshow('Original', frame)
        cv2.imshow('KNN', knn)
        cv2.imshow('GMG', gmg)
        cv2.imshow('CNT', cnt)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
        # e2 = cv2.getTickCount()
        # t = (e2 - e1) / cv2.getTickFrequency()
        # print(t)
main()