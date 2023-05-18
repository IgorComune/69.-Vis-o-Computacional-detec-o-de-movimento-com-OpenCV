import numpy as np 
import cv2
import sys 

VIDEO = 'D:/igor-/Documents/Programmazione/Alura/Data Science/69. Visão Computacional detecção de movimento com OpenCV/Ponte.mp4'

algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
algorithm_type = algorithm_types[3]

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilate'), iterations=2)
    if filter == 'combine':
        closing =cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation    
    
def Subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Detector inválido')
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO)
background_subtractor = Subtractor(algorithm_type)

def main():
    while (cap.isOpened):
        ok, frame = cap.read()
        if not ok:
            print('Frames acabaram!')
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        mask = background_subtractor.apply(frame)
        mask_filter = Filter(mask, 'combine')
        cars_after_mask = cv2.bitwise_and(frame, frame, mask=mask_filter)
        
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', cars_after_mask)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            break

main()