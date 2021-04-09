import os
import shutil
from os.path import isfile, join

import cv2

train_path = r'C:\job\data\segmentation_sofa\for annotation\train'
val_path = r'C:\job\data\segmentation_sofa\for annotation\val'
n = 2


def cut(file, video_path='data/videos/', outpath='data/images/'):

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    cap = cv2.VideoCapture(os.path.join(video_path, file))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not i % n:
            cv2.imwrite(outpath + '{0}.jpg'.format(str(i // n)), frame)

        i += 1
    cap.release()

file = 'mot.webm'
cut(file)